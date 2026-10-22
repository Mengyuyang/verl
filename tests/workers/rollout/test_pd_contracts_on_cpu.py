# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exercise production PD methods without importing vLLM or starting Ray.

AST extraction deliberately limits these to Python contract tests. Hardware,
worker initialization and connector integration require separate validation.
"""

from __future__ import annotations

import ast
import asyncio
import logging
import socket
import types
import uuid
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[3]
SERVER = "verl/workers/rollout/vllm_rollout/vllm_async_server.py"
REPLICA = "verl/workers/rollout/vllm_rollout/vllm_pd_replica.py"


def _load_functions(path, names, owner=None, **namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    body = tree.body if owner is None else next(n for n in tree.body if getattr(n, "name", None) == owner).body
    nodes = [n for n in body if getattr(n, "name", None) in names]
    assert len(nodes) == len(names)
    for node in nodes:
        node.decorator_list = []
    # Postpone annotations, as the optional runtime types are not imported here.
    module = ast.Module(body=[*ast.parse("from __future__ import annotations").body, *nodes], type_ignores=[])
    scope = {"asyncio": asyncio, "uuid": uuid, "logger": logging.getLogger(__name__), **namespace}
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), scope)
    return types.SimpleNamespace(**{name: scope[name] for name in names})


def _remote(function):
    async def invoke(*args, **kwargs):
        return function(*args, **kwargs)

    return types.SimpleNamespace(remote=invoke)


@pytest.mark.parametrize("fields", [[], ["prompt_ids"]])
def test_decode_router_only_receives_declared_fields(fields):
    functions = _load_functions(SERVER, ["_select_decode_peer"], "vLLMHttpServer")
    calls = []
    peer = object()

    def acquire(request_id, **kwargs):
        calls.append((request_id, kwargs))
        assert set(kwargs) == set(fields)
        return "d", peer

    stub = types.SimpleNamespace(
        _pd_router=types.SimpleNamespace(acquire_server=_remote(acquire)),
        _pd_router_acquire_fields=fields,
        _pd_decode_candidates=[types.SimpleNamespace(key="d", handle=peer)],
        _pd_decode_peers=[peer],
        _pd_peer_idx=0,
    )
    result = asyncio.run(functions._select_decode_peer(stub, "session", [1, 2]))
    assert result == (0, peer, "d")
    assert calls == [("session", {"prompt_ids": [1, 2]} if fields else {})]
    assert stub._pd_peer_idx == 0  # No fallback due to an unexpected keyword argument.


@pytest.mark.parametrize("outcome", ["success", "prefill_error", "decode_error", "cancelled"])
@pytest.mark.parametrize("release_fields", [[], ["request_id"]])
def test_dispatch_releases_reservation_on_every_exit(outcome, release_fields):
    helper = _load_functions(SERVER, ["_build_mooncake_decode_kv_params"])
    dispatch = _load_functions(
        SERVER,
        ["_pd_dispatch"],
        "vLLMHttpServer",
        _build_mooncake_decode_kv_params=helper._build_mooncake_decode_kv_params,
    )._pd_dispatch
    release = AsyncMock()
    decode = AsyncMock(return_value="generated")
    if outcome == "decode_error":
        decode.side_effect = RuntimeError("decode failed")
    peer = types.SimpleNamespace(generate=types.SimpleNamespace(remote=decode))
    prefill_params = {"remote_block_ids": [[1, 2]], "remote_engine_id": "original"}
    generate = AsyncMock(return_value=types.SimpleNamespace(extra_fields={"kv_transfer_params": prefill_params}))
    if outcome == "prefill_error":
        generate.side_effect = RuntimeError("prefill failed")
    elif outcome == "cancelled":
        generate.side_effect = asyncio.CancelledError()
    select = AsyncMock(return_value=(0, peer, "d"))
    stub = types.SimpleNamespace(
        _select_decode_peer=select,
        _pd_router=types.SimpleNamespace(release_server=types.SimpleNamespace(remote=release)),
        _pd_router_release_fields=release_fields,
        _disaggregation_kv_transfer_config={"kv_connector": "MooncakeConnectorV1"},
        _pd_prefill_engine_id="fallback",
        _pd_prefill_side_channel_port=12345,
        generate=generate,
    )

    async def run():
        return await dispatch(stub, [1, 2], {"max_tokens": 20}, "turn", session_id="session")

    if outcome == "success":
        assert asyncio.run(run()) == "generated"
        params = decode.call_args.kwargs["kv_transfer_params"]
        assert params["remote_block_ids"] == [[1, 2]]
        assert params["remote_engine_id"] == "original"
        assert params["do_remote_prefill"] is True
        assert prefill_params == {"remote_block_ids": [[1, 2]], "remote_engine_id": "original"}
    else:
        exception = asyncio.CancelledError if outcome == "cancelled" else RuntimeError
        with pytest.raises(exception):
            asyncio.run(run())
    select.assert_awaited_once_with("session", [1, 2])
    release.assert_awaited_once_with(server_id="d", **({"request_id": "session"} if release_fields else {}))


@pytest.mark.parametrize("blocks", [None, [], (), [[]], [[], []], ([], [])])
def test_mooncake_without_transferable_blocks_computes_locally(blocks):
    build = _load_functions(SERVER, ["_build_mooncake_decode_kv_params"])._build_mooncake_decode_kv_params
    original = {"remote_block_ids": blocks, "do_remote_decode": True, "do_remote_prefill": True}
    params = build(original, remote_engine_id="p", remote_bootstrap_addr="addr", transfer_id="t")
    assert params["do_remote_prefill"] is False
    assert params["do_remote_decode"] is False
    assert original["do_remote_decode"] is True


@pytest.mark.parametrize("blocks", [[0], [[0]], [[], [0]], ([], [1, 2])])
def test_mooncake_preserves_block_zero_and_mixed_groups(blocks):
    build = _load_functions(SERVER, ["_build_mooncake_decode_kv_params"])._build_mooncake_decode_kv_params
    params = build({"remote_block_ids": blocks}, remote_engine_id="p", remote_bootstrap_addr="addr", transfer_id="t")
    assert params["do_remote_prefill"] is True
    assert params["do_remote_decode"] is False
    assert params["remote_block_ids"] == blocks


def test_pd_resume_resumes_all_engines_before_any_admission():
    resume = _load_functions(REPLICA, ["resume_generation"], "vLLMPDReplica").resume_generation
    events = []

    def server(name):
        return types.SimpleNamespace(
            resume_engine_generation=_remote(lambda: events.append(("resume", name))),
            open_submission_gate=_remote(lambda: events.append(("open", name))),
        )

    asyncio.run(resume(types.SimpleNamespace(servers=[server("p"), server("d0"), server("d1")])))
    assert events == [(action, name) for action in ("resume", "open") for name in ("p", "d0", "d1")]


def test_pd_resume_failure_keeps_all_gates_closed():
    resume = _load_functions(REPLICA, ["resume_generation"], "vLLMPDReplica").resume_generation
    open_gate = AsyncMock()
    server = types.SimpleNamespace(
        resume_engine_generation=types.SimpleNamespace(remote=AsyncMock(side_effect=RuntimeError("resume failed"))),
        open_submission_gate=types.SimpleNamespace(remote=open_gate),
    )
    with pytest.raises(RuntimeError, match="resume failed"):
        asyncio.run(resume(types.SimpleNamespace(servers=[server])))
    open_gate.assert_not_awaited()


def test_ascend_connector_config_carries_asymmetric_tp():
    build = _load_functions(REPLICA, ["_build_kv_transfer_config"], "vLLMPDReplica", get_device_name=lambda: "npu")
    cfg = build._build_kv_transfer_config(
        "prefill", "p", "mooncake", use_ascend_mooncake_v1=True, kv_port=39000, prefill_tp=4, decode_tp=2
    )
    assert cfg["kv_connector"] == "MooncakeConnectorV1"
    assert cfg["kv_buffer_device"] == "npu"
    assert cfg["kv_port"] == 39000
    assert cfg["kv_connector_extra_config"] == {
        "prefill": {"dp_size": 1, "tp_size": 4},
        "decode": {"dp_size": 1, "tp_size": 2},
    }
    with pytest.raises(ValueError, match="requires kv_port"):
        build._build_kv_transfer_config("decode", "d", "mooncake", use_ascend_mooncake_v1=True)


def test_port_ranges_are_contiguous_and_disjoint():
    reserve = _load_functions(
        REPLICA, ["_reserve_port_range"], "vLLMPDReplica", socket=socket, is_valid_ipv6_address=lambda _: False
    )._reserve_port_range
    sockets = []
    try:
        base1, first = reserve("127.0.0.1", 2)
        sockets.extend(first)
        base2, second = reserve("127.0.0.1", 3)
        sockets.extend(second)
        assert {base1, base1 + 1}.isdisjoint(range(base2, base2 + 3))
        assert [s.getsockname()[1] for s in first] == [base1, base1 + 1]
        assert [s.getsockname()[1] for s in second] == list(range(base2, base2 + 3))
    finally:
        for sock in sockets:
            sock.close()
    with pytest.raises(ValueError, match="positive"):
        reserve("127.0.0.1", 0)


def test_sleep_drains_then_disconnects_all_peers_before_sleep():
    sleep = _load_functions(REPLICA, ["sleep"], "vLLMPDReplica").sleep
    events = []

    def server(name):
        return types.SimpleNamespace(
            wait_for_requests_to_drain=_remote(lambda: events.append(("drain", name))),
            collective_rpc=_remote(lambda **kw: events.append((kw["method"], name))),
            sleep=_remote(lambda: events.append(("sleep", name))),
        )

    prefill, decode = server("p"), server("d")
    stub = types.SimpleNamespace(
        get_request_server_endpoints=lambda: [("p", prefill)],
        servers=[prefill, decode],
        _is_ascend_platform=lambda: True,
    )
    asyncio.run(sleep(stub))
    assert events == [
        ("drain", "p"),
        ("disconnect_kv_transfer_peers", "p"),
        ("disconnect_kv_transfer_peers", "d"),
        ("sleep", "p"),
        ("sleep", "d"),
    ]
