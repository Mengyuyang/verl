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
"""CPU tests for the reduced-surface Ascend P/D implementation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from verl.workers.config import RolloutConfig


def test_ascend_mooncake_v1_config():
    pytest.importorskip("vllm")
    from verl.workers.rollout.vllm_rollout.vllm_pd_replica import vLLMPDReplica

    cfg = vLLMPDReplica._build_kv_transfer_config(
        role="decode",
        engine_id="decode-0",
        transfer_backend="mooncake",
        mooncake_protocol="tcp",
        use_ascend_mooncake_v1=True,
        kv_port=12345,
        prefill_tp=4,
        decode_tp=4,
    )
    assert cfg["kv_connector"] == "MooncakeConnectorV1"
    assert cfg["kv_role"] == "kv_consumer"
    assert cfg["kv_port"] == 12345
    assert cfg["kv_connector_extra_config"] == {
        "prefill": {"dp_size": 1, "tp_size": 4},
        "decode": {"dp_size": 1, "tp_size": 4},
    }


def test_pd_metrics_endpoints_include_prefill_and_decode():
    pytest.importorskip("vllm")
    from verl.workers.rollout.vllm_rollout.vllm_pd_replica import vLLMPDReplica

    replica = object.__new__(vLLMPDReplica)
    replica._prefill_servers = [object()]
    replica._decode_servers = [object(), object()]
    replica._prefill_server_addresses = ["prefill:8000"]
    replica._decode_server_addresses = ["decode0:8000", "decode1:8000"]

    assert replica.get_metrics_server_endpoints() == [
        ("prefill:8000", {"pd_role": "prefill", "pd_index": 0}),
        ("decode0:8000", {"pd_role": "decode", "pd_index": 0}),
        ("decode1:8000", {"pd_role": "decode", "pd_index": 1}),
    ]


def test_mooncake_empty_prefill_falls_back_to_local_decode():
    pytest.importorskip("vllm")
    from verl.workers.rollout.vllm_rollout.vllm_async_server import (
        _build_mooncake_decode_kv_params,
    )

    cfg = _build_mooncake_decode_kv_params(
        {"remote_block_ids": []},
        remote_engine_id="prefill",
        remote_bootstrap_addr="http://127.0.0.1:12345",
        transfer_id="transfer",
    )
    assert cfg["do_remote_prefill"] is False
    assert cfg["do_remote_decode"] is False


def test_non_pd_resume_keeps_upstream_order():
    pytest.importorskip("vllm")
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

    servers = [MagicMock(), MagicMock()]
    for server in servers:
        server.resume_engine_generation.remote = AsyncMock()
        server.open_submission_gate.remote = AsyncMock()
        server.resume_generation.remote = AsyncMock()

    replica = object.__new__(vLLMReplica)
    replica.config = RolloutConfig(name="vllm")
    replica.servers = servers
    asyncio.run(vLLMReplica.resume_generation(replica))

    servers[0].resume_engine_generation.remote.assert_awaited_once()
    servers[1].resume_engine_generation.remote.assert_not_awaited()
    for server in servers:
        server.open_submission_gate.remote.assert_awaited_once()
        server.resume_generation.remote.assert_not_awaited()


def test_shared_sleep_runs_pre_sleep_hook():
    pytest.importorskip("vllm")
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

    server = MagicMock()
    server.wait_for_requests_to_drain.remote = AsyncMock()
    server.sleep.remote = AsyncMock()
    replica = object.__new__(vLLMReplica)
    replica.servers = [server]
    replica._before_server_sleep = AsyncMock()

    asyncio.run(vLLMReplica.sleep(replica))

    server.wait_for_requests_to_drain.remote.assert_awaited_once()
    replica._before_server_sleep.assert_awaited_once()
    server.sleep.remote.assert_awaited_once()
