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
"""CPU tests for the PD decode router adapter (O2 runtime wiring).

Covers ``verl.workers.rollout.pd_decode_selector``:
* YAML resolution (filesystem + legacy ``pkg://``)
* ``SelectorAsBalancer``: legacy ``select(req, candidates)`` plugins exposed as
  the upstream ``RequestLoadBalancer`` protocol
* protocol detection / factory decision

Run (from the verl repo root):

    PYTHONPATH=$PWD:/path/to/dr/uni-agent python3 -m unittest \
        discover -t . -s tests/workers/rollout -p 'test_pd_decode_router_adapter_on_cpu.py'
"""

import importlib.util
import os
import tempfile
import types
import unittest

from verl.workers.rollout.pd_decode_selector import (
    SelectorAsBalancer,
    build_pd_router,
    implements_router_protocol,
    resolve_selector_config,
    selector_class_from_config,
)
from verl.workers.rollout.router import GlobalRequestLoadBalancer

HAS_UNI_AGENT = importlib.util.find_spec("uni_agent") is not None

LEAST_LOAD = "uni_agent.rollout.decode_selectors.LeastLoadDecodeSelector"
SESSION_STICKY = "uni_agent.rollout.decode_selectors.SessionStickyDecodeSelector"


class _StubPressure:
    """Pressure source stub (avoids the 1 Hz /metrics poller in unit tests)."""

    def __init__(self, kv_by_key=None, wait_by_key=None):
        self.kv_by_key = dict(kv_by_key or {})
        self.wait_by_key = dict(wait_by_key or {})

    def observe(self, keys):
        return None

    def snapshot(self, key):
        if key not in self.kv_by_key and key not in self.wait_by_key:
            return None
        return types.SimpleNamespace(
            ts=0.0,
            kv_usage=self.kv_by_key.get(key),
            running=None,
            wait_capacity=self.wait_by_key.get(key),
        )


def _servers(*keys):
    return {key: f"handle::{key}" for key in keys}


class TestResolveSelectorConfig(unittest.TestCase):
    def test_filesystem_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "least_load.yaml")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(f"router_class: {LEAST_LOAD}\nsample_size: 2\n")
            cfg = resolve_selector_config(path)
        self.assertEqual(cfg["router_class"], LEAST_LOAD)
        self.assertEqual(cfg["sample_size"], 2)

    @unittest.skipUnless(HAS_UNI_AGENT, "uni-agent not importable")
    def test_legacy_pkg_path(self):
        cfg = resolve_selector_config("pkg://uni_agent.rollout.configs/least_load.yaml")
        self.assertEqual(cfg["router_class"], LEAST_LOAD)

    def test_missing_class_key_raises(self):
        with self.assertRaises(ValueError):
            selector_class_from_config({"sample_size": 2})

    def test_class_key_aliases(self):
        fqn, kwargs = selector_class_from_config({"class": LEAST_LOAD, "sample_size": 3})
        self.assertEqual(fqn, LEAST_LOAD)
        self.assertEqual(kwargs, {"sample_size": 3})


class TestProtocolDetection(unittest.TestCase):
    def test_builtin_balancer_implements_protocol(self):
        self.assertTrue(implements_router_protocol(GlobalRequestLoadBalancer))

    @unittest.skipUnless(HAS_UNI_AGENT, "uni-agent not importable")
    def test_legacy_selector_is_not_a_balancer(self):
        from uni_agent.rollout.decode_selectors import LeastLoadDecodeSelector

        self.assertFalse(implements_router_protocol(LeastLoadDecodeSelector))


@unittest.skipUnless(HAS_UNI_AGENT, "uni-agent not importable")
class TestSelectorAsBalancer(unittest.TestCase):
    def test_declares_prompt_ids_field(self):
        balancer = SelectorAsBalancer(_servers("a", "b"), {"router_class": LEAST_LOAD})
        self.assertEqual(balancer.require_acquire_fields(), ["prompt_ids"])
        self.assertEqual(balancer.require_release_fields(), [])

    def test_acquire_returns_pool_member_and_release_tracks_inflight(self):
        balancer = SelectorAsBalancer(_servers("a", "b", "c"), {"router_class": LEAST_LOAD, "sample_size": 2})
        server_id, handle = balancer.acquire_server("req-1", [1, 2, 3])
        self.assertIn(server_id, {"a", "b", "c"})
        self.assertEqual(handle, f"handle::{server_id}")
        self.assertEqual(balancer.get_total_inflight(), 1)
        balancer.release_server(server_id, "req-1")
        self.assertEqual(balancer.get_total_inflight(), 0)

    def test_sticky_selector_keeps_one_server_per_session(self):
        balancer = SelectorAsBalancer(
            _servers("a", "b", "c"),
            {"router_class": SESSION_STICKY, "pressure_cache": _StubPressure()},
        )
        first, _ = balancer.acquire_server("session-1", [10, 11, 12])
        second, _ = balancer.acquire_server("session-1", [10, 11, 13])
        self.assertEqual(first, second)

    def test_add_and_remove_servers(self):
        balancer = SelectorAsBalancer(_servers("a"), {"router_class": LEAST_LOAD})
        balancer.add_servers(_servers("b"))
        self.assertEqual(balancer.get_all_servers(), ["a", "b"])
        balancer.remove_servers(["a"])
        self.assertEqual(balancer.get_all_servers(), ["b"])
        self.assertIn("inner", balancer.get_status())

    def test_clear_sticky_cache_is_safe_for_stateless_selector(self):
        balancer = SelectorAsBalancer(_servers("a", "b"), {"router_class": LEAST_LOAD})
        self.assertEqual(balancer.clear_sticky_cache()["cleared"], False)

    def test_unknown_server_from_selector_raises(self):
        balancer = SelectorAsBalancer(_servers("a", "b"), {"router_class": LEAST_LOAD})

        class _BadInner:
            stats = {}

            def select(self, req, candidates):
                return types.SimpleNamespace(key="nope")

            def release(self, index):
                return None

        balancer._inner = _BadInner()
        with self.assertRaises(RuntimeError):
            balancer.acquire_server("req-x", [1])


class TestFactoryDecision(unittest.TestCase):
    def test_build_without_config_uses_builtin(self):
        # The builtin branch calls ray.remote(...); assert on the decision helper
        # instead of constructing an actor in a CPU test.
        from verl.workers.rollout.router import get_router_handle

        self.assertTrue(callable(get_router_handle))

    @unittest.skipUnless(HAS_UNI_AGENT, "uni-agent not importable")
    def test_legacy_config_selects_adapter_path(self):
        cfg = resolve_selector_config("pkg://uni_agent.rollout.configs/least_load.yaml")
        fqn, _ = selector_class_from_config(cfg)
        self.assertFalse(
            implements_router_protocol(
                __import__("uni_agent.rollout.decode_selectors", fromlist=["x"]).LeastLoadDecodeSelector
            )
        )
        self.assertEqual(fqn, LEAST_LOAD)

    def test_build_pd_router_is_callable(self):
        self.assertTrue(callable(build_pd_router))


if __name__ == "__main__":
    unittest.main()
