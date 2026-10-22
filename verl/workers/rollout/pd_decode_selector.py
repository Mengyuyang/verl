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
"""PD decode routing: adapt the uni-agent selector plugins to verl's router.

Background (2026-10-08, O2 migration step)
------------------------------------------
Historically the PD decode side used a verl-local plugin contract
``select(req, candidates) -> DecodeCandidate`` loaded by a second FQN factory
(``get_pd_decode_selector``). Upstream verl has since generalised the mechanism
into ``verl/workers/rollout/router.py`` (``RequestLoadBalancer`` protocol +
FQN/YAML factory ``get_router_handle``).

This module stops maintaining that second factory. It keeps

* the **existing strategy classes** (``uni_agent.rollout.decode_selectors.*``), and
* the **existing YAML files** (``pkg://uni_agent.rollout.configs/<policy>.yaml``),

untouched, and exposes them through the upstream protocol:

    YAML ``router_class: uni_agent.rollout.decode_selectors.<Policy>``
      -> ``SelectorAsBalancer`` (this adapter, a plain class wrapped by Ray)
      -> ``<Policy>.select({"routing_key": ..., "prompt_ids": ...}, candidates)``

YAMLs whose ``router_class`` already implements the upstream protocol
(``acquire_server`` / ``release_server``) are passed straight to Ray **with the
whole YAML dict as their ``config`` argument** (the upstream #7115 convention,
see ``verl/workers/rollout/router.py::_create_plugin_extension``), so an
upstream-style balancer (e.g. ``uni_agent.agent_aware_router.balancer.KVCAwareBalancer``)
can be swapped in by pointing ``router_config_path`` at its own YAML.
"""

from __future__ import annotations

import importlib
import importlib.resources
import logging
from dataclasses import dataclass
from typing import Any, Sequence

import ray
from omegaconf import OmegaConf

from verl.utils.import_utils import load_class_from_fqn

logger = logging.getLogger(__name__)

_PKG_PREFIX = "pkg://"
_FILE_PREFIX = "file://"
_CLASS_KEYS = ("router_class", "class")


@dataclass
class DecodeCandidate:
    """Decode node dispatch metadata (unchanged public shape)."""

    key: str
    handle: Any


def resolve_selector_config(spec: str) -> dict:
    """Load a selector/router YAML into a plain dict.

    ``pkg://<package>/<resource>`` keeps the legacy importlib.resources
    resolution used by ``run_train_swepd_dr.sh``; every other form is delegated
    to the upstream loader (``router._load_router_yaml``), so filesystem
    resolution lives in exactly one place.
    """
    if spec.startswith(_PKG_PREFIX):
        resource = spec[len(_PKG_PREFIX) :]
        parts = resource.split("/")
        last_error: Exception | None = None
        tried: list[str] = []
        for split_at in range(len(parts), 0, -1):
            package = ".".join(parts[:split_at])
            tried.append(package)
            try:
                importlib.import_module(package)
            except Exception as exc:  # noqa: BLE001 - progressively shorten the package prefix
                last_error = exc
                continue
            node = importlib.resources.files(package)
            for part in parts[split_at:]:
                node = node / part
            cfg = OmegaConf.create(node.read_text())
            return OmegaConf.to_container(cfg, resolve=True) or {}
        raise ValueError(f"cannot resolve pkg:// resource: {resource!r} (tried: {tried}; last error: {last_error!r})")

    from verl.workers.rollout.router import _load_router_yaml

    return _load_router_yaml(spec[len(_FILE_PREFIX) :] if spec.startswith(_FILE_PREFIX) else spec)


def selector_class_from_config(cfg: dict) -> tuple[str, dict]:
    """Return ``(router_class_fqn, ctor_kwargs)`` from a selector config dict."""
    fqn = next((cfg.get(key) for key in _CLASS_KEYS if cfg.get(key)), None)
    if not fqn:
        raise ValueError(f"selector YAML must set one of {list(_CLASS_KEYS)}")
    kwargs = {k: v for k, v in cfg.items() if k not in _CLASS_KEYS}
    return str(fqn), kwargs


def implements_router_protocol(cls: type) -> bool:
    """True when ``cls`` already implements verl's ``RequestLoadBalancer``."""
    return callable(getattr(cls, "acquire_server", None)) and callable(getattr(cls, "release_server", None))


class SelectorAsBalancer:
    """Expose a ``select(req, candidates)`` selector as a ``RequestLoadBalancer``.

    The legacy contract is one-shot: ``select(req, candidates)`` returns the
    chosen candidate and ``release(index)`` is called when the request is done.
    The router protocol is pool-scoped: the server set is injected once (and may
    change through ``add_servers`` / ``remove_servers``), and every decision is
    ``acquire_server(request_id, **extra) -> (server_id, handle)``.

    The adapter keeps the insertion order of the pool so the legacy
    index-based ``release()`` keeps working, and forwards ``prompt_ids`` (the
    field declared through ``require_acquire_fields``) as-is.
    """

    def __init__(
        self,
        servers: dict[str, Any],
        config: dict[str, Any] | None = None,
        router_config_path: str | None = None,
    ) -> None:
        cfg = dict(config or {})
        if router_config_path and not cfg:
            cfg = resolve_selector_config(router_config_path)
        fqn, kwargs = selector_class_from_config(cfg)
        self._inner = load_class_from_fqn(fqn, description="PD decode selector")(**kwargs)
        self._servers: dict[str, Any] = dict(servers)
        self._order: list[str] = list(self._servers)
        logger.info(
            "SelectorAsBalancer: inner=%s servers=%d order=%s",
            fqn,
            len(self._order),
            self._order,
        )

    # -- RequestLoadBalancer protocol ---------------------------------------
    def require_acquire_fields(self) -> list[str]:
        return ["prompt_ids"]

    def require_release_fields(self) -> list[str]:
        return []

    def acquire_server(self, request_id: str, prompt_ids: Sequence[int] | None = None) -> tuple[str, Any]:
        candidates = [DecodeCandidate(key=sid, handle=self._servers[sid]) for sid in self._order]
        chosen = self._inner.select(
            {
                "routing_key": request_id,
                "prompt_ids": list(prompt_ids) if prompt_ids is not None else None,
            },
            candidates,
        )
        server_id = str(getattr(chosen, "key", chosen))
        if server_id not in self._servers:
            raise RuntimeError(f"selector returned unknown server: {server_id!r}")
        return server_id, self._servers[server_id]

    def release_server(self, server_id: str, request_id: str | None = None) -> None:
        release = getattr(self._inner, "release", None)
        if release is None or server_id not in self._order:
            return
        release(self._order.index(server_id))

    def add_servers(self, servers: dict[str, Any]) -> None:
        for server_id, handle in servers.items():
            if server_id not in self._servers:
                self._servers[server_id] = handle
                self._order.append(server_id)

    def remove_servers(self, server_ids: list[str]) -> None:
        for server_id in server_ids:
            self._servers.pop(server_id, None)
            if server_id in self._order:
                self._order.remove(server_id)

    def get_all_servers(self) -> list[str]:
        return list(self._order)

    def get_status(self) -> dict:
        return {
            "inner": type(self._inner).__name__,
            "servers": list(self._order),
            "stats": getattr(self._inner, "stats", None),
        }

    def clear_sticky_cache(self) -> dict:
        reset = getattr(self._inner, "reset", None)
        if callable(reset):
            reset()
        return {"inner": type(self._inner).__name__, "cleared": callable(reset)}

    def get_total_inflight(self) -> int:
        for attr in ("_pending_requests", "_pending_tokens"):
            pending = getattr(self._inner, attr, None)
            if isinstance(pending, list | tuple):
                return int(sum(pending))
        return 0


def build_pd_router(
    servers: dict[str, Any],
    router_config_path: str | None,
    full_determinism: bool = False,
) -> tuple[Any, str]:
    """Build the PD decode router.

    Returns ``(actor_handle, kind)`` where ``kind`` is one of
    ``"builtin"`` / ``"plugin"`` / ``"adapter"`` for logging.

    * no config -> upstream default (sticky session + least in-flight)
    * config whose class implements the router protocol -> used directly
    * legacy ``select(req, candidates)`` config -> wrapped by ``SelectorAsBalancer``
    """
    from verl.workers.rollout.router import get_router_handle

    if not router_config_path:
        return get_router_handle(servers, None, full_determinism=full_determinism), "builtin"

    cfg = resolve_selector_config(router_config_path)
    fqn, _ = selector_class_from_config(cfg)
    cls = load_class_from_fqn(fqn, description="PD decode selector")
    if implements_router_protocol(cls):
        logger.info("PD decode router: using router-protocol plugin %s", fqn)
        # Upstream #7115 convention: the whole YAML dict is handed to the plugin as
        # its ``config`` argument, so nested blocks (e.g. ``strategy._target_`` /
        # ``collector``) stay structured and the plugin parses them itself.
        # Legacy ``select(req, candidates)`` selectors keep the flat-kwargs
        # convention through ``SelectorAsBalancer`` below.
        return ray.remote(cls).remote(servers, cfg), "plugin"
    logger.info("PD decode router: wrapping legacy selector %s with SelectorAsBalancer", fqn)
    return ray.remote(SelectorAsBalancer).remote(servers, cfg), "adapter"
