# Ascend PD integration: final-local fork review

Last updated: 10/23/2026.

## Scope and source

This revision follows `uni-agent1013-dr-final-local.rar`, SHA256
`8AF98361FE5C19978106076F1B637687DC179F3A72ED656AD617FDA54124F9D5`.
Its verl baseline is `263653e749306287a5032140cce3d7833c1959d0`;
this PR targets the fork main synchronized to upstream
`d430157d15516d89e0cc853fd10fcd21d9234c44`.

Compared with the previous PR revision, the final source reduces the integration
surface. The runtime changes are limited to `llm_server.py`,
`vllm_async_server.py`, and `vllm_pd_replica.py`. The strategy adapter moves
to Uni-Agent. PD-specific dataclass/YAML extensions, generic replica endpoint
APIs and the Prometheus label merger are restored to upstream.

Basic single-node 1P+ND and asymmetric TP already existed in upstream.
This change adds Ascend connector/ports, optional standard-router integration,
Ascend metrics discovery and KV lifecycle hooks; it does not invent PD itself.

## Configuration migration

```yaml
actor_rollout_ref:
  rollout:
    name: vllm
    tensor_model_parallel_size: 2
    data_parallel_size: 1
    pipeline_model_parallel_size: 1
    disaggregation:
      enabled: true
      prefill_replicas: 1
      decode_replicas: 1
      decode_tensor_model_parallel_size: 2
      transfer_backend: mooncake
    custom:
      pd_router_config_path: /absolute/path/to/session_sticky.yaml
```

This illustrative topology occupies four devices, not two. Complete the
model/training/resource settings for the actual environment.

- Use `rollout.custom.pd_router_config_path`, not the removed
  `disaggregation.decode_policy.router_config_path`.
- The existing upstream router factory loads a filesystem YAML path, not the
  former PD adapter's `pkg://...` URL. Make it accessible to Ray workers.
- The YAML sets `router_class: uni_agent.rollout.router_balancer.SelectorRouter`
  and a `selector_class` plus selector parameters.
- No configured path means **round-robin**. Optional custom PD routing only
  activates on Ascend. GPU PD retains upstream selection behavior.
- Role-local memory/batching extensions from the previous revision are no
  longer supported; P/D share normal rollout settings apart from their TP.
- The private archived baseline launcher and launcher tests still refer to old
  schema/missing deps; they are not shipped or claimed to run.

## Code walkthrough

1. `vllm_pd_replica.py` chooses MooncakeConnectorV1 on Ascend, supplies both
   TP sizes and per-engine KV ports, and reserves consecutive ranges. GPU keeps
   its original port-allocation/connector path.
2. `vllm_async_server.py::set_pd_peer` builds the configured router through
   `get_router_handle`; policy implementation stays in Uni-Agent.
3. `_pd_dispatch` reserves D, runs P, builds connector-specific metadata,
   runs D, and releases in finally. GPU Mooncake retains upstream handshake
   semantics; Ascend V1 preserves P block metadata and computes locally when
   no transferable blocks are available. NIXL rejects missing metadata.
4. Cache-invalidating server operations clear router affinity. Shared replica
   sleep invokes an optional `_before_server_sleep` hook after its drain;
   the PD override disconnects all Ascend KV peers before sleep.
5. `llm_server.py` exposes P+D metrics addresses only for Ascend PD. RLInsight
   receives role labels. Prometheus receives the addresses but this revision
   does **not** add the former custom static role-label merger.

## Porting safeguards beyond the supplied source

- Preserve current main's quantization implementation instead of restoring
  the archive baseline's removed QAT path.
- Honor `require_acquire_fields` and `require_release_fields`; custom native
  routers need not accept prompt_ids or request_id at release.
- Detect nested empty block groups while retaining valid block ID zero.
- Keep PD resume in the subclass: all independent engines resume successfully
  before any submission gate opens. Unified DP retains upstream sequencing.
- Close already acquired sockets if a later port-range reservation fails.
- Extend CPU contracts and retain the source's minimal integration tests.

## Validation and remaining merge gates

```bash
python -m pytest -q tests/workers/rollout/test_pd_contracts_on_cpu.py --noconftest
# 32 passed
python -m pytest -q tests/workers/rollout/test_pd_minimal_custom_on_cpu.py tests/workers/rollout/test_pd_disaggregation.py tests/workers/rollout/test_vllm_pd_disaggregation_on_cpu.py --noconftest --maxfail=1
```

The second command failed during Windows PyTorch collection with WinError 1114
loading c10.dll. It is not counted as passing. AST contracts execute production
methods with fake remote calls; they do not prove engine startup or transport.
Hardware and complete application validation remain pending.

Changed-file Ruff, formatting, mypy, generated-config verification and the
other pre-commit checks passed, except two local Windows hook failures: the
test-structure allowlist uses POSIX paths, and the compileall hook selected an
older Python that cannot parse upstream match statements. Explicit Python 3.12
compileall passed, as did the structure check with the same three upstream
allowlisted files expressed using Windows separators. No upstream hook or
allowlist was changed to mask these failures.

Before merge:

- Resolve optional session-sticky FIFO token-release accounting. Its new
  adapter still ignores request identity at release; moving it does not fix it.
- Verify whole P-to-D drain, abort/resume, partial rollout, weight synchronization,
  sleep/wake and the separate `release_kv_cache` path. The shared sleep drain
  still observes the first engine; it does not prove all D work completed.
- Verify matched dependencies providing MooncakeConnectorV1 and
  `disconnect_kv_transfer_peers`. Final-local contains no `deps` tree.
  The previous archive's empty-group/completion fixes cannot be assumed to be
  present in the installed runtime.
- Releasing reserved ports before child bind still leaves cross-job races.
- Real NPU performance and non-PD/GPU regressions require target-environment tests.

The source now adds an NPU guard to its Megatron scalar-gradient workaround.
It is still excluded from this PD-focused PR as a separate training-memory
change requiring numeric/memory testing.

Related [upstream PR #7616](https://github.com/verl-project/verl/pull/7616)
was checked on 2026-10-23: closed, not merged. It is reference work, not a
prerequisite or proof of upstream support. This remains an owner-requested
fork-only draft; no duplicate upstream PR is being submitted.
An open-PR search also found [#7859](https://github.com/verl-project/verl/pull/7859),
which namespaces vLLM's internal TCPStore port search. This change reserves
Mooncake per-rank KV/handshake ranges instead; it neither includes nor replaces
that independent internal-port fix.

See [companion Uni-Agent PR](https://github.com/Mengyuyang/uni-agent/pull/1).
AI assistance was used; human line-by-line review and hardware validation remain
required.
