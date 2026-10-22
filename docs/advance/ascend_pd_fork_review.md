# Ascend PD integration: fork review notes

Last updated: 10/22/2026.

## Scope

This branch ports a colleague-provided PD snapshot onto current upstream `main`
(`831c5d86e58972ef2e26f00345fde1be700611cf`). The corrected source is
`uni-agent1013-dr.rar` (SHA256
`60E978770C670EF2658D6C8FF08DF5B002647EAC0F0FDFBEB717EA3B435CCE00`),
with verl baseline `2035f58dcb06d8cad4caee107b30f4045d724367`.
It is intended for review in
`Mengyuyang/verl`, with that fork's synchronized `main` as the merge target.
The baseline already supports single-node 1P+ND and asymmetric TP. This change
adds Ascend connector configuration, optional decode routing plugins, role
overrides, metrics labels, and sleep-path connection cleanup.

Related upstream work: [verl-project/verl#7616](https://github.com/verl-project/verl/pull/7616).
This is not another submission to upstream. That PR also implements Ascend PD,
but exposes built-in `decode_policy.type` strategies; this snapshot uses
`decode_policy.router_config_path` to adapt external selectors to the existing
`RequestLoadBalancer` protocol. Coordinate with that work before any upstream PR.

## Read the changes in this order

1. `verl/workers/config/disaggregation.py` and the rollout/reward YAML files:
   choose P/D sizes, role memory and scheduler limits, and an optional router.
2. `verl/workers/rollout/vllm_rollout/vllm_pd_replica.py`: construct P/D servers,
   reserve non-overlapping port ranges, build the Ascend `MooncakeConnectorV1`
   config, expose P-only request endpoints and P+D metric endpoints.
3. `verl/workers/rollout/pd_decode_selector.py`: load YAML and adapt legacy
   `select(req, candidates)` strategies or use a native router-protocol plugin.
   With no plugin configured, use verl's built-in sticky/least-inflight router.
4. `verl/workers/rollout/vllm_rollout/vllm_async_server.py`: reserve a D server,
   run P, preserve its KV transfer metadata, run D, release the reservation in
   `finally`; reset routing state when caches are invalidated.
5. `verl/workers/rollout/replica.py`, `llm_server.py`, and `utils.py`: keep request
   routing separate from metrics discovery; preserve existing Prometheus targets
   while adding role labels.

## Minimal configuration

An illustrative single-node layout uses 4 devices: 1P with TP=2 plus 1D with TP=2.
Set the remaining trainer/model/resource-pool settings for your deployment.

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
      decode_policy:
        router_config_path: null
      prefill_gpu_memory_utilization: 0.7
      decode_gpu_memory_utilization: 0.8
      prefill_engine_kwargs:
        max_num_batched_tokens: 8192
        max_num_seqs: 16
      decode_engine_kwargs:
        max_num_batched_tokens: 2048
        max_num_seqs: 32
```

The default router does not require Uni-Agent. Optional legacy strategies need
the matching Uni-Agent changes, for example
`pkg://uni_agent.rollout.configs/session_sticky.yaml`. Those files are not part
of this verl PR. A native protocol plugin must implement the full lifecycle
contract in `verl/workers/rollout/router.py`, not only acquire/release.

The P+D device footprint must fit one node. DP and PP remain 1. Role-specific
engine overrides are limited to `max_num_batched_tokens` and `max_num_seqs`.

## Porting changes beyond the original snapshot

- Preserve upstream top-k log-probability validation and do not restore removed
  QAT configuration.
- Preserve `_get_rollout_targets` for existing unlabeled-metrics callers/tests.
- Respect `require_acquire_fields` / `require_release_fields` when invoking a
  router. Passing `prompt_ids` unconditionally breaks the default router's
  signature and silently causes round-robin fallback.
- Add CPU-only contract tests for declared fields, KV metadata, release on
  success/error/cancellation, asymmetric TP, port ranges and sleep ordering.
- Port the corrected snapshot's local-compute fallback when P has no transferable
  KV blocks. Also handle nonempty outer containers of empty groups without
  mistaking block ID zero for an empty block list.
- Resume every independent P/D engine, then open all submission gates. Keep this
  in `vLLMPDReplica`; do not replace the unified DP replica's collective resume.
- Exclude the snapshot's global Megatron scalar-gradient monkey patch: it is a
  separate training-memory workaround, has no NPU guard, and needs its own review.

## Validation and limitations

Run the focused tests from the repository root in a suitable environment:

```bash
python -m pytest -q tests/workers/rollout/test_pd_contracts_on_cpu.py
python -m pytest -q tests/workers/rollout/test_pd_disaggregation.py tests/workers/rollout/test_vllm_pd_disaggregation_on_cpu.py tests/workers/rollout/test_pd_decode_router_adapter_on_cpu.py tests/workers/rollout/test_pd_metrics_labels_on_cpu.py tests/workers/rollout/test_prometheus_rollout_targets_on_cpu.py tests/workers/rollout/test_router_on_cpu.py tests/workers/rollout/test_topk_coverage_on_cpu.py
pre-commit run --all-files --show-diff-on-failure
```

The new `*_on_cpu.py` files match the existing CPU workflow collection pattern;
that workflow is owner-gated and is not automatically available on this fork.
AST-based tests execute selected production methods without loading vLLM. They
do not validate Ray actor startup or actual KV transport. vLLM-dependent tests
skip when vLLM is unavailable; optional Uni-Agent tests skip without that package.

Local results on Windows/Python 3.12: **25 contract tests passed**, changed-file
Ruff lint/format passed, and Python compilation passed. The four generated
trainer YAMLs were regenerated and verified with Python 3.12. Mypy and the
license/docs checks passed. Broader pytest collection failed while loading the
Windows PyTorch DLL; it was not counted as a pass. The full pre-commit invocation
was not green: its default Python 3.9/python3 resolution and a pre-existing
Windows path-separator issue in the test-layout check required explicit Python
3.12/manual checks. Hardware, complete application tests and CI remain pending.

Before merge/deployment, review these remaining integration requirements:

- Use a matched vLLM-Ascend/Mooncake build providing `MooncakeConnectorV1` and the
  worker RPC `disconnect_kv_transfer_peers`. Those dependency patches are not
  bundled here. Merely installing an arbitrary released wheel is not sufficient.
- Validate real NPU training, concurrent/partial rollout, abort/resume, weight
  synchronization, sleep/wake and the separate `release_kv_cache` path. Only the
  replica `sleep` path gains the drain/disconnect ordering in this patch.
- The colleague's optional legacy selector uses FIFO token-load release;
  out-of-order completion can miscount pending tokens. The adapter still follows
  its index-based release contract. Do not treat this as a request-exact tracker;
  the default built-in router avoids that optional token-accounting implementation.
- The corrected archive DOES contain a bottom-layer empty KV-group and
  notification-only completion fix in its patched vLLM-Ascend connector. That
  dependency is outside this repository and is NOT included in this PR. Verify
  the actually imported connector, not just the Python package version.
- Port reservations are released before child servers bind. This avoids internal
  range overlap but cannot eliminate races with unrelated processes on the host.

AI assistance was used for extraction, porting, regression tests and PR preparation.
Human line-by-line review and hardware validation remain required.
