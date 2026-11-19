# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import argparse
import asyncio
import inspect
import json
import logging
import os
import time
import uuid
import zlib
from collections.abc import Mapping
from pprint import pprint
from typing import Any, Callable, Optional

import ray
import vllm.entrypoints.cli.serve
from packaging import version
from ray.actor import ActorHandle
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.cli.serve import run_headless
from vllm.entrypoints.openai.api_server import build_app, init_app_state
from vllm.entrypoints.openai.parser.harmony_utils import get_encoding
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.outputs import STREAM_FINISHED, RequestOutput
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.metrics.loggers import PrometheusStatLogger

from verl.plugin.platform import get_platform
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_resource_name,
    get_visible_devices_keyword,
    is_torch_npu_available,
)
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.utils.profiler import (
    build_rollout_dist_profiler,
    build_vllm_profiler_args,
    relocate_rollout_traces,
    rollout_profiler_global_ranks,
)
from verl.utils.tokenizer import normalize_token_ids
from verl.utils.tracking import RLInsightLogger
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput, prefix_cache_salt
from verl.workers.rollout.utils import (
    extract_response_topk_logprobs_with_coverage,
    get_max_position_embeddings,
    get_node_visible_devices,
    get_vision_placeholder_token_ids,
    qwen2_5_vl_dedup_image_tokens,
    run_uvicorn,
    validate_topk_logprob_request,
    warn_topk_logprob_budget,
)
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    SuppressSignalInThread,
    build_cli_args_from_config,
    build_mtp_speculative_config,
    extract_prompt_logprobs,
    get_vllm_max_lora_rank,
)

_VLLM_VERSION = version.parse(vllm.__version__)

# vLLM args that verl sets from its own config, and the config that controls each one.
# rollout.engine_kwargs.vllm passes everything else through but may not set these.
_VERL_MANAGED_VLLM_ARGS = {
    "tensor_parallel_size": "rollout.tensor_model_parallel_size",
    "data_parallel_size": "rollout.data_parallel_size",
    "enable_expert_parallel": "rollout.expert_parallel_size (= tensor_model_parallel_size * data_parallel_size)",
    "enable_lora": "actor_rollout_ref.model.lora",
    "max_lora_rank": "actor_rollout_ref.model.lora",
    "enable_return_routed_experts": "rollout.enable_rollout_routing_replay",
}

# Max wait for admissions already past the submission gate to reach the engine.
_GATE_BARRIER_TIMEOUT_S = 60.0

if os.getenv("VERL_USE_GPT_OSS", "0") == "1":
    get_encoding()


logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


# 【讲解入口 4：一个 P/D 服务怎样完成请求】vLLMHttpServer 是服务层，模型计算在其底层引擎/worker 中。
# 外层 client 选一个 P → generate 识别完整请求 → _pd_dispatch 轮询选 D → P 算 KV → D 生成答案。
# 本版每个 P 只有自己的轮询游标，不加载可配置 D selector，也不登记/释放 D reservation。
# session_id 仍向引擎传递，但这里不根据 session 选择 D；不要套用旧版策略路由的说明。
# Ray 句柄用于调用；KV 参数描述如何交接；Connector 才负责实际 KV 数据搬运。
# 下面这个辅助函数只处理 Ascend MooncakeConnectorV1 的 D 侧参数，GPU 分支在类中单独处理。
def _build_mooncake_decode_kv_params(
    prefill_kv_params: Optional[Mapping[str, Any]],
    *,
    remote_engine_id: str,
    remote_bootstrap_addr: str,
    transfer_id: str,
) -> dict[str, Any]:
    """Build decode-side Mooncake params from the completed prefill leg."""
    # 把 P 返回的交接元数据复制为新字典；元数据含块编号/引擎/地址等，并非整份 KV 张量。
    # None 或空映射按空字典处理；dict 只复制外层，下面的顶层赋值不污染 P 返回字典。
    params = dict(prefill_kv_params or {})
    # setdefault 只补缺失的键，保留 Connector 已返回的实际引擎ID、联络地址和 transfer_id。
    # 已有键的值即使为 None 也不会被替换；这里不是对元数据做完整有效性检查。
    params.setdefault("remote_engine_id", remote_engine_id)
    params.setdefault("remote_bootstrap_addr", remote_bootstrap_addr)
    params.setdefault("transfer_id", transfer_id)

    # A prefill request that stops before producing transferable blocks (for
    # example, because its first sampled token is EOS) returns no
    # remote_block_ids. Advertising a remote prefill in that case makes the
    # Mooncake scheduler assert num_external_tokens == 0 and kills the engine.
    # Fall back to computing the prompt locally on decode instead.
    # 块字段缺失、None、[] 都按空序列处理，方便安全遍历；不能无条件声明 D 已有远端 KV 可取。
    # remote_block_ids 是 P 缓存块的编号；这个字段存在也不代表数据已经传到 D。
    blocks = params.get("remote_block_ids") or ()
    # A nonempty container of empty KV groups is not transferable. Block ID
    # zero is valid, so do not use the truth value of each scalar block ID.
    # 兼容平坦块列表 [0,5] 和按 KV 组组织的 [[0,5],[]]；any 表示至少有一块。
    # []、[[]]、[[],[]] 无块；[0]、[[0]]、[[],[5]] 有块。块号 0 合法，不能用 bool(0) 判空。
    # 组用是否非空判断；标量按块ID处理。这里只查有无块，不保证所有组完整或远端块仍存活。
    if any(bool(block) if isinstance(block, list | tuple) else True for block in blocks):
        # 站在接收这些参数的 D 角度：prefill 由另一个 P 完成，D 需要走接收远端 KV 的分支。
        # 下面 do_remote_decode=False 表示由当前 D 生成，不再转交另一个 D，不是停止生成。
        # P/D 即使同机，只要是不同引擎，这里也可以称为远端。
        params["do_remote_prefill"] = True
        params["do_remote_decode"] = False
    else:
        logger.warning(
            "Mooncake prefill returned no remote blocks; decoding the prompt locally (engine_id=%s, transfer_id=%s)",
            remote_engine_id,
            transfer_id,
        )
        # 没有可交接的块，让 D 按本地流程处理完整 prompt 后继续生成；是否命中本地前缀缓存由引擎决定。
        # 关闭远端 prefill 声明，避免声明要接收外部 KV 却没有对应块信息，导致底层参数不一致。
        params["do_remote_prefill"] = False
        params["do_remote_decode"] = False
    return params


class vLLMHttpServer:
    """vLLM http server in single node, this is equivalent to launch server with command line:
    ```
    vllm serve --tensor-parallel-size=8 ...
    ```
    """

    def __init__(
        self,
        config,
        model_config,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
        cuda_visible_devices: str,
        disaggregation_role: str = "null",
        disaggregation_kv_transfer_config: Optional[dict] = None,
    ):
        """
        Args:
            config (RolloutConfig): full config.
            model_config (HFModelConfig): model config.
            rollout_mode (RolloutMode): rollout mode.
            replica_rank (int): replica rank, a replica may contain multiple nodes.
            node_rank (int): node rank.
            gpus_per_node (int): number of gpus per node.
            nnodes (int): number of nodes.
            cuda_visible_devices (str): cuda visible devices.
            disaggregation_role: PD role, or ``"null"`` for normal rollout.
            disaggregation_kv_transfer_config: vLLM KVTransferConfig dict for PD.
        """
        if disaggregation_role not in ("null", "prefill", "decode"):
            raise ValueError(f"disaggregation_role must be 'null'|'prefill'|'decode', got {disaggregation_role!r}")
        if disaggregation_role != "null" and disaggregation_kv_transfer_config is None:
            raise ValueError(
                f"disaggregation_role={disaggregation_role!r} requires disaggregation_kv_transfer_config to be set"
            )
        # role 为 prefill/decode 时，此服务只代表组内一个 P/D；整个组由 vLLMPDReplica 创建并管理。
        # _pd_decode_peers 保存本 P 可用的 D 句柄；_pd_peer_idx 保存本 P 独立的轮询位置。
        # 不同 P 不共享这个游标，也没有跨 P 的 D 负载账本。
        self._disaggregation_role = disaggregation_role
        self._disaggregation_kv_transfer_config = disaggregation_kv_transfer_config
        # Filled by vLLMPDReplica.set_pd_peer for prefill-side routing.
        self._pd_decode_peers: list[ActorHandle] = []
        self._pd_prefill_side_channel_port: Optional[int] = None
        self._pd_prefill_engine_id: Optional[str] = None
        self._pd_peer_idx: int = 0

        os.environ[get_visible_devices_keyword()] = cuda_visible_devices
        os.environ["VERL_REPLICA_RANK"] = str(replica_rank)
        # Forward the Ray job id into the vLLM worker subprocess so the
        # colocated weight-transfer IPC socket path is unique per Ray job.
        # Without this, two concurrent verl jobs on the same node both bind
        # the same /tmp/rl-colocate-zmq-replica-0-rank-0.sock and one fails
        # with EADDRINUSE; a stale socket from a crashed run trips the same
        # error on restart.
        os.environ["VERL_RAY_JOB_ID"] = ray.get_runtime_context().get_job_id()

        self.config = self._init_config(config)
        self.model_config = self._init_model_config(model_config)
        self._validate_configs()

        if self.config.full_determinism:
            from verl.workers.engine.utils import enable_full_determinism

            rollout_seed = replica_rank + self.config.seed
            enable_full_determinism(seed=rollout_seed)
            os.environ["VERL_FULL_DETERMINISM"] = "1"
            os.environ["VERL_SEED"] = str(rollout_seed)
            os.environ["VLLM_BATCH_INVARIANT"] = "1"

        self.rollout_mode = rollout_mode
        self.workers = workers

        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.gpus_per_node = gpus_per_node
        self.nnodes = nnodes
        # model weights version, set by ServerAdapter when update weights.
        self.global_steps = None
        self._warned_missing_spec_decode_stats = False

        # vLLM's pause stops requests being scheduled but still accepts them, and a request
        # admitted after the pause is invisible to the drain's liveness check.
        self._submission_paused = False
        self._admitting = 0
        self._resume_event = asyncio.Event()
        self._resume_event.set()
        # Set by abort_all_requests(reject_request=True) when no resume is coming soon.
        self._rejecting = False

        # used for http server
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None

        # used for controlling vllm server profiler
        profiler_config = self.config.profiler
        tool_config = None
        if profiler_config is not None:
            if profiler_config.tool in ["torch", "npu"]:
                tool_config = omega_conf_to_dataclass((profiler_config.tool_config or {}).get(profiler_config.tool))
            else:
                logger.warning(f"agent loop only support torch and npu profiler, got {profiler_config.tool}")
                profiler_config = None
        # `ranks` in the rollout profiler config are global GPU ranks (as in the training roles);
        # map them to the replica that owns them so e.g. ranks=[0, 8] with tp=8 profiles the replicas
        # holding global ranks 0 and 8 (replicas 0 and 1), not replica indices 0 and 8.
        self.replica_world_size = (
            self.config.tensor_model_parallel_size
            * self.config.data_parallel_size
            * self.config.pipeline_model_parallel_size
        )
        self.profiler_controller = build_rollout_dist_profiler(
            self.replica_rank, self.replica_world_size, config=profiler_config, tool_config=tool_config
        )
        # A tp>1 engine profiles its whole replica, but the user asked for specific global GPU ranks;
        # keep only those when relocating so ranks=[0, 8] yields exactly GPU 0 and 8, not their tp-mates.
        self.profiler_keep_global_ranks = rollout_profiler_global_ranks(profiler_config)

        # used for data parallel: --data-parallel-address, --data-parallel-rpc-port
        if self.node_rank == 0:
            self._master_address = self._server_address
            # used for torch.distributed.init_process_group
            self._master_port, self._master_sock = get_free_port(self._server_address, with_alive_sock=True)
            # used for data parallel: --data-parallel-address, --data-parallel-rpc-port
            self._dp_rpc_port, self._dp_rpc_sock = get_free_port(self._server_address, with_alive_sock=True)
            self._dp_master_port, self._dp_master_sock = get_free_port(self._server_address, with_alive_sock=True)
        else:
            self._master_address = None
            self._master_port = None
            self._dp_rpc_port = None
            self._dp_master_port = None

        self._post_init(cuda_visible_devices)

    def get_master_address(self):
        """Get master address and port for data parallel.
        Returns:
            tuple: (master_address, master_port, dp_rpc_port)
        """
        return self._master_address, self._master_port, self._dp_rpc_port

    def get_server_address(self):
        """Get http server address and port."""
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    def get_rollout_config(self):
        """Get the RolloutConfig (e.g. max_num_seqs, max_model_len).

        Lets external routers fetch server-side config (vLLM doesn't expose
        these on /metrics) via the same handler-getter pattern as
        ``get_server_address``.
        """
        return self.config

    @property
    def lora_as_adapter(self) -> bool:
        return (
            self.model_config.lora_rank > 0 or self.model_config.lora.get("rank", 0) > 0
        ) and not self.model_config.lora.get("merge", False)

    async def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ):
        return await self.engine.collective_rpc(
            method=method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
        )

    async def set_pd_peer(
        self,
        decode_peers: list,
        prefill_side_channel_port: int,
        prefill_engine_id: str,
    ) -> None:
        # 【启动后接线】只允许对 P 设置 D 通讯录；启动器会对本组每个 P 各调用一次。
        # decode_peers 必须是非空列表；下面 list(...) 复制列表外层，保存的是 Ray 句柄。
        assert self._disaggregation_role == "prefill", (
            f"set_pd_peer must be called on the prefill server (got role={self._disaggregation_role!r})"
        )
        assert isinstance(decode_peers, list) and decode_peers, "decode_peers must be a non-empty list"
        self._pd_decode_peers = list(decode_peers)
        # 保存当前 P 自己的端口和 engine_id，后续告诉 D 去哪个 P 获取本次 KV。
        # 这里不查询 D 的 HTTP 地址、不创建内层 Router，也不执行 KV 传输。
        self._pd_prefill_side_channel_port = prefill_side_channel_port
        self._pd_prefill_engine_id = prefill_engine_id

    async def launch_server(self, master_address: str = None, master_port: int = None, dp_rpc_port: int = None):
        if self.node_rank != 0:
            assert master_address and master_port and dp_rpc_port, (
                "non-master node should provide master_address, master_port and dp_rpc_port"
            )
            self._master_address = master_address
            self._master_port = master_port
            self._dp_rpc_port = dp_rpc_port

        # 1. setup vllm serve cli args
        engine_kwargs = self.config.get("engine_kwargs", {}).get(self._get_engine_kwargs_key(), {}) or {}
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if self.config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": self.config.get("limit_images")}

        self._preprocess_engine_kwargs(engine_kwargs)

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        override_generation_config = self._get_override_generation_config()
        logger.info(f"override_generation_config: {override_generation_config}")

        logger.info(f"enable_sleep_mode: {self.config.enable_sleep_mode}")
        if not self.config.enable_sleep_mode:
            from verl.utils.device import set_expandable_segments

            set_expandable_segments(True)

        hf_overrides = self._apply_quantization()
        hf_quantization_config = getattr(self.model_config.hf_config, "quantization_config", None)
        effective_quantization_config = hf_overrides.get("quantization_config") or hf_quantization_config
        from verl.utils.vllm.vllm_quant_utils import set_default_vllm_quantization_backends

        set_default_vllm_quantization_backends(effective_quantization_config, engine_kwargs)

        compilation_config = engine_kwargs.pop("compilation_config", None) or {}
        if isinstance(compilation_config, str):
            compilation_config = json.loads(compilation_config)
        from verl.utils.real_nvfp4 import check_nvfp4_per_token_rollout

        check_nvfp4_per_token_rollout(self.config, effective_quantization_config, compilation_config, engine_kwargs)
        compilation_config.setdefault("cudagraph_mode", "FULL_AND_PIECEWISE")

        # FULL cuda graph is not yet supported with DCP, downgrade to PIECEWISE
        dcp_size = engine_kwargs.get("decode_context_parallel_size", 1) or 1
        if dcp_size > 1 and compilation_config["cudagraph_mode"] == "FULL_AND_PIECEWISE":
            logger.warning(
                "FULL cuda graph is not supported with DCP (decode_context_parallel_size=%d), "
                "downgrading cudagraph_mode to PIECEWISE.",
                dcp_size,
            )
            compilation_config["cudagraph_mode"] = "PIECEWISE"
        if self.config.cudagraph_capture_sizes:
            compilation_config["cudagraph_capture_sizes"] = self.config.cudagraph_capture_sizes

        compilation_config = json.dumps(compilation_config)
        args = {
            "dtype": self.config.dtype,
            "load_format": self.config.load_format,
            "skip_tokenizer_init": False,
            "distributed_executor_backend": "mp",
            "worker_extension_cls": self._get_worker_extension_cls(),
            "trust_remote_code": self.model_config.trust_remote_code,
            "max_model_len": self.config.max_model_len,
            "max_num_seqs": self.config.max_num_seqs,
            "enable_chunked_prefill": self.config.enable_chunked_prefill,
            "max_num_batched_tokens": self.config.max_num_batched_tokens,
            "enable_prefix_caching": self.config.enable_prefix_caching,
            "enable_sleep_mode": self.config.enable_sleep_mode,
            "logprobs_mode": self.config.logprobs_mode,
            "enforce_eager": self.config.enforce_eager,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "disable_log_stats": self.config.disable_log_stats,
            "tensor_parallel_size": self.config.tensor_model_parallel_size,
            "seed": self.replica_rank + self.config.seed,
            "override_generation_config": json.dumps(override_generation_config),
            "hf_overrides": hf_overrides,
            "scheduling_policy": self.config.scheduling_policy,
            "compilation_config": compilation_config,
            **engine_kwargs,
        }

        # update profiler args, only on the replica that will actually be profiled: configuring
        # the engine profiler everywhere makes every replica log that profiling is enabled while
        # only the selected one is ever started.
        if self._should_profile():
            # vLLM >= 0.13.0 takes the profiler config as a CLI arg and warns about the legacy
            # VLLM_TORCH_PROFILER_* environment variables it no longer reads.
            use_cli_args = _VLLM_VERSION >= version.parse("0.13.0")
            profiler_args = build_vllm_profiler_args(
                self.profiler_controller.config,
                self.profiler_controller.tool_config,
                self.replica_rank,
                legacy_env=not use_cli_args,
            )
            if use_cli_args:
                args.update(profiler_args)

        if self.config.prometheus.enable:
            if self.config.prometheus.served_model_name:
                # Extract model name from path if it's a full path
                served_model_name = self.config.prometheus.served_model_name
                if "/" in served_model_name:
                    # If it's a full path, extract the last part as model name
                    served_model_name = served_model_name.split("/")[-1]
                args["served_model_name"] = served_model_name

        if self.config.mtp is not None and self.config.mtp.enable and self.config.mtp.enable_rollout:
            args["speculative_config"] = build_mtp_speculative_config(
                self.config.mtp.method,
                self.config.mtp.num_speculative_tokens,
                args.get("speculative_config"),
            )

        if self.config.data_parallel_size > 1:
            assert self.gpus_per_node % self.config.tensor_model_parallel_size == 0, (
                "gpus_per_node should be divisible by tensor_model_parallel_size"
            )
            data_parallel_size_local = self.gpus_per_node // self.config.tensor_model_parallel_size
            assert len(self.workers) == data_parallel_size_local * self.config.tensor_model_parallel_size, (
                f"num workers ({len(self.workers)}) should be equal to "
                f"dp_size_local ({data_parallel_size_local}) * tp_size ({self.config.tensor_model_parallel_size})"
            )
            dp_args = {
                "data_parallel_size": self.config.data_parallel_size,
                "data_parallel_size_local": data_parallel_size_local,
                "data_parallel_address": self._master_address,
                "data_parallel_rpc_port": self._dp_rpc_port,
            }
            if self.node_rank > 0:
                # Only the headless follower nodes get a start rank. Since vLLM 0.28 a
                # non-headless server that receives ``data_parallel_start_rank`` is put
                # into hybrid-LB mode (EngineArgs.create_engine_config), and in that
                # mode the leader rejects headless remote engines with
                # "Remote engine N must not use --headless in external or hybrid dp lb
                # mode". The node-0 API server is the internal-LB leader and must not
                # carry a start rank (its local engines start at rank 0 anyway).
                dp_args["data_parallel_start_rank"] = self.node_rank * data_parallel_size_local
            else:
                # A start rank configured through engine_kwargs would put the leader into
                # hybrid-LB mode just the same; the leader's engines always start at rank 0.
                args.pop("data_parallel_start_rank", None)
            args.update(dp_args)

        args.update({"enable_expert_parallel": self.config.expert_parallel_size > 1})

        # used for torch.distributed.init_process_group
        if self.nnodes > 1:
            args.update(
                {
                    "master_addr": self._master_address,
                    "master_port": self._master_port,
                    "node_rank": self.node_rank,
                    "nnodes": self.nnodes,
                    "data_parallel_address": self._master_address,
                    "data_parallel_rpc_port": self._dp_rpc_port,
                }
            )

        # update lora-related args
        lora_rank = self.model_config.lora.get("rank", 0)
        if lora_rank <= 0:
            lora_rank = (
                self.model_config.lora_rank
            )  # FIXME: fallback to lora_rank for now, we should unify lora settings.

        if self.model_config.lora.get("merge", False):
            lora_rank = 0

        if lora_rank > 0:
            lora_args = {
                "enable_lora": True,
                "max_loras": 1,
                "max_lora_rank": get_vllm_max_lora_rank(lora_rank),
            }
            if self.model_config.lora.get("fully_sharded_loras", False):
                lora_args["fully_sharded_loras"] = True
            args.update(lora_args)

        if self.config.enable_rollout_routing_replay:
            # R3 (Rollout Router Replay) relies on vLLM's ``enable_return_routed_experts``
            # path (RoutedExpertsManager / RoutedExpertsCapturer), which is only correct
            # for hybrid-attention MoE models (e.g. Qwen3.5, whose linear + full attention
            # layout produces >1 KV-cache group) starting from vLLM 0.22.0. Earlier
            # releases either lack the feature or under-size the routed-experts host
            # buffer and crash with an IndexError. Fail fast with an actionable message
            # instead of surfacing an opaque runtime error deep inside vLLM.
            if _VLLM_VERSION < version.parse("0.22.0"):
                raise RuntimeError(
                    "rollout.enable_rollout_routing_replay=True requires vLLM >= 0.22.0 "
                    f"(installed: {vllm.__version__}). Upgrade vLLM (e.g. `pip install -U "
                    "'vllm>=0.22.0'`) or disable enable_rollout_routing_replay."
                )
            args.update({"enable_return_routed_experts": True})

        if self._disaggregation_role != "null":
            args["kv_transfer_config"] = json.dumps(self._disaggregation_kv_transfer_config)

        server_args = ["serve", self.model_config.local_path] + build_cli_args_from_config(args)

        if self.replica_rank == 0:
            pprint(server_args)

        CMD_MODULES = self._get_cli_modules()
        parser = FlexibleArgumentParser(description=self._get_cli_description())
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds = {}
        for cmd_module in CMD_MODULES:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        server_args = parser.parse_args(args=server_args)
        server_args.model = server_args.model_tag
        if server_args.subparser in cmds:
            cmds[server_args.subparser].validate(server_args)

        # 3. launch server
        if self.node_rank == 0:
            await self.run_server(server_args)
        else:
            await self.run_headless(server_args)

    async def run_server(self, args: argparse.Namespace):
        engine_args = AsyncEngineArgs.from_cli_args(args)
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)
        vllm_config.parallel_config.data_parallel_master_port = self._dp_master_port

        fn_args = set(dict(inspect.signature(AsyncLLM.from_vllm_config).parameters).keys())
        kwargs = {}
        if "enable_log_requests" in fn_args:
            kwargs["enable_log_requests"] = engine_args.enable_log_requests
        if "disable_log_stats" in fn_args:
            kwargs["disable_log_stats"] = engine_args.disable_log_stats

        engine_client = AsyncLLM.from_vllm_config(vllm_config=vllm_config, usage_context=usage_context, **kwargs)

        # Don't keep the dummy data in memory
        await engine_client.reset_mm_cache()
        # A sampled <|image_pad|>/<|video_pad|> has no image behind it, and every consumer of the
        # sequence assumes it does. Mask them out with the OOV tail, so the policy cannot pick one.
        await engine_client.collective_rpc(
            method="monkey_patch_model",
            kwargs={
                "vocab_size": len(self.model_config.tokenizer),
                "banned_token_ids": get_vision_placeholder_token_ids(self.model_config.processor),
            },
        )

        build_app_sig = inspect.signature(build_app)
        supported_tasks: tuple[Any, ...] = ()
        build_app_kwargs: dict[str, Any] = {}
        if "supported_tasks" in build_app_sig.parameters:
            supported_tasks = await engine_client.get_supported_tasks()
            build_app_kwargs["supported_tasks"] = supported_tasks
        # vLLM >= 0.20.0 requires `model_config` to register pooling API routes
        # (e.g. ``/classify``, ``/embed``); see
        # ``register_pooling_api_routers`` in vllm/entrypoints/pooling/factories.py
        # which short-circuits when ``model_config`` is ``None``.
        if "model_config" in build_app_sig.parameters:
            build_app_kwargs["model_config"] = engine_client.model_config
        app = build_app(args, **build_app_kwargs)

        init_app_sig = inspect.signature(init_app_state)
        if "vllm_config" in init_app_sig.parameters:
            await init_app_state(engine_client, vllm_config, app.state, args)
        elif "supported_tasks" in init_app_sig.parameters:
            await init_app_state(engine_client, app.state, args, supported_tasks)
        else:
            await init_app_state(engine_client, app.state, args)
        if self.replica_rank == 0 and self.node_rank == 0:
            logger.info(f"Initializing a V1 LLM engine with config: {vllm_config}")

        self.engine = engine_client
        self._server_port, self._server_task = await run_uvicorn(app, args, self._server_address)

    async def run_headless(self, args: argparse.Namespace):
        """Run headless server in a separate thread."""
        args.api_server_count = 0

        def run_headless_wrapper():
            with SuppressSignalInThread():
                run_headless(args)

        def on_run_headless_done(future: asyncio.Future):
            try:
                exc = future.exception()
                if exc:
                    logger.exception(f"run_headless failed with exception: {exc}")
                else:
                    logger.warning("run_headless completed successfully, but it's not expected.")
            except Exception as e:
                logger.exception(f"get result from run_headless failed: {e}")
            finally:
                os._exit(1)

        self.task = asyncio.create_task(asyncio.to_thread(run_headless_wrapper))
        self.task.add_done_callback(on_run_headless_done)

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        priority: int = 0,
        kv_transfer_params: Optional[dict] = None,
        session_id: Optional[str] = None,
    ) -> TokenOutput:
        """Generate sequence with token-in-token-out.

        Args:
            kv_transfer_params: vLLM KV-transfer payload for PD requests.
            session_id: Sticky caller id; turns of one session go to the same DP engine.
        """
        # 【完整请求还是内部阶段】当前是 P、有 D peers、且没有阶段 KV 参数，才进入 _pd_dispatch。
        # _pd_dispatch 内部会再调 self.generate，但带了 KV 参数，因此不会重复进入 dispatch，避免递归。
        # D 的角色不是 prefill，也会直接执行自己的引擎路径。
        if self._disaggregation_role == "prefill" and self._pd_decode_peers and kv_transfer_params is None:
            return await self._pd_dispatch(
                prompt_ids,
                sampling_params,
                request_id,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                mm_processor_kwargs=mm_processor_kwargs,
                priority=priority,
                session_id=session_id,
            )

        prompt_ids = normalize_token_ids(prompt_ids)

        # Calculate the maximum possible new tokens based on available context space
        # This serves as a safety upper bound. vLLM v0.20+ rejects `max_tokens < 1`
        # (see vllm.sampling_params.SamplingParams._verify_args), so we require at
        # least one token of headroom to be able to generate at all.
        max_possible_tokens = self.config.max_model_len - len(prompt_ids)
        if max_possible_tokens < 1:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) leaves no room to generate within the "
                f"model's maximum context length ({self.config.max_model_len}); need at least "
                f"1 token of headroom."
            )

        # Determine max_tokens from sampling_params or use configured response_length as default
        if "max_tokens" in sampling_params:
            max_tokens = sampling_params.pop("max_tokens")
        elif "max_new_tokens" in sampling_params:
            # support sglang-style 'max_new_tokens' param
            max_tokens = sampling_params.pop("max_new_tokens")
        else:
            # Default to a calculation that considers configured lengths
            # Cap max_tokens by response_length to ensure tensor alignment,
            # and by remaining budget to prevent OOM in multi-turn rollouts.
            max_tokens = min(
                self.config.response_length, self.config.prompt_length + self.config.response_length - len(prompt_ids)
            )

        # Clamp max_tokens to the valid range [1, max_possible_tokens]. The lower bound
        # is 1 because vLLM v0.20+ raises VLLMValidationError when max_tokens < 1.
        max_tokens = max(1, min(max_tokens, max_possible_tokens))

        assert 1 <= max_tokens <= max_possible_tokens, (
            f"max_tokens {max_tokens} not in valid range [1, {max_possible_tokens}]"
        )
        topk_log_probs = sampling_params.pop("topk_log_probs", 0)
        sampling_params["logprobs"] = (topk_log_probs or 0) if sampling_params.pop("logprobs", False) else None
        if sampling_params["logprobs"]:
            # the top-k head is read from flat lists, see extract_response_topk_logprobs
            sampling_params["flat_logprobs"] = True
        sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
        sampling_params.setdefault("ignore_eos", self.config.get("ignore_eos", False))
        # Inject per-request seed for deterministic sampling when full_determinism is enabled.
        if self.config.full_determinism:
            sampling_params.setdefault("seed", self.replica_rank + self.config.seed)

        if kv_transfer_params is not None:
            extra_args = dict(sampling_params.pop("extra_args", None) or {})
            extra_args["kv_transfer_params"] = kv_transfer_params
            sampling_params["extra_args"] = extra_args

        # verl rollout is token-in-token-out: the output path below only reads token_ids
        # and numeric logprobs (never .text), so detokenizing every generated token into a
        # string is pure overhead. Skip it. Guard: string-based `stop` / `bad_words` need
        # detokenized text to match, so only skip when neither is used (`stop_token_ids`
        # is unaffected). Teacher requests already set detokenize=False upstream, so this
        # setdefault leaves them untouched.
        if not sampling_params.get("stop") and not sampling_params.get("bad_words"):
            sampling_params.setdefault("detokenize", False)

        self._preprocess_sampling_params(sampling_params)
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
        validate_topk_logprob_request(sampling_params)
        budget_warnings = getattr(self, "_topk_budget_warnings", set())
        warn_topk_logprob_budget(sampling_params, budget_warnings)
        self._topk_budget_warnings = budget_warnings
        prompt_ids = qwen2_5_vl_dedup_image_tokens(prompt_ids, self.model_config.processor)
        multi_modal_data = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data
        if audio_data is not None:
            multi_modal_data["audio"] = audio_data

        prompt_kwargs = {"prompt_token_ids": prompt_ids, "multi_modal_data": multi_modal_data}
        if mm_processor_kwargs:
            prompt_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
        try:
            prompt = TokensPrompt(**prompt_kwargs)
        except TypeError:
            prompt = prompt_kwargs

        # Add lora request
        lora_request = None
        if self.lora_as_adapter:
            # Make sure we also check that the lora is already loaded in the engine
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
            if lora_loaded:
                lora_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH
                )

        rejected = await self._park_until_admitted(request_id)
        if rejected is not None:
            return rejected

        cache_salt = prefix_cache_salt(self.config, self.global_steps)
        if cache_salt is not None:
            prompt["cache_salt"] = cache_salt

        with RLInsightLogger.trace_state(
            "vllm_generate",
            state_lane_id=ray.get_runtime_context().get_actor_name(),
        ):
            final_res = await self._generate_admitted(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                lora_request=lora_request,
                priority=priority,
                # 这选的是当前服务内部的 DP 编号，不是选本组哪个 D；实际编号最后传给 engine.add_request。
                # 本版 PD 限制 DP=1，所以这里不额外指定 DP rank。
                data_parallel_rank=self._dp_rank_for_session(session_id),
            )
            assert final_res is not None

        extra_fields = {"global_steps": self.global_steps}
        # Handle abort case: when the request is aborted by pause_generation(abort),
        # outputs may be empty. Return empty results with stop_reason="aborted"
        # instead of crashing with "IndexError: list index out of range".
        if not final_res.outputs:
            aborted = TokenOutput(
                token_ids=[],
                log_probs=None,
                routed_experts=None,
                stop_reason="aborted",
                extra_fields=extra_fields,
            )
            return self._postprocess_output(aborted, final_res, sampling_params)

        # Prefix-cache hit count for this request; consumers surface it as
        # OpenAI usage.prompt_tokens_details.cached_tokens.
        extra_fields["num_cached_tokens"] = getattr(final_res, "num_cached_tokens", None)
        extract_prompt_logprobs(
            output=final_res,
            num_prompt_logprobs=sampling_params.prompt_logprobs,
            result_dict=extra_fields,
        )
        token_ids = final_res.outputs[0].token_ids
        log_probs = None
        if sampling_params.logprobs:
            # indexing FlatLogprobs per position would rebuild a k+1 dict per token
            (
                log_probs,
                extra_fields["response_topk_ids"],
                extra_fields["response_topk_log_probs"],
                extra_fields["response_support_complete"],
            ) = extract_response_topk_logprobs_with_coverage(
                final_res.outputs[0].logprobs,
                sampling_params.logprobs,
                vocab_size=self.engine.model_config.get_vocab_size(),
            )
            truncated = sampling_params.top_p < 1 or sampling_params.top_k > 0 or sampling_params.min_p > 0
            if truncated and not extra_fields["response_support_complete"].all():
                # Use the effective request after defaults, hooks and vLLM canonicalization.
                # Repeated requests with the same settings need only one diagnostic.
                signature = (
                    sampling_params.temperature,
                    sampling_params.top_p,
                    sampling_params.top_k,
                    sampling_params.min_p,
                    sampling_params.logprobs,
                )
                warned = getattr(self, "_topk_coverage_warnings", set())
                if signature not in warned:
                    logger.info(
                        "The recorded top-k head does not prove complete sampling support for every token; "
                        "score centering retains its head-plus-tail approximation for unknown rows. "
                        "Effective request: temperature=%s, top_p=%s, top_k=%s, min_p=%s, logprobs=%s.",
                        *signature,
                    )
                    warned.add(signature)
                    self._topk_coverage_warnings = warned
        elif sampling_params.logprobs is not None:
            log_probs = [logprobs[token_ids[i]].logprob for i, logprobs in enumerate(final_res.outputs[0].logprobs)]

        routed_experts = None
        if self.config.enable_rollout_routing_replay:
            routed_experts = final_res.outputs[0].routed_experts

        # Determine stop reason from finish_reason
        finish_reason = final_res.outputs[0].finish_reason
        if finish_reason == "abort":
            stop_reason = "aborted"
        elif finish_reason == "stop":
            stop_reason = "completed"
        else:
            stop_reason = finish_reason  # for more stop reason in the future

        num_preempted = None

        if hasattr(final_res.outputs[0], "num_preempted"):
            num_preempted = final_res.outputs[0].num_preempted

        response_kv_transfer_params = getattr(final_res, "kv_transfer_params", None)
        if response_kv_transfer_params is not None:
            extra_fields["kv_transfer_params"] = response_kv_transfer_params

        # Re-key backend spec-decoding stats to the rollout-common names.
        if self.config.mtp is not None and self.config.mtp.enable and self.config.mtp.enable_rollout:
            spec_decode_stats = getattr(final_res.metrics, "request_spec_decode_stats", None)
            if spec_decode_stats is None:
                if not self._warned_missing_spec_decode_stats:
                    logger.warning(
                        "vLLM MTP rollout metrics do not include request_spec_decode_stats; "
                        "speculative decoding acceptance metrics will be skipped."
                    )
                    self._warned_missing_spec_decode_stats = True
            else:
                extra_fields["spec_num_draft_tokens"] = spec_decode_stats.num_draft_tokens
                extra_fields["spec_num_accepted_tokens"] = spec_decode_stats.num_accepted_tokens
                extra_fields["spec_num_verify_steps"] = spec_decode_stats.num_verify_steps
        output = TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            routed_experts=routed_experts,
            stop_reason=stop_reason,
            num_preempted=num_preempted,
            extra_fields=extra_fields,
        )
        return self._postprocess_output(output, final_res, sampling_params)

    def _select_decode_peer(self) -> ActorHandle:
        """Round-robin across decode peers."""
        # 【组内 D 轮询】两个 D 时，游标 0/1/2/3 取余得到下标 0/1/0/1，依次选择 D0/D1/D0/D1。
        # 该选择只看本 P 的游标，不看 D 当前队列、会话亲和或缓存命中，也不登记 reservation。
        peer = self._pd_decode_peers[self._pd_peer_idx % len(self._pd_decode_peers)]
        # 选中后游标加一，供下次使用；即使后续 P 失败，这次已经推进，不会回滚游标。
        self._pd_peer_idx += 1
        return peer

    def _build_pd_decode_kv_params(
        self,
        prefill_out: TokenOutput,
        connector: str,
        transfer_id: str,
        request_id: str,
    ) -> dict[str, Any]:
        """Build decode KV parameters for NIXL, GPU Mooncake, or Ascend Mooncake."""
        # 【把 P 的结果整理成 D 的取货单】先读取 P 输出中的交接字段，按 Connector 协议分别处理。
        # 这里只构建描述信息，真正获取 KV 由 D 引擎及 Connector 完成。
        prefill_kv_params = prefill_out.extra_fields.get("kv_transfer_params")
        # GPU Mooncake 分支按已有协议补出 P 的身份、联络地址及传输ID；不依赖 P 返回同一套 V1 块结构。
        # 127.0.0.1 对应当前整组必须同机的限制，不能直接据此宣称支持跨机 PD。
        if connector == "MooncakeConnector":
            return {
                "do_remote_decode": False,
                "do_remote_prefill": True,
                "remote_engine_id": self._pd_prefill_engine_id,
                "remote_bootstrap_addr": (f"http://127.0.0.1:{self._pd_prefill_side_channel_port}"),
                "transfer_id": transfer_id,
            }
        # Ascend V1 调用文件上方的辅助函数，保留返回元数据，并根据实际有无远端块决定是否远端 prefill。
        if connector == "MooncakeConnectorV1":
            return _build_mooncake_decode_kv_params(
                prefill_kv_params,
                remote_engine_id=self._pd_prefill_engine_id,
                remote_bootstrap_addr=(f"http://127.0.0.1:{self._pd_prefill_side_channel_port}"),
                transfer_id=transfer_id,
            )
        # 其余路径（当前主要是 NIXL）需要 P 返回交接参数；缺失就报错，不能把不完整信息交给 D。
        # 正常时复制并传递返回字典，不把 GPU NIXL 协议强行改成 Mooncake V1 格式。
        if prefill_kv_params is None:
            raise RuntimeError(f"PD prefill leg returned no kv_transfer_params (request_id={request_id})")
        return dict(prefill_kv_params)

    def _dp_rank_for_session(self, session_id: Optional[str]) -> Optional[int]:
        """Pin a session to one DP engine for prefix-cache reuse; hash-based, so it ignores engine load."""
        # 【服务内部 DP 会话固定】DP 是数据并行副本，不是 Decode 的缩写，也不是这里的 D 选择策略。
        # 相同 session_id 且 DP 数量不变时，crc32 后取余会落到相同编号，提高复用前缀缓存的机会。
        dp_size = self.config.data_parallel_size
        # 没有会话ID，或只有一个 DP 引擎，就返回 None：不显式指定编号，不代表拒绝请求。
        if session_id is None or dp_size <= 1:
            return None
        # str 转字符串，encode 转字节，crc32 得到稳定整数，% 把结果限制到 0..dp_size-1。
        # 这种固定映射不考虑忙闲；缓存是否命中仍由实际前缀、缓存保留和引擎策略决定。
        return zlib.crc32(str(session_id).encode()) % dp_size

    # 【完整 PD 请求主线】本函数在 P 服务执行：轮询选 D → 本地 P 计算 → 整理交接参数 → 调 D。
    # prompt_ids 是完整输入的 token 编号；sampling_params 是原始生成设置；request_id 标识本轮请求。
    # 图片/视频/音频及 mm_processor_kwargs 是可选多模态输入；priority 是优先级；session_id 是会话ID。
    # async/await 让等待引擎和远程结果时可让出事件循环，不表示 P 与 D 的阶段没有先后依赖。
    # 本版已移除旧的 acquire_task/shield/reservation/finally-release；D 的选择仍发生在 P 计算之前。
    async def _pd_dispatch(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        priority: int = 0,
        session_id: Optional[str] = None,
    ) -> TokenOutput:
        """Select a round-robin decode peer, run prefill, then dispatch decode."""
        # 步骤1：取得这次要调用的 D 的 Ray Actor 句柄；它不是 HTTP 地址，也不是 KV 张量。
        # 这里只轮询选目标，不预留 D 显存、不限制 D 队列，也不按会话或负载挑选。
        decode_peer = self._select_decode_peer()
        # 读取当前 Connector 类名，后续据此构建不同协议的 D 参数；空配置按空字典处理。
        # get 的空字符串是缺键默认值，并不表示缺少 Connector 时这条路径也能正常工作。
        connector = (self._disaggregation_kv_transfer_config or {}).get("kv_connector", "")

        # Prefill only materializes KV; discard its single generated token.
        # 步骤2：给 P 复制一份独立的顶层参数；后面 generate 会 pop/修改字段，不能直接复用原字典。
        # dict 是浅拷贝，不是所有嵌套对象的深拷贝；这里修改的是副本的顶层长度与 EOS 字段。
        prefill_sp = dict(sampling_params)
        # 删除两种生成长度写法，避免原请求的大长度或 max_new_tokens 优先级影响这个临时 P 阶段。
        # pop(..., None) 表示键缺失时也不报错，随后统一设置 max_tokens=1。
        prefill_sp.pop("max_tokens", None)
        prefill_sp.pop("max_new_tokens", None)
        # P 借用生成接口完成输入计算，最多临时生成 1 个 token；这个 token 会丢弃，不作为答案前缀。
        # D 稍后使用原始 prompt 和 KV 自己生成答案，而不是接着这个临时 token 输出。
        prefill_sp["max_tokens"] = 1
        # 只对 P 副本忽略 EOS，避免临时 token 被当成自然结束信号而影响交接收尾；1-token 上限仍有效。
        # GPU 和 Ascend 的 PD 都经过此行；D 使用原始参数，普通非 PD 请求不经过此处。
        # 该设置不替代 Connector 兼容性与 EOS 边界实测，也不能阻止请求被取消或暂停。
        prefill_sp["ignore_eos"] = True
        # 步骤3：为本次 KV 交接生成唯一标识。request_id 是用户本轮请求，session_id 是多轮会话。
        # 本版没有 reservation_id；不要把 transfer_id 理解成路由负载预订编号。
        transfer_id = uuid.uuid4().hex
        # 以下标志从接收者 P 的角度读：本地做 prefill（remote_prefill=False），后续交给远端 D。
        # do_remote_decode=True 是声明 KV 交接角色；真正调用 D 的 Ray RPC 写在本函数末尾。
        prefill_kv_params = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "transfer_id": transfer_id,
        }

        # 步骤4：调用当前 P 的 generate；因为传入了 KV 阶段参数，这次直接走 P 引擎，不再进入 dispatch。
        # prompt_ids 是原输入，prefill_sp 是 P 专用副本；request_id 后缀 _P 区分本轮的 P 阶段。
        # 多模态输入、priority 和 session_id 原样转交；await 等 P 返回结果和 KV 交接描述。
        prefill_out = await self.generate(
            prompt_ids,
            prefill_sp,
            f"{request_id}_P",
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            mm_processor_kwargs=mm_processor_kwargs,
            priority=priority,
            kv_transfer_params=prefill_kv_params,
            session_id=session_id,
        )
        # An aborted prefill has no completed handoff; let the caller retry
        # after resume instead of sending incomplete KV metadata to D.
        # 步骤5：getattr 读取停止原因，没有该属性就取 None；abort/aborted 两种写法都视为中断。
        # P 被中断就原样返回，不构建不完整的交接描述，也不调用 D。恢复后的重试由调用方负责。
        # 这不是无远端块的正常回退：中断直接结束本轮，无块回退则允许 D 本地处理输入。
        if getattr(prefill_out, "stop_reason", None) in ("aborted", "abort"):
            return prefill_out
        # 步骤6：P 正常返回后，按 Connector 类型把输出整理成 D 的 KV 接收参数。
        # 传入 P 的结果、Connector 名、交接ID和请求ID；这里只整理元数据，不在 Python 层复制 KV tensor。
        decode_kv_params = self._build_pd_decode_kv_params(
            prefill_out,
            connector,
            transfer_id,
            request_id,
        )

        # 步骤7：通过 Ray 句柄调用选中的 D，等待真正的生成结果，并把该结果返回给上层 client。
        # dict(sampling_params) 仍是原始生成设置的副本，保留原长度和 EOS 设置；_D 区分 D 阶段请求。
        # 仍传完整 prompt、多模态数据和 session_id；KV 参数指引 Connector 取 P 的结果，无块时按协议回退。
        # remote 表示另一 Actor，可与 P 同机；本版没有 D 路由预订，所以这里也没有 release 清理分支。
        return await decode_peer.generate.remote(
            prompt_ids,
            dict(sampling_params),
            f"{request_id}_D",
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            mm_processor_kwargs=mm_processor_kwargs,
            priority=priority,
            kv_transfer_params=decode_kv_params,
            session_id=session_id,
        )

    @RLInsightLogger.trace_state_decorator(
        "wake_up", state_lane_id=lambda self: ray.get_runtime_context().get_actor_name()
    )
    # 【单引擎恢复】这是上游已有流程，按运行模式恢复当前一个 P/D 的引擎，并执行原有缓存策略。
    # 组级恢复屏障在 vllm_pd_replica.resume_generation：先恢复全组，再开放全组请求入口。
    # 本版已去掉 selector 专用的缓存失效通知，不改变这里原有的引擎生命周期。
    async def wake_up(self, tags: list[str] | None = None):
        if self.node_rank != 0:
            return

        if self.rollout_mode == RolloutMode.HYBRID:
            # engine.wake_up() broadcasts via the DP coordinator to ALL EngineCore
            # processes across all DP shards (unlike collective_rpc which only reaches
            # TP workers within a single shard).
            await self.engine.wake_up(tags=tags or self._get_wake_up_tags())
            await self._reset_prefix_cache()
        elif self.rollout_mode == RolloutMode.COLOCATED:
            # Directly call engine to wake up without sync weights.
            await self.engine.wake_up(tags=self._get_wake_up_tags())
            await self._reset_prefix_cache()
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")

    @RLInsightLogger.trace_state_decorator(
        "sleep", state_lane_id=lambda self: ray.get_runtime_context().get_actor_name()
    )
    # 【单引擎休眠：这里确实有 sleep】这是当前这个 P/D 服务的执行入口，不负责决定整组先后顺序。
    # vllm_pd_replica.sleep 负责等全部 P 排空，并在 Mooncake 路径先调用全部 D，再调用全部 P 的本方法。
    # 本方法属于上游已有代码，PR Changes 可能折叠或不显示；不代表这个文件没有休眠逻辑。
    async def sleep(self):
        # 非主节点或未启用 free_cache_engine 就返回；主节点根据 HYBRID/COLOCATED/STANDALONE 选择原有行为。
        # HYBRID 委托 _sleep_hybrid，COLOCATED 调 _engine_sleep(level=1)，STANDALONE 仅记录跳过。
        if self.node_rank != 0 or not self.config.free_cache_engine:
            return

        if self.rollout_mode == RolloutMode.HYBRID:
            await self._sleep_hybrid()
        elif self.rollout_mode == RolloutMode.COLOCATED:
            await self._engine_sleep(level=1)
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")

    # 【实际引擎休眠】保留上游 salt/reset 策略：salt 尽量避免清掉其他副本仍使用的外部缓存。
    # 引擎/worker 拥有传输内存注销生命周期；此处没有新增 Mooncake 私有 disconnect RPC。
    async def _engine_sleep(self, level: int):
        # Sleep discards the KV cache and, by default, resets the KV connector, which for a shared store
        # (e.g. MooncakeStoreConnector) deletes entries other replicas still use. Under "salt" that is never
        # needed: unchanged weights keep the current salt, and new weights get a new one.
        if self.config.external_kv_cache_invalidation == "salt":
            if "reset_connector" in inspect.signature(self.engine.sleep).parameters:
                await self.engine.sleep(level=level, reset_connector=False)
                return
            logger.warning(
                "external_kv_cache_invalidation='salt': this vLLM's sleep() cannot keep the KV connector's cache "
                "(needs vllm-project/vllm#57810), so the external KV store is reset on sleep."
            )
        await self.engine.sleep(level=level)

    async def _reset_prefix_cache(self):
        # Under "reset", reset_connector=True also drops any attached external KV store
        # (e.g. MooncakeStoreConnector) whose entries were computed against the previous
        # weights; with no connector it is a no-op success. Under "salt" those entries
        # are unreachable through their old salt, so the store, which other replicas may
        # still be reading and writing, is left intact.
        await self.engine.reset_prefix_cache(reset_connector=self.config.external_kv_cache_invalidation == "reset")

    @RLInsightLogger.trace_state_decorator(
        "clear_kv_cache", state_lane_id=lambda self: ray.get_runtime_context().get_actor_name()
    )
    # 【单引擎清缓存】保留上游的前缀缓存与多模态缓存重置；本版不维护额外 D selector 缓存状态。
    # 它是独立接口，不自动获得组级 sleep 的“先 D 后 P”顺序。
    async def clear_kv_cache(self):
        if self.node_rank == 0:
            await self._reset_prefix_cache()

            await self.engine.reset_mm_cache()
            await self.engine.reset_encoder_cache()

    @RLInsightLogger.trace_state_decorator(
        "release_kv_cache", state_lane_id=lambda self: ray.get_runtime_context().get_actor_name()
    )
    # 【独立 KV 释放入口】不同于组级 sleep；调用此入口不能自动获得整组 P/D 的排空及角色顺序。
    # 训练调用方仍需协调请求、权重更新与 KV 传输生命周期，不能只凭函数名认定全组已静默。
    async def release_kv_cache(self):
        """Free the kv_cache pool for the duration of a weight sync."""
        # TODO: use the real release_kv_cache() method after vllm supports it (vllm#44890/46438)
        if self.node_rank != 0 or not self.config.free_cache_engine:
            return
        if self.rollout_mode == RolloutMode.COLOCATED:
            return
        await self._engine_sleep(level=self._resolve_sleep_level())
        await self.engine.wake_up(tags=["weights"])

    @RLInsightLogger.trace_state_decorator(
        "resume_kv_cache", state_lane_id=lambda self: ray.get_runtime_context().get_actor_name()
    )
    async def resume_kv_cache(self):
        """Restore kv_cache GPU memory after a weight sync. Counterpart to release_kv_cache()."""
        if self.node_rank != 0 or not self.config.free_cache_engine:
            return
        if self.rollout_mode == RolloutMode.COLOCATED:
            return
        await self.engine.wake_up(tags=["kv_cache"])
        await self._reset_prefix_cache()

    async def snapshot(self) -> dict[str, Any]:
        """Return live KV-cache and scheduler queue observations.

        Values come from vLLM's PrometheusStatLogger gauges so callers can
        route or admit traffic without scraping the HTTP metrics endpoint.

        PD disaggregation is not supported: prefill and decode engines keep
        independent KV pools and schedulers, so one engine's gauges are not a
        replica snapshot.

        Only node rank 0 owns AsyncLLM. Other actors in a multi-node replica
        run vLLM headless and have no engine to observe.
        """
        # TODO: aggregate prefill + decode KV/scheduler stats for PD replicas.
        if self._disaggregation_role != "null":
            raise NotImplementedError("vLLMHttpServer.snapshot() does not support PD disaggregation")
        # Match abort_all_requests: headless actors must fail before touching self.engine.
        if self.node_rank != 0 or not hasattr(self, "engine"):
            raise RuntimeError(
                "vLLMHttpServer.snapshot() requires the node-rank-0 AsyncLLM; "
                f"node_rank={self.node_rank} actors run headless and have no engine."
            )
        prometheus_logger = self._prometheus_logger
        kv_cache_usage = self._prometheus_values(
            prometheus_logger.gauge_kv_cache_usage,
            "vllm:kv_cache_usage_perc",
        )
        if not kv_cache_usage:
            raise RuntimeError("vLLM KV-cache gauges are unavailable")
        return {
            "kv_cache_usage": max(kv_cache_usage),
            "num_waiting_requests": int(
                sum(
                    self._prometheus_values(
                        prometheus_logger.gauge_scheduler_waiting,
                        "vllm:num_requests_waiting",
                    )
                )
            ),
            "num_running_requests": int(
                sum(
                    self._prometheus_values(
                        prometheus_logger.gauge_scheduler_running,
                        "vllm:num_requests_running",
                    )
                )
            ),
        }

    @staticmethod
    def _prometheus_values(metrics_by_engine: dict[int, Any], sample_name: str) -> list[float]:
        return [
            float(sample.value)
            for metric in metrics_by_engine.values()
            for family in metric.collect()
            for sample in family.samples
            if sample.name == sample_name
        ]

    @property
    def _prometheus_logger(self) -> PrometheusStatLogger:
        # vLLM has moved PrometheusStatLogger between logger_manager attributes.
        logger_manager = self.engine.logger_manager
        if logger_manager is None:
            raise RuntimeError("Metrics monitoring requires disable_log_stats=False, but it is currently True.")

        prometheus_logger = getattr(logger_manager, "prometheus_logger", None)
        if isinstance(prometheus_logger, PrometheusStatLogger):
            return prometheus_logger

        for stat_logger in getattr(logger_manager, "stat_loggers", []):
            if isinstance(stat_logger, PrometheusStatLogger):
                return stat_logger

        raise RuntimeError("Unable to locate vLLM PrometheusStatLogger on logger_manager.")

    def _should_profile(self) -> bool:
        """Whether this replica drives the engine profiler."""
        return (
            self.profiler_controller.check_enable()
            and self.profiler_controller.check_this_rank()
            and self.profiler_controller.is_discrete_mode()
        )

    async def start_profile(self, **kwargs):
        if self.node_rank != 0:
            return
        if self._should_profile():
            await self.engine.start_profile(**kwargs)

    async def stop_profile(self):
        if self.node_rank != 0:
            return
        if self._should_profile():
            await self.engine.stop_profile()
            # Relocate the engine's traces into save_path (when relocate_results is set) so the
            # training worker's single end-of-run upload of the whole save_path picks them up. The
            # rollout engine does not run the finish command itself: it shares save_path with the
            # colocated training worker, so uploading here too would send the same directory twice.
            relocate_rollout_traces(
                self.profiler_controller.config,
                self.replica_rank,
                self.replica_world_size,
                self.profiler_keep_global_ranks,
            )

    async def set_global_steps(self, global_steps: int):
        """Set the global steps of the model weights."""
        self.global_steps = global_steps

    async def wait_for_requests_to_drain(self):
        await self.engine.wait_for_requests_to_drain()

    async def abort_all_requests(
        self,
        reset_prefix_cache: bool = True,
        reject_request: bool = False,
        abort_only: bool = False,
    ) -> dict[str, Any]:
        """Abort in-flight requests, optionally pausing the replica.

        The default closes admission, aborts in-flight work, and optionally
        clears caches. The engine remains paused after this call — use
        resume_generation() to accept new requests (e.g. before validation).

        abort_only=True only cancels current engine requests. Admission stays
        open and caches are left untouched. reset_prefix_cache and
        reject_request apply only when abort_only is False. Use abort_request()
        for a single id.

        Args:
            reset_prefix_cache: Clear the prefix/mm caches along with the pause.
            reject_request: Fail requests that arrive while the gate is closed
                instead of parking them. Pass True when this server is leaving
                the load balancer with no resume_generation coming soon (a hybrid
                replica handed back to training); parked requests would otherwise
                wait until the replica returns to rotation. The flag is re-declared
                by every pause and cleared by the matching resume, so a subsequent
                plain pause (e.g. the one inside a weight sync) restores parking.
            abort_only: Cancel in-flight requests and leave admission open.
                reset_prefix_cache and reject_request are ignored.

        Returns:
            dict[str, Any]: Dictionary containing:
                - aborted_count: Number of requests aborted
                - request_ids: List of aborted request IDs. Parallel-sampling
                  parent ids are aborted so their bookkeeping is released, and
                  are not included in the count.
        """
        # Only node rank 0 owns AsyncLLM/self.engine. The remaining actors in a
        # multi-node replica run vLLM's headless entry point, so there is no
        # engine object to abort through on those actors.
        if self.node_rank != 0:
            return {"aborted_count": 0, "request_ids": []}

        if abort_only:
            processor = self.engine.output_processor
            # request_states holds in-flight requests. For sampling n>1 those are
            # child ids; the ParentRequest lives in parent_requests under the parent
            # id and is not a request_states key. engine.abort pops that entry only
            # when the parent id is passed. Children come first: aborting the parent
            # id first recursively aborts children, and this call has already
            # detached them from external_req_ids, so the recursive remove throws.
            request_ids = list(processor.request_states)
            parent_ids = [
                parent_id for parent_id in processor.parent_requests if parent_id not in processor.request_states
            ]
            await self.engine.abort([*request_ids, *parent_ids], internal=True)
            return {"aborted_count": len(request_ids), "request_ids": request_ids}

        try:
            # Close the gate first, then let admissions already past it land, so the pause
            # below actually covers them. Closing the gate and declaring how late arrivals
            # are handled happen together, so no request can observe one without the other.
            self._submission_paused = True
            self._rejecting = reject_request
            self._resume_event.clear()
            deadline = time.monotonic() + _GATE_BARRIER_TIMEOUT_S
            while self._admitting > 0:
                if time.monotonic() > deadline:
                    if self.config.external_kv_cache_invalidation == "salt":
                        # A straggler would run on the new weights under its old salt and write
                        # into the kept KV store, so do not proceed with the weight update.
                        raise RuntimeError(
                            f"Submission gate barrier timed out with {self._admitting} admission(s) in flight"
                        )
                    logger.warning(
                        "Submission gate barrier timed out with %d admission(s) in flight, proceeding",
                        self._admitting,
                    )
                    break
                await asyncio.sleep(0.01)

            # Snapshot request IDs before pausing for reporting
            request_ids = list(self.engine.output_processor.request_states.keys())

            # pause_generation in abort mode will:
            # 1. Set engine to paused state (new requests are accepted but not scheduled)
            # 2. Abort all in-flight requests
            # 3. Wait for requests to drain
            # 4. Clear prefix and mm caches if clear_cache=True.
            #    EngineCore._reset_caches defaults reset_connector=True
            #    on this path, so any attached external KV store (e.g.
            #    MooncakeStoreConnector) is invalidated along with the
            #    local prefix cache — RL-correct hard-reset at every
            #    weight update boundary. pause_generation cannot keep the
            #    connector, so "salt" pauses without clearing and then
            #    clears only the local caches via clear_kv_cache().
            reset_connector = self.config.external_kv_cache_invalidation == "reset"
            clear_cache = reset_prefix_cache and reset_connector
            if _VLLM_VERSION >= version.parse("0.32.0"):
                await self.engine.pause_generation(mode="abort", clear_cache=clear_cache)
            else:
                await self.engine.pause_generation(
                    wait_for_inflight_requests=False,
                    clear_cache=clear_cache,
                )
            if reset_prefix_cache and not reset_connector:
                await self.clear_kv_cache()

            logger.info(f"Aborted {len(request_ids)} requests: {request_ids}")
            return {"aborted_count": len(request_ids), "request_ids": request_ids}

        except Exception:
            # Weight updates must not proceed unless every in-flight request was
            # actually aborted and the old-weight caches were cleared.
            logger.exception("Error aborting requests")
            raise

    async def resume_engine_generation(self):
        """Resume the DP engines while request admission remains closed."""
        if self.node_rank == 0:
            await self.engine.resume_generation()

    async def open_submission_gate(self):
        """Admit requests after the replica has finished resuming its engines."""
        self._rejecting = False
        self._submission_paused = False
        self._resume_event.set()

    async def resume_generation(self):
        """Resume generation after abort_all_requests (pause_generation)."""
        await self.resume_engine_generation()
        await self.open_submission_gate()

    async def _park_until_admitted(self, request_id: str) -> Optional[TokenOutput]:
        """Wait out a closed gate, or fail the request when the gate rejects late arrivals.

        Returns None once the request is admitted, or an aborted TokenOutput that tells the
        client to re-route. Rejection is checked before waiting, so a rejecting gate never
        parks anyone and the request never reaches the engine.
        """
        while self._submission_paused:
            if self._rejecting:
                logger.debug("rejecting request %s: server left rotation behind a closed gate", request_id)
                return TokenOutput(
                    token_ids=[],
                    log_probs=None,
                    routed_experts=None,
                    stop_reason="aborted",
                    extra_fields={"global_steps": self.global_steps},
                )
            logger.debug("parking request %s until weight sync completes", request_id)
            await self._resume_event.wait()
        # No await between the final gate check and the bump: on the actor's single event loop
        # that keeps "gate closed and _admitting == 0" from being observed mid-admission.
        self._admitting += 1
        return None

    async def _generate_admitted(
        self,
        prompt: Any,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest],
        priority: int,
        data_parallel_rank: Optional[int] = None,
    ) -> Optional[RequestOutput]:
        """Run one admitted request to completion and return its final RequestOutput.

        The caller has already bumped ``_admitting`` in ``_park_until_admitted``; this method drops
        it exactly once, as soon as the request is guaranteed to be covered by a later
        ``pause_generation()``. That is when ``AsyncLLM.add_request()`` returns: the request is
        then registered in the output processor and has been sent to EngineCore over the same
        ordered input socket that ``pause_scheduler`` travels on.

        Releasing the barrier only on the first yielded token (the previous behaviour, via
        ``AsyncLLM.generate()``) made it wait for the request to be *scheduled and prefilled*.
        With ``len(batch) > max_num_seqs`` most requests sit in the scheduler's WAITING queue for
        seconds, and ``abort_all_requests()`` could not pause the engine until every one of them
        had been scheduled -- a 5-15s stall per weight sync on TPU GRPO runs.
        """
        final_res: Optional[RequestOutput] = None
        queue = None
        admitted = False
        try:
            queue = await self.engine.add_request(
                request_id,
                prompt,
                sampling_params,
                lora_request=lora_request,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
            )
            # Landed in the engine: a pause from here on covers this request.
            admitted = True
            self._admitting -= 1

            # Drain the per-request collector exactly as AsyncLLM.generate() does.
            finished = False
            while not finished:
                output = queue.get_nowait() or await queue.get()
                finished = output.finished
                if output is STREAM_FINISHED:
                    continue
                final_res = output
        except (asyncio.CancelledError, Exception) as e:
            # Client disconnected or the request failed mid-flight: drop it from the engine like
            # AsyncLLM.generate() does, so no orphaned request is left in the scheduler. The collector
            # holds vLLM's internal request id, which may differ from the caller's request_id.
            if queue is not None and not isinstance(e, EngineDeadError):
                try:
                    await self.engine.abort(queue.request_id, internal=True)
                except Exception:
                    logger.debug("failed to abort request %s after %s", request_id, type(e).__name__, exc_info=True)
            raise
        finally:
            if not admitted:
                self._admitting -= 1
            if queue is not None:
                queue.close()
        return final_res

    async def abort_request(self, request_id: str, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort a specific generation request.

        Args:
            request_id: The ID of the request to abort.

        Returns:
            dict[str, Any]: Dictionary containing abort result.
        """
        if self.node_rank != 0:
            return {"aborted": False, "request_id": request_id}

        try:
            request_states = self.engine.output_processor.request_states
            req_state = request_states.get(request_id)

            if req_state is None:
                return {"aborted": False, "error": f"Request {request_id} not found"}

            # Create abort output and put it to the queue
            from vllm.v1.engine import FinishReason

            request_output = req_state.make_request_output(
                [], pooling_output=None, finish_reason=FinishReason.ABORT, stop_reason=None
            )
            req_state.queue.put(request_output)

            # Abort in output processor and engine core
            self.engine.output_processor.abort_requests([request_id])
            await self.engine.engine_core.abort_requests_async([request_id])

            # Try to reset prefix cache to ensure clean state
            if reset_prefix_cache:
                await self.clear_kv_cache()
                logger.info(f"Prefix cache reset after abort request {request_id}")

            logger.info(f"Aborted request: {request_id}")
            return {"aborted": True, "request_id": request_id}

        except Exception as e:
            logger.error(f"Error aborting request {request_id}: {e}")
            return {"aborted": False, "request_id": request_id, "error": str(e)}

    # -----------------------------------------------------------------------
    # Hook methods for subclass overrides
    # -----------------------------------------------------------------------

    def _init_config(self, config):
        """Initialise config. Override when a specific dataclass_type is needed."""
        return omega_conf_to_dataclass(config)

    def _init_model_config(self, model_config):
        """Initialise model_config. Override when a specific dataclass_type is needed."""
        return omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)

    def _validate_configs(self) -> None:
        """Validate config/model_config after initialisation."""
        max_position_embeddings = get_max_position_embeddings(self.model_config.hf_config)
        if self.config.max_model_len is None:
            self.config.max_model_len = max_position_embeddings
        else:
            if self.config.max_model_len > max_position_embeddings:
                raise ValueError(
                    f"max_model_len ({self.config.max_model_len}) should be less than or equal to "
                    f"max_position_embeddings ({max_position_embeddings})"
                )

        if not self.config.enable_chunked_prefill and self.config.max_num_batched_tokens < self.config.max_model_len:
            logger.warning(
                "enable_chunked_prefill=False requires max_num_batched_tokens >= max_model_len "
                f"({self.config.max_model_len}); raising max_num_batched_tokens from "
                f"{self.config.max_num_batched_tokens} to {self.config.max_model_len}."
            )
            self.config.max_num_batched_tokens = self.config.max_model_len

    def _post_init(self, cuda_visible_devices: str) -> None:
        """Called at the end of __init__. Default logs server metadata."""
        logger.info(
            f"{self.__class__.__name__}, replica_rank: {self.replica_rank}, node_rank: {self.node_rank}, "
            f"{get_visible_devices_keyword()}: {cuda_visible_devices}, "
            f"master_address: {self._master_address}, master_port: {self._master_port}, "
            f"data_parallel_rpc_port: {self._dp_rpc_port}, data_parallel_master_port: {self._dp_master_port}"
        )

    def _get_engine_kwargs_key(self) -> str:
        """Return the key under config.engine_kwargs for this engine (e.g. 'vllm')."""
        return "vllm"

    def _preprocess_engine_kwargs(self, engine_kwargs: dict) -> None:
        """Mutate engine_kwargs in-place before the CLI args dict is built."""
        managed = [key for key in engine_kwargs if key.replace("-", "_") in _VERL_MANAGED_VLLM_ARGS]
        if managed:
            raise ValueError(
                "rollout.engine_kwargs.vllm may not set args that verl derives from its config: "
                + "; ".join(f"{key} (set {_VERL_MANAGED_VLLM_ARGS[key.replace('-', '_')]})" for key in managed)
            )
        if _VLLM_VERSION >= version.parse("0.27.1") and is_torch_npu_available(check_device=False):
            # Ascend RL defaults disable NZ weights for sleep/wake and weight updates.
            additional_config = dict(engine_kwargs.get("additional_config") or {})
            rl_config = dict(additional_config.get("rl_config") or {})
            rl_config.setdefault("enabled", True)
            additional_config["rl_config"] = rl_config
            engine_kwargs["additional_config"] = additional_config

        if _VLLM_VERSION < version.parse("0.22.0"):
            # Work around multimodal processor cache desync across pause/resume.
            # See: https://github.com/vllm-project/vllm/pull/43001/
            engine_kwargs.setdefault("mm_processor_cache_gb", 0)

        checkpoint_config = getattr(self.config, "checkpoint_engine", None)
        if getattr(checkpoint_config, "backend", None) == "delta_sharded":
            from verl.workers.rollout.vllm_rollout.delta_weight_transfer import (
                VERL_DELTA_WEIGHT_TRANSFER_BACKEND,
                is_moe_model,
                require_vllm_delta_support,
            )

            require_vllm_delta_support()
            delta_engine_kwargs = getattr(checkpoint_config, "engine_kwargs", {}).get("delta_sharded", {})
            if int(delta_engine_kwargs.get("verify_every", 0)) > 0:
                raise NotImplementedError("delta_sharded with vLLM does not support verify_every > 0")
            if self.config.data_parallel_size != 1:
                raise NotImplementedError("delta_sharded with vLLM requires data_parallel_size=1")
            if self.config.disaggregation.enabled:
                raise NotImplementedError("delta_sharded with vLLM does not support PD disaggregation")
            # config.pipeline_model_parallel_size > 1 is already rejected globally and
            # engine_kwargs cannot set TP/DP (see _VERL_MANAGED_VLLM_ARGS); close the
            # remaining pipeline override that would bypass VERL's worker and IPC mapping.
            if int(engine_kwargs.get("pipeline_parallel_size") or 1) > 1:
                raise NotImplementedError("delta_sharded with vLLM requires pipeline_parallel_size=1")

            if is_moe_model(self.model_config.hf_config):
                moe_backend = engine_kwargs.get("moe_backend")
                if moe_backend not in {None, "auto", "triton"}:
                    raise NotImplementedError(
                        f"delta_sharded with vLLM MoE requires moe_backend='triton'; got {moe_backend!r}"
                    )
                engine_kwargs["moe_backend"] = "triton"
                if engine_kwargs.get("enable_eplb", False):
                    raise NotImplementedError("delta_sharded with vLLM MoE does not support EPLB")

            weight_transfer_config = engine_kwargs.get("weight_transfer_config")
            if weight_transfer_config is None:
                weight_transfer_backend = None
            elif isinstance(weight_transfer_config, Mapping):
                weight_transfer_backend = weight_transfer_config.get("backend")
            else:
                raise TypeError("weight_transfer_config must be a mapping when using delta_sharded")

            if weight_transfer_backend not in {None, VERL_DELTA_WEIGHT_TRANSFER_BACKEND}:
                raise ValueError(
                    "checkpoint_engine.backend='delta_sharded' requires vLLM "
                    f"weight transfer backend {VERL_DELTA_WEIGHT_TRANSFER_BACKEND!r}, "
                    f"but got {weight_transfer_backend!r}"
                )
            engine_kwargs["weight_transfer_config"] = {"backend": VERL_DELTA_WEIGHT_TRANSFER_BACKEND}

    def _get_override_generation_config(self) -> dict:
        """Return the override_generation_config dict."""
        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        return dict(
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repetition_penalty=1.0,
            max_new_tokens=self.config.response_length,
        )

    def _apply_quantization(self) -> dict:
        """hf_overrides for a rollout quantization set in the rollout config.

        Models shipping their own ``quantization_config`` need nothing here; either way the
        workers install the refit patches from the resulting vLLM ``quant_config``.
        """
        quantization = self.config.quantization
        if not quantization.enabled:
            return {}

        # The trainer quantizes every refit to this config; vLLM picks the method up from hf_overrides.
        quantization_config = quantization.build_quantization_config(self.model_config.local_path)
        logger.info(f"Rollout compressed-tensors quantization_config: {quantization_config}")
        return {"quantization_config": quantization_config}

    def _get_worker_extension_cls(self) -> str:
        """Return the fully-qualified colocate worker extension class name."""
        return "verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension"

    def _get_cli_modules(self) -> list:
        """Return the list of CLI command modules used for argument parsing."""
        return [vllm.entrypoints.cli.serve]

    def _get_cli_description(self) -> str:
        """Return the description string for the CLI argument parser."""
        return "vLLM CLI"

    def _get_wake_up_tags(self) -> list[str]:
        """Return the tags passed to engine.wake_up(). Default includes kv_cache."""
        return ["kv_cache", "weights"]

    def _resolve_sleep_level(self) -> int:
        """Deepest sleep level whose discarded state a subsequent weight sync can restore.

        MTP drafter-only weights are initialized by vLLM and are not guaranteed
        to be restored by actor weight sync after level 2 sleep discards them.
        lora only update adapter weights, so set sleep level to 1.
        vllm-ascend < 0.30.0 requires sleep level 1 to avoid EP precision issues.
        """
        mtp_config = getattr(self.config, "mtp", None)
        mtp_rollout_enabled = (
            mtp_config is not None
            and getattr(mtp_config, "enable", False)
            and getattr(mtp_config, "enable_rollout", False)
        )
        if mtp_rollout_enabled or self.lora_as_adapter:
            return 1
        if _VLLM_VERSION < version.parse("0.30.0") and is_torch_npu_available(check_device=False):
            return 1
        return 2

    def _preprocess_sampling_params(self, sampling_params: dict[str, Any]) -> None:
        """Mutate a request's sampling_params in-place after defaults are filled.

        Runs before SamplingParams is built and before the request is admitted, so an
        exception fails the request without touching the admission gate.
        """

    def _postprocess_output(
        self, output: TokenOutput, final_res: RequestOutput, sampling_params: SamplingParams
    ) -> TokenOutput:
        """Return the output of a request that reached the engine, including aborted ones.

        Runs after the admission counter is released, so an exception fails only this
        request. Requests rejected at the admission gate never reach this hook.
        """
        return output

    async def _sleep_hybrid(self):
        """HYBRID sleep: adapters and MTP need level=1; full weights need level=2.

        Uses engine.sleep() instead of engine.collective_rpc("sleep") to ensure
        that sleep is properly propagated to all data-parallel worker processes.
        collective_rpc only reaches the TP workers within a single DP shard,
        leaving other DP shards' weights unreleased, which causes OOM during
        FSDP training backward when DP > 1.
        """
        await self._engine_sleep(level=self._resolve_sleep_level())
        await self.engine.reset_encoder_cache()


class vLLMReplica(RolloutReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
        name_suffix: str = "",
    ):
        super().__init__(
            replica_rank, config, model_config, gpus_per_node, is_reward_model, is_teacher_model, name_suffix
        )
        self.server_class = ray.remote(vLLMHttpServer)

    def _get_server_env_vars(self, node_ids: list[str]) -> dict[str, str]:
        """Environment variables of this replica's server actors (one per node, ``node_ids`` in node-rank order).

        Platforms whose engine needs per-replica settings (e.g. the device topology of the replica's hosts) can
        override this instead of ``launch_servers``.
        """
        return {
            **{var: "1" for var in get_platform().ray_noset_envvars()},
            **get_platform().rollout_env_vars(),
        }

    async def launch_servers(self):
        """Launch http server in each node."""
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        # get (node_id, accelerator ids) of all workers; a worker that holds no accelerator has no ids
        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (
                        ray.get_runtime_context().get_node_id(),
                        ray.get_runtime_context().get_accelerator_ids().get(get_resource_name(), []),
                    )
                )
                for worker in self.workers
            ]
        )
        worker_accelerator_ids = [worker_info[1] for worker_info in worker_infos]
        worker_node_ids = [worker_info[0] for worker_info in worker_infos]

        # create server actor in each node with node affinity and cuda visible devices
        nnodes, gpus_per_replica_node = self.nnodes, self.gpus_per_replica_node
        env_vars = self._get_server_env_vars(worker_node_ids[::gpus_per_replica_node])
        for node_rank in range(nnodes):
            workers = self.workers[node_rank * gpus_per_replica_node : (node_rank + 1) * gpus_per_replica_node]
            node_cuda_visible_devices = get_node_visible_devices(
                worker_accelerator_ids[node_rank * gpus_per_replica_node : (node_rank + 1) * gpus_per_replica_node]
            )
            node_id = worker_node_ids[node_rank * gpus_per_replica_node]
            prefix = self._get_server_name_prefix()
            if self.is_reward_model:
                name = f"{prefix}server_reward_{self.replica_rank}_{node_rank}{self.name_suffix}"
            elif self.is_teacher_model:
                name = f"{prefix}server_teacher_{self.replica_rank}_{node_rank}{self.name_suffix}"
            else:
                name = f"{prefix}server_{self.replica_rank}_{node_rank}{self.name_suffix}"
            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={"env_vars": env_vars},
                name=name,
                max_concurrency=self.config.ray_actor_max_concurrency,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                gpus_per_node=gpus_per_replica_node,
                nnodes=nnodes,
                cuda_visible_devices=node_cuda_visible_devices,
            )
            self.servers.append(server)

        # launch http server in each node
        master_address, master_port, dp_rpc_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(
                    master_address=master_address, master_port=master_port, dp_rpc_port=dp_rpc_port
                )
                for server in self.servers
            ]
        )

        # get http server address from first server
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )

    async def sleep(self):
        """Sleep each rollout server."""
        # Drain DP engines for safe sleep.
        await self.servers[0].wait_for_requests_to_drain.remote()
        await asyncio.gather(*[server.sleep.remote() for server in self.servers])

    async def abort_all_requests(
        self,
        reject_request: bool = False,
        reset_prefix_cache: bool = True,
        abort_only: bool = False,
    ) -> dict[str, Any]:
        """Abort in-flight requests on every server.

        abort_only=False (default) also closes admission. See
        vLLMHttpServer.abort_all_requests(). reset_prefix_cache and
        reject_request apply only when abort_only is False.
        """
        results = await asyncio.gather(
            *[
                server.abort_all_requests.remote(
                    reject_request=reject_request,
                    reset_prefix_cache=reset_prefix_cache,
                    abort_only=abort_only,
                )
                for server in self.servers
            ]
        )

        total_aborted = sum(r.get("aborted_count", 0) for r in results)
        all_request_ids = []
        for r in results:
            all_request_ids.extend(r.get("request_ids", []))

        return {
            "aborted_count": total_aborted,
            "request_ids": all_request_ids,
            "server_results": results,
        }

    async def resume_generation(self):
        """Resume the DP engines before admitting requests on any server."""
        # KV-cache wake-up may already have resumed scheduling. Keep admission
        # closed during this resume too: new requests could otherwise start a
        # DP wave while only some ranks are still in the resume collective.
        await self.servers[0].resume_engine_generation.remote()
        await asyncio.gather(*[server.open_submission_gate.remote() for server in self.servers])

    async def abort_request(self, request_id: str) -> dict[str, Any]:
        """Abort a specific request. Tries all servers since we don't know which one has it.

        Args:
            request_id: The ID of the request to abort.

        Returns:
            dict[str, Any]: Abort result.
        """
        # TODO(petersh6): we should only abort on the server that has the request.
        results = await asyncio.gather(*[server.abort_request.remote(request_id) for server in self.servers])

        for r in results:
            if r.get("aborted", False):
                return r

        return {"aborted": False, "request_id": request_id, "error": "Request not found on any server"}

    async def snapshot(self) -> dict[str, Any]:
        """Return live KV-cache and scheduler observations from the head server.

        Colocated / non-PD replicas report the head HTTP server. PD is out of
        scope: prefill and decode engines have independent KV pools.
        """
        # TODO: aggregate prefill + decode snapshots for PD replicas.
        if self.config.disaggregation.enabled:
            raise NotImplementedError("vLLMReplica.snapshot() does not support PD disaggregation")
        return await self.servers[0].snapshot.remote()

    async def release_kv_cache(self):
        # Drain all in-flight requests so that vLLM worker threads go idle
        # before we touch engine.release_kv_cache()
        await self.servers[0].wait_for_requests_to_drain.remote()
        await asyncio.gather(*[server.release_kv_cache.remote() for server in self.servers])

    # -----------------------------------------------------------------------
    # Hook methods for subclass overrides
    # -----------------------------------------------------------------------

    def _get_server_name_prefix(self) -> str:
        """Return the Ray actor name prefix (e.g. 'vllm_')."""
        return "vllm_"
