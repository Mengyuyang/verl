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

import yaml

from verl.workers.rollout.llm_server import _replica_label
from verl.workers.rollout.utils import _merge_rollout_static_configs


def test_replica_label_distinguishes_prefill_and_decode():
    assert _replica_label(0, {"pd_role": "prefill", "pd_index": 0}) == "0-prefill"
    assert _replica_label(0, {"pd_role": "decode", "pd_index": 2}) == "0-decode2"
    assert _replica_label(3, {"pd_role": "unified", "pd_index": 0}) == 3


def test_merge_rollout_static_configs_preserves_and_labels_targets(tmp_path):
    config_path = tmp_path / "prometheus.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "scrape_configs": [
                    {
                        "job_name": "rollout",
                        "static_configs": [{"targets": ["old:1", "new-p:2"]}],
                    }
                ]
            }
        )
    )

    merged = _merge_rollout_static_configs(
        str(config_path),
        ["new-p:2", "new-d0:3", "new-d1:4"],
        [
            {"pd_role": "prefill", "pd_index": 0},
            {"pd_role": "decode", "pd_index": 0},
            {"pd_role": "decode", "pd_index": 1},
        ],
    )

    assert {"targets": ["old:1"]} in merged
    assert {"targets": ["new-p:2"], "labels": {"pd_role": "prefill", "pd_index": 0}} in merged
    assert {"targets": ["new-d0:3"], "labels": {"pd_role": "decode", "pd_index": 0}} in merged
    assert {"targets": ["new-d1:4"], "labels": {"pd_role": "decode", "pd_index": 1}} in merged
