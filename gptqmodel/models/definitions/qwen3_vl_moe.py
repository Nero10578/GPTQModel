# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..moe_lifecycle import GateUpDownMoELifecycleHooks
from .base_qwen3_vl import BaseQwen3VLGPTQ


class Qwen3VLMoeQModel(BaseQwen3VLGPTQ):
    # Qwen3-VL-MoE (e.g. Qwen3-VL-235B-A22B): VL wrapper around a Qwen3-MoE text
    # model. Inherits VL loading/processor/hooks + `model.language_model.layers`
    # prefix from BaseQwen3VLGPTQ, and only swaps the dense MLP tree for the MoE
    # expert tree. This model has NO shared expert.

    # expand `mlp.experts.#` to `mlp.experts.0..num_experts-1` automatically
    dynamic_expert_index = "num_experts"

    pre_lm_head_norm_module = "model.language_model.norm"
    rotary_embedding = "model.language_model.rotary_emb"

    # fuse per-expert gate_proj/up_proj/down_proj -> checkpoint gate_up_proj/down_proj
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_norm:!", "k_norm:!", "q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe:?": {
                "gate": ("gate:!",),  # router: tiny + accuracy-sensitive, never quantize
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        }
    ]


__all__ = ["Qwen3VLMoeQModel"]
