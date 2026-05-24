# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from .deepseek_v3 import DeepSeekV3QModel


class DeepSeekV4QModel(DeepSeekV3QModel):
    # DeepSeek-V4 Flash uses a custom rotary embedding with per-layer-type
    # inv_freq buffers. These must be materialized to the quant device
    # before the calibration forward pass, otherwise `apply_rotary_pos_emb`
    # fails with a cuda:0 / cpu device mismatch.
    require_fast_init = False

    dynamic_expert_index = "n_routed_experts"
    rotary_embedding = "model.rotary_emb"

    def pre_quantize_generate_hook_start(self):
        inner = self.model.model
        self.shell_module_materialize(inner.embed_tokens, self.quantize_config.device)
        self.shell_module_materialize(inner.rotary_emb, self.quantize_config.device)
        hc_head = getattr(inner, "hc_head", None)
        if hc_head is not None:
            self.shell_module_materialize(hc_head, self.quantize_config.device)

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_a_norm:!",
                "q_a_proj:0",
                "q_b_norm:!",
                "q_b_proj:0",
                "o_a_proj:!",
                "o_b_proj:1",
                "kv_norm:!",
                "kv_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]



__all__ = ["DeepSeekV4QModel"]
