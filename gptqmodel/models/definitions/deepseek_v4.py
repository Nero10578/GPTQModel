from .deepseek_v3 import DeepSeekV3QModel


class DeepSeekV4QModel(DeepSeekV3QModel):
    # DeepSeek-V4 Flash uses a custom rotary embedding with per-layer-type
    # inv_freq buffers and passes position_embeddings as a nested dict of
    # tuples to each decoder layer. The default nested_move_to does not
    # recurse into dicts, so cos/sin tensors inside position_embeddings
    # stay on CPU during layer replay. Override prepare_layer_replay_kwargs
    # to materialize them onto the target device.
    require_fast_init = False

    dynamic_expert_index = "n_routed_experts"
    rotary_embedding = "model.rotary_emb"

    def prepare_layer_replay_kwargs(
        self,
        layer,
        layer_input,
        additional_inputs,
        target_device,
    ):
        from ..utils.model import move_to

        pe = additional_inputs.pop("position_embeddings", None)
        if pe is not None:
            moved = {}
            for k, (cos, sin) in pe.items():
                moved[k] = (
                    move_to(cos, device=target_device),
                    move_to(sin, device=target_device),
                )
            additional_inputs["position_embeddings"] = moved
        return additional_inputs

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