"""
utils/ckpt_utils.py — Checkpoint conversion helpers.

vLLM fuses several weight matrices for inference efficiency.  When a
checkpoint is saved from a live vLLM engine (via save_self_weights_to_disk)
the keys follow vLLM's fused naming convention, not HuggingFace's.  This
module provides unfuse_vllm_state_dict() to convert the fused keys back to
the split keys expected by AutoModelForCausalLM.load_state_dict().

Fused → split mapping
---------------------
  self_attn.qkv_proj.weight  →  q_proj.weight + k_proj.weight + v_proj.weight
  self_attn.qkv_proj.bias    →  q_proj.bias   + k_proj.bias   + v_proj.bias
  mlp.gate_up_proj.weight    →  mlp.gate_proj.weight + mlp.up_proj.weight
"""

import torch


def unfuse_vllm_state_dict(state_dict: dict, config) -> dict:
    """Convert a vLLM-saved state dict to HuggingFace split-weight format.

    Parameters
    ----------
    state_dict:
        Raw dict loaded from pytorch_model.pth (saved by
        WorkerExtension.save_self_weights_to_disk).
    config:
        HuggingFace model config (e.g. model.config after
        AutoModelForCausalLM.from_pretrained).  Must expose
        hidden_size, num_attention_heads, num_key_value_heads.

    Returns
    -------
    A new state dict with all fused keys replaced by their HF equivalents.
    Non-fused keys are passed through unchanged.
    """
    head_dim = config.hidden_size // config.num_attention_heads
    q_size   = config.num_attention_heads    * head_dim
    kv_size  = config.num_key_value_heads    * head_dim

    new_sd: dict = {}
    for key, tensor in state_dict.items():
        if "self_attn.qkv_proj.weight" in key:
            prefix = key[: key.index("self_attn.qkv_proj.weight")]
            q, k, v = tensor.split([q_size, kv_size, kv_size], dim=0)
            new_sd[prefix + "self_attn.q_proj.weight"] = q
            new_sd[prefix + "self_attn.k_proj.weight"] = k
            new_sd[prefix + "self_attn.v_proj.weight"] = v
        elif "self_attn.qkv_proj.bias" in key:
            prefix = key[: key.index("self_attn.qkv_proj.bias")]
            q, k, v = tensor.split([q_size, kv_size, kv_size], dim=0)
            new_sd[prefix + "self_attn.q_proj.bias"] = q
            new_sd[prefix + "self_attn.k_proj.bias"] = k
            new_sd[prefix + "self_attn.v_proj.bias"] = v
        elif "mlp.gate_up_proj.weight" in key:
            prefix = key[: key.index("mlp.gate_up_proj.weight")]
            gate, up = tensor.chunk(2, dim=0)
            new_sd[prefix + "mlp.gate_proj.weight"] = gate
            new_sd[prefix + "mlp.up_proj.weight"]   = up
        else:
            new_sd[key] = tensor

    return new_sd
