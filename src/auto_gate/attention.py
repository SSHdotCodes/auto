"""Exact bidirectional ModernBERT attention whose memory grows linearly with context, on every PyTorch backend.

One unpadded example at a time. No N x N score matrix or mask is ever allocated:

- Global layers use PyTorch's fused scaled-dot-product attention without a mask. On CUDA/ROCm and CPU only the
  memory-efficient fused kernels are allowed; if none applies (and always on MPS), queries are processed in blocks
  sized so that even the plain math kernel stays within a fixed score budget.
- Sliding-window layers attend each block of queries only to its own window of keys.
- The per-token MLP is evaluated in sequence chunks for long inputs.

No FlashAttention extension, CUDA compiler, or vendor-specific code is required.
"""

import os
import types

import torch
import torch.nn.functional as F

# Upper bound for the temporary attention scores of one kernel call if a backend falls back to the math kernel.
# GPUs get larger blocks (fewer launches); measured fastest on Apple MPS at 64k tokens with the same peak memory.
_BUDGET_MB = os.environ.get("AUTO_ATTENTION_BUDGET_MB")
SCORE_BUDGET = int(_BUDGET_MB) * 2**20 if _BUDGET_MB else None
DEFAULT_BUDGET = {"cpu": 256 * 2**20}
GPU_BUDGET = 1024 * 2**20
LOCAL_BLOCK = 1024
MLP_CHUNK = 8192

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _FUSED = {
        "cuda": [
            b for b in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION") if hasattr(SDPBackend, b)
        ],
        "cpu": ["FLASH_ATTENTION"],
    }
except ImportError:  # pragma: no cover - torch < 2.3
    sdpa_kernel, _FUSED = None, {}
_UNAVAILABLE = set()


def _sdpa(query, key, value, mask, scaling):
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False, scale=scaling
    )


def _fused_global(query, key, value, scaling):
    """One unmasked call restricted to memory-efficient kernels; None if this device/dtype has none."""
    kind = query.device.type
    if sdpa_kernel is None or kind not in _FUSED or (kind, query.dtype) in _UNAVAILABLE:
        return None
    try:
        with sdpa_kernel([getattr(SDPBackend, b) for b in _FUSED[kind]]):
            return _sdpa(query, key, value, None, scaling)
    except RuntimeError:
        _UNAVAILABLE.add((kind, query.dtype))
        return None


def _rows(device, heads, keys, element_size):
    budget = SCORE_BUDGET or DEFAULT_BUDGET.get(device.type, GPU_BUDGET)
    return max(1, budget // max(1, heads * keys * max(4, element_size)))


def global_attention(query, key, value, scaling=None):
    out = _fused_global(query, key, value, scaling)
    if out is not None:
        return out
    n = query.shape[-2]
    block = _rows(query.device, query.shape[1], n, query.element_size())
    if block >= n:
        return _sdpa(query, key, value, None, scaling)
    out = torch.empty_like(query)
    for s in range(0, n, block):
        out[:, :, s : s + block] = _sdpa(query[:, :, s : s + block], key, value, None, scaling)
    return out


def local_attention(query, key, value, radius, scaling=None):
    """Each query i attends to keys j with |i - j| <= radius."""
    n = query.shape[-2]
    block = max(
        1,
        min(LOCAL_BLOCK, _rows(query.device, query.shape[1], LOCAL_BLOCK + 2 * radius, query.element_size())),
    )
    out, cached = torch.empty_like(query), None
    for start in range(0, n, block):
        end = min(n, start + block)
        low, high = max(0, start - radius), min(n, end + radius)
        interior = low == start - radius and high == end + radius and end - start == block
        if interior and cached is not None:
            mask = cached
        else:
            q_pos = torch.arange(start, end, device=query.device)[:, None]
            k_pos = torch.arange(low, high, device=query.device)[None, :]
            mask = (q_pos - k_pos).abs() <= radius
            if interior:
                cached = mask
        out[:, :, start:end] = _sdpa(
            query[:, :, start:end], key[:, :, low:high], value[:, :, low:high], mask, scaling
        )
    return out


def chunked_attention(
    module, query, key, value, attention_mask=None, scaling=None, dropout=0.0, sliding_window=None, **kwargs
):
    """Transformers attention interface (registered as ``auto_chunked``)."""
    if query.shape[0] != 1 or attention_mask is not None:
        raise ValueError("Auto chunked attention requires one unpadded example and no external mask")
    if dropout:
        raise ValueError("Auto's chunked attention is inference-only")
    if sliding_window is None:
        out = global_attention(query, key, value, scaling)
    else:
        # Transformers passes ModernBERT's half-window + 1 (FlashAttention's inclusive convention).
        out = local_attention(query, key, value, sliding_window - 1, scaling)
    return out.transpose(1, 2).contiguous(), None


def _chunked_mlp_forward(self, hidden_states):
    n = hidden_states.shape[-2]
    if n <= MLP_CHUNK or torch.is_grad_enabled():
        return self._auto_full_forward(hidden_states)
    # The GLU acts on each token independently, so chunking along the sequence is exact.
    return torch.cat(
        [self._auto_full_forward(hidden_states[..., s : s + MLP_CHUNK, :]) for s in range(0, n, MLP_CHUNK)],
        dim=-2,
    )


def bound_mlp_memory(model):
    """Evaluate each layer's MLP in sequence chunks during inference (idempotent)."""
    for layer in model.model.layers:
        mlp = layer.mlp
        if not hasattr(mlp, "_auto_full_forward"):
            mlp._auto_full_forward = mlp.forward
            mlp.forward = types.MethodType(_chunked_mlp_forward, mlp)
    return model


def forward_classifier(model, input_ids):
    """Mirror Transformers' classification head with no dense attention-mask construction."""
    hidden = model.model(
        input_ids=input_ids, attention_mask={"full_attention": None, "sliding_attention": None}
    )[0]
    pooling = model.config.classifier_pooling
    if pooling == "cls":
        pooled = hidden[:, 0]
    elif pooling == "mean":
        pooled = hidden.mean(dim=1)
    else:
        raise ValueError(f"Unsupported classifier pooling: {pooling}")
    return model.classifier(model.drop(model.head(pooled)))
