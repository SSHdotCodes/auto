"""Exact bidirectional SDPA in bounded query blocks, including ModernBERT's local window.

One unpadded example at a time. Does not allocate an N x N sliding-window mask.
No FlashAttention extension, CUDA compiler, or vendor-specific code is required.
"""

import torch
import torch.nn.functional as F


def chunked_attention(
    module, query, key, value, attention_mask=None, scaling=None, dropout=0.0, sliding_window=None, **kwargs
):
    if query.shape[0] != 1 or attention_mask is not None:
        raise ValueError("Auto chunked attention requires one unpadded example and no external mask")
    if dropout:
        raise ValueError("Auto's chunked attention is inference-only")
    n = query.shape[-2]
    # Bound worst-case temporary attention scores to ~32 MiB, even on math-only devices.
    bytes_per_row = max(1, query.shape[1] * n * max(4, query.element_size()))
    block = min(256, max(1, (32 * 1024 * 1024) // bytes_per_row))
    outputs = []
    radius = sliding_window - 1 if sliding_window is not None else None
    for start in range(0, n, block):
        end = min(n, start + block)
        low, high = (max(0, start - radius), min(n, end + radius)) if radius is not None else (0, n)
        mask = None
        if radius is not None:
            q_pos = torch.arange(start, end, device=query.device)[:, None]
            k_pos = torch.arange(low, high, device=query.device)[None, :]
            mask = (q_pos - k_pos).abs() <= radius
        outputs.append(
            F.scaled_dot_product_attention(
                query[:, :, start:end],
                key[:, :, low:high],
                value[:, :, low:high],
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
                scale=scaling,
            )
        )
    return torch.cat(outputs, dim=-2).transpose(1, 2).contiguous(), None


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
