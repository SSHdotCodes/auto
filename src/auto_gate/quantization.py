"""Portable, weight-only auto-quant-v1 inference. No downloaded Python is executed.

Integers stay packed between calls. Dequantization multiplies in FP32 before casting,
matching the checkpoint format on CPU, CUDA/ROCm and MPS. Only one linear weight
(or the current chunked MLP's two weights) is expanded at a time.
"""

import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

EMBEDDING_CHUNK = 4096


def dequantize(qweight, scales, bits, group_size, dtype):
    if bits == 4:
        low = (qweight & 15).to(torch.int8) - 8
        high = (qweight >> 4).to(torch.int8) - 8
        qweight = torch.stack((low, high), dim=-1).flatten(-2)
        shape = qweight.shape
        weight = qweight.float().reshape(*shape[:-1], shape[-1] // group_size, group_size)
        weight = (weight * scales.float().unsqueeze(-1)).reshape(shape)
    else:
        weight = qweight.float() * scales.float().unsqueeze(-1)
    return weight.to(dtype)


class QuantLinear(nn.Module):
    def __init__(self, source, bits, group_size):
        super().__init__()
        self.in_features, self.out_features = source.in_features, source.out_features
        self.bits, self.group_size = bits, group_size
        columns = self.in_features // 2 if bits == 4 else self.in_features
        scale_shape = (
            (self.out_features, self.in_features // group_size) if bits == 4 else (self.out_features,)
        )
        self.register_buffer(
            "qweight", torch.empty(self.out_features, columns, dtype=torch.uint8 if bits == 4 else torch.int8)
        )
        self.register_buffer("scales", torch.empty(scale_shape, dtype=torch.float16))
        self.register_buffer(
            "bias", torch.empty(self.out_features, dtype=torch.float16) if source.bias is not None else None
        )
        self._expanded = None

    @contextmanager
    def expanded(self, dtype):
        """Reuse this weight across the current MLP's chunks, never across layers/requests."""
        try:
            self._expanded = dequantize(self.qweight, self.scales, self.bits, self.group_size, dtype)
            yield
        finally:
            self._expanded = None

    def forward(self, inputs):
        weight = self._expanded
        if weight is None:
            weight = dequantize(self.qweight, self.scales, self.bits, self.group_size, inputs.dtype)
        return F.linear(inputs, weight, None if self.bias is None else self.bias.to(inputs.dtype))


class QuantEmbedding(nn.Module):
    def __init__(self, source, bits, group_size, dtype):
        super().__init__()
        self.num_embeddings, self.embedding_dim = source.num_embeddings, source.embedding_dim
        self.bits, self.group_size = bits, group_size
        columns = self.embedding_dim // 2 if bits == 4 else self.embedding_dim
        scale_shape = (
            (self.num_embeddings, self.embedding_dim // group_size) if bits == 4 else (self.num_embeddings,)
        )
        self.register_buffer(
            "qweight",
            torch.empty(self.num_embeddings, columns, dtype=torch.uint8 if bits == 4 else torch.int8),
        )
        self.register_buffer("scales", torch.empty(scale_shape, dtype=torch.float16))
        # Unlike a Python dtype attribute, this follows Module.to during GPU -> CPU fallback.
        self.register_buffer("_compute", torch.empty(0, dtype=dtype), persistent=False)

    def forward(self, input_ids):
        ids = input_ids.reshape(-1)
        output = torch.empty(ids.numel(), self.embedding_dim, device=ids.device, dtype=self._compute.dtype)
        for start in range(0, ids.numel(), EMBEDDING_CHUNK):
            chunk = ids[start : start + EMBEDDING_CHUNK]
            output[start : start + EMBEDDING_CHUNK] = dequantize(
                self.qweight[chunk], self.scales[chunk], self.bits, self.group_size, self._compute.dtype
            )
        return output.view(*input_ids.shape, self.embedding_dim)


def quantized_modules(model):
    return ["model.embeddings.tok_embeddings"] + [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and (name.startswith("model.layers.") or name == "head.dense")
    ]


def load_quantized(path, *, bits, device, dtype, attn_implementation, **kwargs):
    """Build on meta, replace quantized modules, then strictly load the pinned local tensors."""
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForSequenceClassification
    from transformers.models.modernbert.modeling_modernbert import ModernBertRotaryEmbedding

    path = Path(path)
    spec = json.loads((path / "quant_config.json").read_text(encoding="utf-8"))
    filename = f"auto_quant_int{bits}.safetensors"
    if (
        spec.get("format") != "auto-quant-v1"
        or spec.get("bits") != bits
        or bits not in (4, 8)
        or spec.get("weights") != filename
        or spec.get("symmetric") is not True
    ):
        raise ValueError("Unsupported Auto quantization configuration")
    group = spec.get("group_size")
    if bits == 4 and (type(group) is not int or group <= 0):
        raise ValueError("Invalid int4 group size")
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if config.model_type != "modernbert":
        raise ValueError("Auto quantization requires ModernBERT")
    with torch.device("meta"):
        model = AutoModelForSequenceClassification.from_config(
            config, dtype=dtype, attn_implementation=attn_implementation, trust_remote_code=False, **kwargs
        )
        expected = quantized_modules(model)
        if not isinstance(spec.get("modules"), list) or sorted(spec["modules"]) != sorted(expected):
            raise ValueError("Unexpected quantized module list")
        for name in expected:
            parent, _, child = name.rpartition(".")
            source = model.get_submodule(name)
            width = source.embedding_dim if isinstance(source, nn.Embedding) else source.in_features
            if bits == 4 and (width % 2 or width % group):
                raise ValueError(f"Invalid int4 dimensions for {name}")
            replacement = (
                QuantEmbedding(source, bits, group, dtype)
                if isinstance(source, nn.Embedding)
                else QuantLinear(source, bits, group)
            )
            setattr(model.get_submodule(parent), child, replacement)
    model.to_empty(device=device)
    state = load_file(path / filename, device="cpu")
    # load_state_dict would silently cast corrupt packed integers, so check their storage type first.
    for name, value in model.state_dict().items():
        if name.endswith((".qweight", ".scales")) and (name not in state or state[name].dtype != value.dtype):
            raise ValueError(f"Unexpected quantized tensor dtype: {name}")
    model.load_state_dict(state, strict=True)
    del state
    with torch.device(device):
        # Nonpersistent RoPE buffers are computed, not part of the checkpoint.
        model.model.rotary_emb = ModernBertRotaryEmbedding(config=config)
    return model.eval()
