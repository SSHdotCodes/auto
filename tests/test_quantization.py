"""Packed checkpoint loading, numerical parity and bounded scratch allocations."""

import json

import pytest
import torch
from safetensors.torch import save_file
from test_attention import LargestTensor, tiny_config
from transformers import ModernBertForSequenceClassification
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from auto_gate import attention, quantization
from auto_gate.attention import bound_mlp_memory, chunked_attention, forward_classifier
from auto_gate.config import MODELS, Settings
from auto_gate.model import model_files
from auto_gate.quantization import QuantEmbedding, QuantLinear, load_quantized, quantized_modules


def checkpoint(path, bits):
    torch.manual_seed(17)
    config = tiny_config()
    config.id2label = {0: "approve", 1: "deny"}
    config.save_pretrained(path)
    config._attn_implementation = "sdpa"
    model = ModernBertForSequenceClassification(config).eval()
    names = quantized_modules(model)
    state = dict(model.state_dict())
    for name in names:
        weight = state.pop(name + ".weight")
        shape = weight.shape
        grouped = weight.reshape(shape[0], -1, 8) if bits == 4 else weight
        scales = (grouped.abs().amax(-1) / (7 if bits == 4 else 127)).clamp_min(1e-6).half()
        q = (
            (grouped / scales.float().unsqueeze(-1))
            .round()
            .clamp(-8 if bits == 4 else -127, 7 if bits == 4 else 127)
        )
        # Independent dense reference, using the stored FP16 scales before rounding to compute dtype.
        restored = (q * scales.float().unsqueeze(-1)).reshape(shape)
        model.get_submodule(name).weight.data.copy_(restored)
        q = q.reshape(shape).to(torch.int8)
        if bits == 4:
            u = (q.to(torch.int16) + 8).byte()
            q = u[:, ::2] | (u[:, 1::2] << 4)
        state[name + ".qweight"] = q
        state[name + ".scales"] = scales
    filename = f"auto_quant_int{bits}.safetensors"
    save_file(state, path / filename)
    (path / "quant_config.json").write_text(
        json.dumps(
            {
                "format": "auto-quant-v1",
                "bits": bits,
                "group_size": 8 if bits == 4 else None,
                "symmetric": True,
                "modules": names,
                "weights": filename,
            }
        )
    )
    ALL_ATTENTION_FUNCTIONS.register("auto_chunked", chunked_attention)
    return model


def load(path, bits):
    return bound_mlp_memory(
        load_quantized(path, bits=bits, device="cpu", dtype=torch.float32, attn_implementation="auto_chunked")
    )


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("length", [1, 17, 65, 263])
def test_packed_model_matches_independent_dense_reference(tmp_path, monkeypatch, bits, length):
    reference = checkpoint(tmp_path, bits)
    model = load(tmp_path, bits)
    monkeypatch.setattr(attention, "MLP_CHUNK", 7)
    monkeypatch.setattr(quantization, "EMBEDDING_CHUNK", 5)
    monkeypatch.setattr(attention, "_FUSED", {})
    monkeypatch.setattr(attention, "SCORE_BUDGET", 4 * 263 * 4 * 3)
    monkeypatch.setattr(attention, "LOCAL_BLOCK", 7)
    ids = torch.randint(1, 64, (1, length))
    with torch.inference_mode():
        expected = reference(input_ids=ids).logits
        actual = forward_classifier(model, ids)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert all(m._expanded is None for m in model.modules() if isinstance(m, QuantLinear))
    assert not any(
        ".weight" in name and name.startswith("model.layers") and value.ndim == 2
        for name, value in model.state_dict().items()
    )


@pytest.mark.parametrize("bits", [4, 8])
def test_embedding_dtype_follows_device_fallback(tmp_path, bits):
    reference = checkpoint(tmp_path, bits)
    model = load(tmp_path, bits).to(dtype=torch.bfloat16).to(dtype=torch.float32)
    embedding = model.model.embeddings.tok_embeddings
    assert isinstance(embedding, QuantEmbedding)
    ids = torch.tensor([[1, 3, 1]])
    with torch.inference_mode():
        actual = embedding(ids)
    assert actual.dtype == torch.float32
    # Module.to(bfloat16) also rounds scales; dtype must still follow the final CPU destination.
    assert torch.isfinite(actual).all()
    assert reference.model.embeddings.tok_embeddings(ids).shape == actual.shape


@pytest.mark.parametrize("bits", [4, 8])
def test_mlp_reuses_weights_only_within_layer_and_cleans_up_on_error(tmp_path, monkeypatch, bits):
    checkpoint(tmp_path, bits)
    model = load(tmp_path, bits)
    monkeypatch.setattr(attention, "MLP_CHUNK", 5)
    mlp = model.model.layers[0].mlp
    calls = []
    original = mlp._auto_full_forward

    def record(hidden):
        calls.append((mlp.Wi._expanded, mlp.Wo._expanded))
        return original(hidden)

    mlp._auto_full_forward = record
    with torch.inference_mode():
        mlp(torch.randn(1, 17, 32))
    assert len(calls) == 4
    assert calls[0][0] is not None
    assert all(a is calls[0][0] and b is calls[0][1] for a, b in calls)
    assert mlp.Wi._expanded is mlp.Wo._expanded is None

    def fail(hidden):
        raise RuntimeError("simulated backend failure")

    mlp._auto_full_forward = fail
    with torch.inference_mode(), pytest.raises(RuntimeError, match="simulated"):
        mlp(torch.randn(1, 17, 32))
    assert mlp.Wi._expanded is mlp.Wo._expanded is None


@pytest.mark.parametrize("bits", [4, 8])
def test_quantized_allocations_are_linear_with_no_fused_kernel(tmp_path, monkeypatch, bits):
    checkpoint(tmp_path, bits)
    model = load(tmp_path, bits)
    monkeypatch.setattr(attention, "_FUSED", {})
    monkeypatch.setattr(attention, "SCORE_BUDGET", 4 * 4096 * 4 * 64)
    monkeypatch.setattr(attention, "MLP_CHUNK", 256)
    monkeypatch.setattr(quantization, "EMBEDDING_CHUNK", 256)
    sizes = []
    for n in (4096, 8192):
        with torch.inference_mode(), LargestTensor() as mode:
            forward_classifier(model, torch.randint(1, 64, (1, n)))
        assert mode.numel < n * n // 8
        sizes.append(mode.numel)
    assert sizes[1] <= 2.1 * sizes[0]


@pytest.mark.parametrize(
    "change",
    [
        {"format": "unknown"},
        {"bits": 3},
        {"group_size": 0},
        {"group_size": 7},
        {"weights": "../elsewhere.safetensors"},
        {"modules": []},
        {"symmetric": False},
    ],
)
def test_rejects_invalid_checkpoint_metadata(tmp_path, change):
    checkpoint(tmp_path, 4)
    path = tmp_path / "quant_config.json"
    spec = json.loads(path.read_text())
    spec.update(change)
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):
        load(tmp_path, 4)


def test_pins_download_only_data_and_settings_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTO_HOME", str(tmp_path))
    for name, entry in MODELS.items():
        settings = Settings(model=name)
        settings.save()
        assert Settings.load() == settings
        files = model_files(settings)
        assert not any(f.endswith(".py") for f in files)
        assert len(entry["revision"]) == 40
        assert ("quant_config.json" in files) == ("bits" in entry)


@pytest.mark.parametrize("bits", [4, 8])
def test_classifier_gpu_failure_retries_packed_model_on_cpu(tmp_path, monkeypatch, bits):
    from auto_gate.model import Classifier
    from auto_gate.schema import ScoreRequest

    checkpoint(tmp_path, bits)
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: str(tmp_path))

    class Tokenizer:
        cls_token_id, sep_token_id = 1, 2

        def __call__(self, *args, **kwargs):
            return {"input_ids": torch.tensor([[1, 3, 5, 2]])}

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: Tokenizer())
    classifier = Classifier(Settings(model=f"auto-200m-2-int{bits}", device="cpu"))
    request = ScoreRequest(user_request="Read README", call={"tool": "read", "args": "README"})
    expected = classifier.score(request)
    classifier.device = "cuda"
    original_to = torch.Tensor.to

    def unavailable(tensor, *args, **kwargs):
        if args and args[0] == "cuda":
            raise RuntimeError("simulated unavailable GPU")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", unavailable)
    actual = classifier.score(request)
    assert actual.decision == expected.decision
    assert actual.p_deny == pytest.approx(expected.p_deny, abs=1e-6)
    assert classifier.info()["quantization"] == f"int{bits}"
    assert classifier.device == "cpu" and classifier.fallback is not None
    embedding = classifier.model.model.embeddings.tok_embeddings
    assert embedding.qweight.dtype == (torch.uint8 if bits == 4 else torch.int8)
    assert embedding._compute.dtype == torch.float32


def test_kernel_trust_option_is_not_forwarded_to_modernbert_constructor(tmp_path):
    checkpoint(tmp_path, 4)
    model = load_quantized(
        tmp_path,
        bits=4,
        device="cpu",
        dtype=torch.float32,
        attn_implementation="auto_chunked",
        allow_all_kernels=True,
    )
    with torch.inference_mode():
        assert torch.isfinite(forward_classifier(model, torch.tensor([[1, 2]]))).all()
