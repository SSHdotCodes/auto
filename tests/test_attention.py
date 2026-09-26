import copy

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import ModernBertConfig, ModernBertForSequenceClassification
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from auto_gate import attention
from auto_gate.attention import bound_mlp_memory, chunked_attention, forward_classifier


def tiny_config(pooling="cls", max_positions=512):
    return ModernBertConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=4,
        max_position_embeddings=512,
        local_attention=16,
        classifier_pooling=pooling,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
    )


def models(pooling="cls", max_positions=512):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    config = tiny_config(pooling, max_positions)
    config._attn_implementation = "sdpa"
    reference = ModernBertForSequenceClassification(config).eval()
    ALL_ATTENTION_FUNCTIONS.register("auto_chunked", chunked_attention)
    custom_config = copy.deepcopy(config)
    custom_config._attn_implementation = "auto_chunked"
    custom = ModernBertForSequenceClassification(custom_config).eval()
    custom.load_state_dict(reference.state_dict())
    return reference, bound_mlp_memory(custom)


@pytest.fixture
def small_blocks(monkeypatch):
    """Force every block boundary and the non-fused fallback on a tiny model."""
    monkeypatch.setattr(attention, "_FUSED", {})
    monkeypatch.setattr(attention, "SCORE_BUDGET", 4 * 16 * 4 * 3)  # three global query rows at 16 keys
    monkeypatch.setattr(attention, "LOCAL_BLOCK", 7)
    monkeypatch.setattr(attention, "MLP_CHUNK", 5)


@pytest.mark.parametrize("length", [1, 17, 65, 263])
@pytest.mark.parametrize("pooling", ["cls", "mean"])
def test_chunked_matches_standard_bidirectional_model(length, pooling):
    reference, custom = models(pooling)
    ids = torch.randint(1, 64, (1, length))
    with torch.inference_mode():
        expected = reference(input_ids=ids).logits
        actual = forward_classifier(custom, ids)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("length", [1, 6, 7, 8, 17, 65, 263])
@pytest.mark.parametrize("pooling", ["cls", "mean"])
def test_blocked_fallback_matches_standard_model(small_blocks, length, pooling):
    reference, custom = models(pooling)
    ids = torch.randint(1, 64, (1, length))
    with torch.inference_mode():
        expected = reference(input_ids=ids).logits
        actual = forward_classifier(custom, ids)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


class LargestTensor(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.numel = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for value in out if isinstance(out, (tuple, list)) else [out]:
            if isinstance(value, torch.Tensor):
                self.numel = max(self.numel, value.numel())
        return out


@pytest.mark.parametrize("fallback", [False, True])
def test_largest_tensor_grows_linearly_with_context(monkeypatch, fallback):
    if fallback:
        monkeypatch.setattr(attention, "_FUSED", {})
        monkeypatch.setattr(attention, "SCORE_BUDGET", 4 * 4096 * 4 * 64)
        monkeypatch.setattr(attention, "MLP_CHUNK", 256)
    largest = {}
    for n in (4096, 8192):
        _, custom = models(max_positions=n)
        with torch.inference_mode(), LargestTensor() as mode:
            forward_classifier(custom, torch.randint(1, 64, (1, n)))
        largest[n] = mode.numel
        # A dense score matrix would hold 4 * n * n values and a dense mask n * n.
        assert mode.numel < n * n // 8
    assert largest[8192] <= 2.1 * largest[4096]
