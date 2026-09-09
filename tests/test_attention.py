import copy

import pytest
import torch
from transformers import ModernBertConfig, ModernBertForSequenceClassification
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from auto_gate.attention import chunked_attention, forward_classifier


@pytest.mark.parametrize("length", [1, 17, 65, 263])
@pytest.mark.parametrize("pooling", ["cls", "mean"])
def test_chunked_matches_standard_bidirectional_model(length, pooling):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    config = ModernBertConfig(
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
    config._attn_implementation = "sdpa"
    reference = ModernBertForSequenceClassification(config).eval()
    ALL_ATTENTION_FUNCTIONS.register("auto_chunked", chunked_attention)
    custom_config = copy.deepcopy(config)
    custom_config._attn_implementation = "auto_chunked"
    custom = ModernBertForSequenceClassification(custom_config).eval()
    custom.load_state_dict(reference.state_dict())
    ids = torch.randint(1, 64, (1, length))
    with torch.inference_mode():
        expected = reference(input_ids=ids).logits
        actual = forward_classifier(custom, ids)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_local_attention_does_not_leak_across_window():
    q = torch.zeros(1, 1, 10, 4)
    k = q.clone()
    v = torch.arange(10, dtype=torch.float32)[None, None, :, None].expand(1, 1, 10, 4)
    output, _ = chunked_attention(None, q, k, v, sliding_window=3)
    assert output[0, 0, 0, 0].item() == pytest.approx(1)  # keys 0,1,2 only
    assert output[0, 9, 0, 0].item() == pytest.approx(8)  # keys 7,8,9 only
