"""Decoder classifiers (auto-3b, SmolLM3) through the real runtime, on a tiny random model."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from auto_gate.config import Settings  # noqa: E402
from auto_gate.schema import ScoreRequest, serialize  # noqa: E402

TEXTS = ["Delete the merged branches in the training repo.", "git push origin --delete exp-7", "rm -rf /"]


@pytest.fixture(scope="module")
def tiny_decoder(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, SmolLM3Config, SmolLM3ForSequenceClassification

    path = tmp_path_factory.mktemp("tiny-smollm3")
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=["[PAD]", "[UNK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(
        TEXTS * 20 + ["### PROPOSED TOOL CALL tool: args: USER REQUEST AGENT HISTORY"], trainer
    )
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]")
    fast.save_pretrained(path)
    torch.manual_seed(0)
    config = SmolLM3Config(
        vocab_size=fast.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        pad_token_id=0,
        num_labels=2,
        id2label={0: "approve", 1: "deny"},
        label2id={"approve": 0, "deny": 1},
    )
    SmolLM3ForSequenceClassification(config).save_pretrained(path, safe_serialization=True)
    return path


def request(history=()):
    return ScoreRequest(
        user_request=TEXTS[0], call={"tool": "bash", "args": {"command": TEXTS[1]}}, history=list(history)
    )


def test_decoder_classifier_matches_transformers(tiny_decoder, tmp_path, monkeypatch):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from auto_gate.model import Classifier

    monkeypatch.setenv("AUTO_HOME", str(tmp_path))
    monkeypatch.setenv("AUTO_MODEL_DIR", str(tiny_decoder))
    classifier = Classifier(Settings(device="cpu"))
    info = classifier.info()
    assert info["architecture"] == "decoder" and info["attention"] == "sdpa"
    assert info["dtype"] == "torch.bfloat16" and info["model"].startswith("local:")
    decision = classifier.score(request())
    reference = AutoModelForSequenceClassification.from_pretrained(tiny_decoder, dtype=torch.bfloat16).eval()
    ids = AutoTokenizer.from_pretrained(tiny_decoder)(serialize(request()), return_tensors="pt")["input_ids"]
    with torch.inference_mode():
        expected = reference(input_ids=ids).logits.float().softmax(-1)[0, 1].item()
    assert decision.decision in {"approve", "deny"}
    assert decision.p_deny == pytest.approx(expected, abs=1e-3)


def test_decoder_long_sessions_are_fitted(tiny_decoder, tmp_path, monkeypatch):
    from auto_gate.model import Classifier

    monkeypatch.setenv("AUTO_HOME", str(tmp_path))
    monkeypatch.setenv("AUTO_MODEL_DIR", str(tiny_decoder))
    classifier = Classifier(Settings(device="cpu", fit_tokens=512))
    history = [{"tool": "bash", "args": {"n": i}, "result": TEXTS[2] * 5} for i in range(200)]
    decision = classifier.score(request(history))
    assert decision.decision in {"approve", "deny"} and decision.tokens <= 512
    assert decision.trimmed["actions"] > 0


def test_unpublished_model_explains_itself(tmp_path, monkeypatch):
    from auto_gate import config
    from auto_gate.model import download

    monkeypatch.setenv("AUTO_HOME", str(tmp_path))
    monkeypatch.delenv("AUTO_MODEL_DIR", raising=False)
    entry = {"id": "ProCreations/auto-next", "revision": None, "parameters": "-", "download": "-"}
    monkeypatch.setitem(config.MODELS, "auto-next", entry)
    with pytest.raises(ValueError, match="not published yet"):
        download(Settings(model="auto-next"))
