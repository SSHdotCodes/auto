# Local runtime and integration protocol

```text
Agent pre-tool hook → small adapter → UTF-8 JSON over stdin
  → lightweight Python bridge → authenticated loopback HTTP
  → pinned ModernBERT classifier → approve / deny / review
```

The Python bridge never imports PyTorch. The persistent daemon holds one copy of the model. Its random available port and per-run bearer token are stored in a private user-data file. Startup is serialized with an OS-compatible file lock. Requests ignore HTTP proxy environment variables and go only to `127.0.0.1`. The daemon rejects browser Origin headers and unrecognized Host names. Do not publish this daemon or its management routes.

Models are fetched at pinned revisions, using safetensors and `trust_remote_code=False`: `ProCreations/auto-0.4b-2` at `456e153dad2b12db14babb1dbdc8f8ece607e80a` (default) or `ProCreations/auto-200m-2` at `0bbb929f07564a22f6bd67d6624467bca6f1ab14` (`auto configure --model auto-200m-2`). Runtime loading is local-only. `auto download` and installation are the explicit network steps. There is no analytics, external inference, or prompt logging in Auto. The host coding agent may have its own network use.

## JSON contract

```json
{
  "user_request": "Read the README and summarize setup instructions.",
  "call": {"tool": "read_file", "args": {"path": "README.md"}},
  "history": [
    {"tool": "list_files", "args": {"path": "."}, "result": "README.md, src/"}
  ],
  "context_complete": true
}
```

Send it to `auto score --file request.json`, or pipe JSON to `auto score`. The internal `auto bridge` command always emits one JSON response; malformed input and runtime failures return `review`. Adapters invoke the absolute environment Python as `python -m auto_gate bridge`, send arguments through stdin, and parse the response. They do not place user content in shell commands or process command-line arguments.

```json
{
  "decision": "approve",
  "p_deny": 0.02,
  "reason": "Auto: approve (P(deny)=0.0200)",
  "tokens": 97,
  "device": "cpu",
  "model": "ProCreations/auto-0.4b-2",
  "revision": "456e153dad2b12db14babb1dbdc8f8ece607e80a"
}
```

This response is illustrative. Probabilities and token counts depend on the actual input. A review response may omit them. Never convert a parser error, timeout, HTTP error, or unknown decision into approval. `approve` means continue through existing host policy; `deny` means block; `review` means ask a human or block if no reviewer is available.

For in-process Python consumers, `auto_gate.client.score(payload)` provides the same behavior. Native clients can also read the private runtime state and use authenticated `POST /v1/score`; the JSON CLI is simpler and keeps endpoint discovery out of other-language integrations. Authenticated `GET /health` reports actual backend and readiness. `/shutdown` is private management only. Neither tokens nor runtime state belong in bug reports or source control.

## Serialization and inference

Text sections appear in the trained order: `PROPOSED TOOL CALL`, `USER REQUEST`, then `AGENT HISTORY`. Arguments are deterministic compact JSON; tool output remains history, not authorization. The default budget is 32,768 tokens, configurable up to the model's 65,536 limit. Missing context, oversized input, invalid schemas, non-finite logits, queue saturation, and inference failures return review. The service limits request bodies to 2 MB and permits at most four pending evaluations.

Portable attention computes the exact bidirectional local/global attention pattern without allocating any N×N tensor. Global layers call PyTorch's fused scaled-dot-product attention without a mask; on CUDA/ROCm and CPU only memory-efficient fused kernels are allowed, and where none applies (and always on MPS) queries are processed in blocks sized from a score budget (256 MB on CPU, 1 GB on GPUs; `AUTO_ATTENTION_BUDGET_MB` overrides). With only PyTorch's plain math kernel, which also keeps softmax intermediates, the measured peak on an RTX PRO 6000 was 2.4–3.8 GB above the weights from 8k to 64k tokens. Sliding-window layers process 1,024-query blocks against only their ±64-token window. The GLU MLP, which acts on each token independently, runs in 8,192-token chunks. The same classifier pooling/head is used. Tests check numerical equivalence against the standard implementation on a small ModernBERT at every block boundary (fused and fallback paths), and that the largest tensor allocated grows linearly with context. Dtype and backend rounding can still change probabilities; published benchmark scores use the original BF16/FlashAttention reference. On an RTX PRO 6000 in BF16, the full 3,000-item benchmark through the portable path came within one item of each model's published FlashAttention predictions: auto-0.4b-2 as pinned by Auto 2,902 vs 2,903 (5 borderline decisions differ; identical on the 16k–64k slice), auto-0.4b-2 iteration-1 weights 2,910 vs 2,910 (every decision identical), auto-200m-2 2,889 vs 2,890 (3 differ). Two FlashAttention runs that only batch differently disagree by as much (2,909 vs 2,910).

Model weights are about 0.8 GB (auto-0.4b-2) or 0.3 GB (auto-200m-2) in BF16; CPU and MPS use FP32 parameters. Native BF16 CUDA/ROCm devices use BF16. An optional pinned FlashAttention kernel can be enabled explicitly; `auto download` stores it in Auto's data directory and the runtime loads it offline. If it is missing or fails to load, the runtime uses the portable attention on the same GPU and reports why. The normal installer has no custom compiler requirement.
