# Local runtime and integration protocol

```text
Agent pre-tool hook → small adapter → UTF-8 JSON over stdin
  → lightweight Python bridge → authenticated loopback HTTP
  → pinned ModernBERT classifier → approve / deny / review
```

The Python bridge never imports PyTorch. The persistent daemon holds one copy of the model. Its random available port and per-run bearer token are stored in a private user-data file. Startup is serialized with an OS-compatible file lock. Requests ignore HTTP proxy environment variables and go only to `127.0.0.1`. The daemon rejects browser Origin headers and unrecognized Host names. Do not publish this daemon or its management routes.

The model is fetched from `ProCreations/auto-0.4b-2` at revision `456e153dad2b12db14babb1dbdc8f8ece607e80a`, using safetensors and `trust_remote_code=False`. Runtime loading is local-only. `auto download` and installation are the explicit network steps. There is no analytics, external inference, or prompt logging in Auto. The host coding agent may have its own network use.

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

Text sections appear in the trained order: `PROPOSED TOOL CALL`, `USER REQUEST`, then `AGENT HISTORY`. Arguments are deterministic compact JSON; tool output remains history, not authorization. The default budget is 8,192 tokens, configurable up to the model's 65,536 limit. Missing context, oversized input, invalid schemas, non-finite logits, queue saturation, and inference failures return review. The service limits request bodies to 2 MB and permits at most four pending evaluations.

Portable attention computes the exact bidirectional local/global attention pattern in query blocks. Local layers use ModernBERT's sliding-window boundary; global layers attend over the full sequence. No dense N×N local mask is created. The same classifier pooling/head is used. Numerical equivalence is tested against the standard implementation on a small ModernBERT with boundary-crossing sequences. Dtype and backend rounding can still change probabilities; published benchmark scores use the original BF16/FlashAttention reference.

Model weights are about 0.8 GB in BF16; CPU and MPS use FP32 parameters. Native BF16 CUDA/ROCm devices use BF16. An optional pinned FlashAttention kernel can be enabled explicitly, but the normal installer has no custom compiler requirement.
