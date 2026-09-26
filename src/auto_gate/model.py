import gc
import os
import threading

from .config import FLASH, Settings, home
from .schema import Decision, ScoreRequest, serialize

MODEL_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]


def download(settings: Settings):
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        settings.model_id, revision=settings.revision, allow_patterns=MODEL_FILES, cache_dir=home() / "models"
    )
    if settings.attention == "flash":
        download_flash_kernel()
    return path


def download_flash_kernel():
    """Fetch the pinned FlashAttention kernel so the runtime can load it offline."""
    import subprocess
    import sys

    # kernels 0.16 sends an invalid (empty) User-Agent when HF_HUB_DISABLE_TELEMETRY is set.
    env = {k: v for k, v in os.environ.items() if k != "HF_HUB_DISABLE_TELEMETRY"}
    env["KERNELS_CACHE"] = str(home() / "kernels")
    repo, revision = FLASH.split("@")
    code = "import sys; from kernels import install_kernel; install_kernel(sys.argv[1], revision=sys.argv[2])"
    subprocess.run([sys.executable, "-c", code, repo, revision], check=True, env=env)


def flash_kernel_path():
    """The pinned kernel snapshot written by `download_flash_kernel` (Hugging Face cache layout)."""
    repo, revision = FLASH.split("@")
    return home() / "kernels" / ("kernels--" + repo.replace("/", "--")) / "snapshots" / revision


class Classifier:
    def __init__(self, settings: Settings):
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        # Inference never touches the network: weights and any optional kernel were fetched by `auto download`.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        from .attention import bound_mlp_memory, chunked_attention

        self.settings = settings
        self.lock = threading.Lock()
        self.fallback = None
        self.device = settings.device
        if self.device == "auto":
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device, self.fallback = "cpu", "CUDA/ROCm unavailable; using CPU"
        if self.device == "mps" and not torch.backends.mps.is_available():
            self.device, self.fallback = "cpu", "MPS unavailable; using CPU"
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
        self.path = snapshot_download(
            settings.model_id,
            revision=settings.revision,
            allow_patterns=MODEL_FILES,
            cache_dir=home() / "models",
            local_files_only=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path, local_files_only=True, trust_remote_code=False
        )
        ALL_ATTENTION_FUNCTIONS.register("auto_chunked", chunked_attention)
        self.attention = "chunked"
        if settings.attention == "flash" and self.device == "cuda" and not torch.version.hip:
            kernel = flash_kernel_path()
            if kernel.is_dir():
                # kernels 0.16 cannot resolve a single-variant cached snapshot offline (huggingface_hub reports it
                # incomplete), so point it at the pinned local copy instead.
                repo = FLASH.split("@")[0]
                overrides = os.environ.get("LOCAL_KERNELS", "")
                if repo + "=" not in overrides:
                    os.environ["LOCAL_KERNELS"] = ":".join(filter(None, [overrides, f"{repo}={kernel}"]))
                self.attention = "flash"
            else:
                self.fallback = (
                    "FlashAttention kernel not downloaded (run `auto download`); using chunked attention"
                )
        dtype = (
            torch.bfloat16
            if self.device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
            else torch.float32
        )
        plan = [(self.device, self.attention, dtype)]
        if self.attention == "flash":
            plan.append((self.device, "chunked", dtype))
        if self.device != "cpu":
            plan.append(("cpu", "chunked", torch.float32))
        warmup = torch.tensor([[self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]])
        for step, (device, attention, dtype) in enumerate(plan):
            try:
                self.model = (
                    AutoModelForSequenceClassification.from_pretrained(
                        self.path,
                        dtype=dtype,
                        local_files_only=True,
                        trust_remote_code=False,
                        attn_implementation=FLASH if attention == "flash" else "auto_chunked",
                        # The kernel is pinned to an immutable commit and was fetched explicitly at download time.
                        **({"allow_all_kernels": True} if attention == "flash" else {}),
                    )
                    .to(device)
                    .eval()
                )
                self.device, self.attention = device, attention
                # A tiny forward pass surfaces kernel and backend failures now instead of on the first request.
                with torch.inference_mode():
                    self._forward(warmup.to(device))
                break
            except (RuntimeError, ImportError, ValueError, KeyError, OSError, NotImplementedError) as error:
                if step == len(plan) - 1:
                    raise
                self.model = None
                gc.collect()
                following = plan[step + 1]
                self.fallback = (
                    f"{attention} attention on {device} failed to initialize ({type(error).__name__}); "
                    f"using {following[1]} attention on {following[0]}"
                )
        if self.model.config.id2label != {0: "approve", 1: "deny"}:
            raise ValueError("Unexpected model label mapping")
        # Exact; keeps the per-token MLP from materializing full-length intermediates on long inputs.
        bound_mlp_memory(self.model)

    def info(self):
        import torch

        return {
            "device": self.device,
            "backend": "rocm" if self.device == "cuda" and torch.version.hip else self.device,
            "attention": self.attention,
            "dtype": str(next(self.model.parameters()).dtype),
            "max_tokens": self.settings.max_tokens,
            "model": self.settings.model_id,
            "revision": self.settings.revision,
            "fallback": self.fallback,
        }

    def _forward(self, ids):
        from .attention import forward_classifier

        if self.attention == "chunked":
            return forward_classifier(self.model, ids)
        return self.model(input_ids=ids).logits

    def score(self, request: ScoreRequest) -> Decision:
        import torch

        if not request.context_complete or not request.user_request.strip():
            return Decision(decision="review", reason="User context is incomplete; host review required")
        ids = self.tokenizer(serialize(request), return_tensors="pt", truncation=False)["input_ids"]
        count = ids.shape[-1]
        if count > self.settings.max_tokens:
            return Decision(
                decision="review",
                tokens=count,
                reason=f"Context exceeds the {self.settings.max_tokens}-token runtime budget; no text was dropped",
            )
        # Queue size is constrained by the HTTP layer; one model evaluation per process.
        with self.lock, torch.inference_mode():
            try:
                logits = self._forward(ids.to(self.device))
            except (RuntimeError, NotImplementedError):
                if self.device == "cpu":
                    return Decision(decision="review", reason="Inference failed; host review required")
                self.model = self.model.to("cpu", dtype=torch.float32)
                self.device, self.attention = "cpu", "chunked"
                self.model.set_attn_implementation("auto_chunked")
                self.fallback = "GPU inference failed; using CPU"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                try:
                    logits = self._forward(ids)
                except (RuntimeError, NotImplementedError):
                    return Decision(decision="review", reason="Inference failed; host review required")
            if not torch.isfinite(logits).all():
                return Decision(decision="review", reason="Non-finite model output; host review required")
            probability = logits.detach().cpu().double().softmax(-1)[0, 1].item()
            if self.device == "mps" and count > 4096:
                # Return long-request buffers to the system instead of keeping them in MPS's cache.
                torch.mps.empty_cache()
        verdict = "deny" if probability >= self.settings.threshold else "approve"
        return Decision(
            decision=verdict,
            p_deny=probability,
            reason=f"Auto: {verdict} (P(deny)={probability:.4f})",
            tokens=count,
            device=self.device,
            model=self.settings.model_id,
            revision=self.settings.revision,
        )
