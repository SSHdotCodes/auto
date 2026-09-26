import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from platformdirs import user_data_path

# Pinned, verified models. Upgrade Auto to change a pin.
MODELS = {
    "auto-0.4b-2": {
        "id": "ProCreations/auto-0.4b-2",
        "revision": "456e153dad2b12db14babb1dbdc8f8ece607e80a",
        "parameters": "395.8M",
        "download": "about 0.8 GB",
    },
    "auto-200m-2": {
        "id": "ProCreations/auto-200m-2",
        "revision": "0bbb929f07564a22f6bd67d6624467bca6f1ab14",
        "parameters": "149.6M",
        "download": "about 0.3 GB",
    },
}
DEFAULT_MODEL = "auto-0.4b-2"
MODEL_ID = MODELS[DEFAULT_MODEL]["id"]
MODEL_REVISION = MODELS[DEFAULT_MODEL]["revision"]
FLASH = "kernels-community/flash-attn2@81fb77c12b2ad5d69380669b46739d5868614502"


def home() -> Path:
    path = Path(os.environ.get("AUTO_HOME") or user_data_path("auto", "SSHDotCodes"))
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as stream:
        if os.name != "nt":
            os.chmod(temp, 0o600)
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temp, path)


@dataclass
class Settings:
    device: str = "auto"
    attention: str = "chunked"
    max_tokens: int = 32768
    threshold: float = 0.5
    timeout: float = 120.0
    model: str = DEFAULT_MODEL

    @property
    def model_id(self) -> str:
        return MODELS[self.model]["id"]

    @property
    def revision(self) -> str:
        return MODELS[self.model]["revision"]

    def __post_init__(self):
        if self.model not in MODELS:
            raise ValueError("model must be one of: " + ", ".join(MODELS))
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps (ROCm uses cuda)")
        if self.attention not in {"chunked", "flash"}:
            raise ValueError("attention must be chunked or flash")
        if not 128 <= self.max_tokens <= 65536:
            raise ValueError("max_tokens must be between 128 and 65536")
        if not 0 < self.threshold < 1 or not 1 <= self.timeout <= 600:
            raise ValueError("threshold must be in (0,1); timeout must be 1–600 seconds")

    def save(self):
        write_json(home() / "config.json", asdict(self))

    @classmethod
    def load(cls):
        path = home() / "config.json"
        if not path.exists():
            return cls()
        values = json.loads(path.read_text(encoding="utf-8"))
        # v0.1.0 stored the pinned repository and revision; map them to the model name. Its saved 8,192-token
        # default was a memory precaution that the bounded attention path no longer needs.
        model_id = values.pop("model_id", None)
        values.pop("revision", None)
        if "model" not in values and model_id:
            values["model"] = next((name for name, m in MODELS.items() if m["id"] == model_id), model_id)
            if values.get("max_tokens") == 8192:
                values["max_tokens"] = cls.max_tokens
        return cls(**values)
