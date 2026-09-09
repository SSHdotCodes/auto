import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from platformdirs import user_data_path

MODEL_ID = "ProCreations/auto-0.4b-2"
MODEL_REVISION = "456e153dad2b12db14babb1dbdc8f8ece607e80a"
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
    max_tokens: int = 8192
    threshold: float = 0.5
    timeout: float = 120.0
    model_id: str = MODEL_ID
    revision: str = MODEL_REVISION

    def __post_init__(self):
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps (ROCm uses cuda)")
        if self.attention not in {"chunked", "flash"}:
            raise ValueError("attention must be chunked or flash")
        if not 128 <= self.max_tokens <= 65536:
            raise ValueError("max_tokens must be between 128 and 65536")
        if not 0 < self.threshold < 1 or not 1 <= self.timeout <= 600:
            raise ValueError("threshold must be in (0,1); timeout must be 1–600 seconds")
        if self.model_id != MODEL_ID or self.revision != MODEL_REVISION:
            raise ValueError("This release uses a pinned, verified Auto model. Upgrade Auto to change it.")

    def save(self):
        write_json(home() / "config.json", asdict(self))

    @classmethod
    def load(cls):
        path = home() / "config.json"
        return cls(**json.loads(path.read_text(encoding="utf-8"))) if path.exists() else cls()
