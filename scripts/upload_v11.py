"""Upload ChatTLA v11 GGUF to HuggingFace."""
import os
import time
from pathlib import Path

# Load .env
env_path = Path(__file__).resolve().parents[1] / ".env"
for line in env_path.read_text().splitlines():
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from huggingface_hub import HfApi

token = os.environ["HF_TOKEN"]
api = HfApi(token=token)
repo = "EricSpencer00/chattla-20b"
gguf = Path(__file__).resolve().parents[1] / "outputs" / "gguf" / "chattla-20b-Q8_0.gguf"

print(f"Uploading {gguf.name} ({gguf.stat().st_size / 1e9:.1f} GB) as gguf/chattla-20b-v11-Q8_0.gguf …", flush=True)
t0 = time.time()
url = api.upload_file(
    path_or_fileobj=str(gguf),
    path_in_repo="gguf/chattla-20b-v11-Q8_0.gguf",
    repo_id=repo,
    repo_type="model",
    commit_message="v11: add Q8_0 GGUF (21 GB, harmony TEMPLATE fix)",
)
elapsed = time.time() - t0
print(f"Done in {elapsed / 60:.1f} min — {url}")
