from huggingface_hub import snapshot_download
from dotenv import load_dotenv
import os

# Loads variables from .env into environment
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN not found. "
        "Make sure your .env file exists and has HF_TOKEN set."
    )

snapshot_download(
    repo_id="daaim10/surveillance-triage-rank16-lr2e4-ep3",
    token=HF_TOKEN,
    local_dir="adapters/rank16_lr2e4_ep3",
)

print("Downloaded successfully")
print("Files:", os.listdir("adapters/rank16_lr2e4_ep3"))