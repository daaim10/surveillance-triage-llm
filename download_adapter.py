from huggingface_hub import snapshot_download
import os

snapshot_download(
    repo_id="daaim10/surveillance-triage-rank16-lr2e4-ep3",
    token="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",
    local_dir="adapters/rank16_lr2e4_ep3",
)

print("Adapter downloaded to adapters/rank16_lr2e4_ep3")
print("Files:", os.listdir("adapters/rank16_lr2e4_ep3"))