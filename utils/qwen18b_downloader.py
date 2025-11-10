from huggingface_hub import snapshot_download
import os


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

snapshot_download(
    repo_id="Qwen/Qwen-1_8B-Chat",
    local_dir="models/qwen-1.8b",      # Relative path within project
    local_dir_use_symlinks=False,      # Copy files for easy packaging
    revision="v1.1"                    # Lock version to prevent update issues
)
print("✅ Qwen-1.8B-Chat downloaded to models/qwen-1.8b/")