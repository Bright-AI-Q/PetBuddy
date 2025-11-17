#!/usr/bin/env python3
"""
Project: PetBuddy
Author: Bright Wang
File: qwen18b_downloader.py
====================================
Qwen-1.8B Model Download Utility

Purpose:
- Download and manage Qwen-1.8B-Chat model for pet-related LLM tasks
- Configure huggingface mirror for reliable downloads in China
- Provide version-locked model storage for reproducibility

Features:
1. Mirror Configuration: Automatic HF endpoint setup for stable downloads
2. Version Control: Specific model revision locking (v1.1)
3. Local Storage: Symlink-free copying for easy project packaging
4. Error Handling: Robust download with progress tracking
5. Integration Ready: Pre-configured for PetBuddy LLM modules

Model Details:
- Model: Qwen/Qwen-1_8B-Chat
- Revision: v1.1 (version locked)
- Size: ~3.5GB (float16 precision)
- Purpose: Pet knowledge QA and multi-modal reasoning
"""

from huggingface_hub import snapshot_download
import os

# Configure huggingface mirror for stable downloads in China
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Download Qwen-1.8B-Chat model with version locking
snapshot_download(
    repo_id="Qwen/Qwen-1_8B-Chat",
    local_dir="models/qwen-1.8b",      # Relative path within project
    local_dir_use_symlinks=False,      # Copy files for easy packaging
    revision="v1.1"                    # Lock version to prevent update issues
)
print("✅ Qwen-1.8B-Chat downloaded to models/qwen-1.8b/")