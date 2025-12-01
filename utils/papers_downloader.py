#!/usr/bin/env python3
"""
Project: PetBuddy
File: utils/paper_downloader.py
Purpose: Automatically download all referenced papers for the WACV submission.
Usage: python utils/papers_downloader.py --output report/papers
"""

import os
import re
import requests
import argparse
from pathlib import Path
from tqdm import tqdm

# ================= PAPER LIST (Synchronized with Bibliography Checklist) =================
PAPERS = [
    # --- 1. The Problem Context: FGVC ---
    {
        "title": "[1] Bilinear CNN Models for Fine-grained Visual Recognition",
        "url": "https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Lin_Bilinear_CNN_Models_ICCV_2015_paper.pdf"
    },
    {
        "title": "[2] Learning to Navigate for Fine-grained Classification (NTS-Net)",
        "url": "https://arxiv.org/pdf/1809.00287.pdf"
    },
    {
        "title": "[3] Destruction and Construction Learning for Fine-grained Image Recognition (DCL)",
        "url": "https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Destruction_and_Construction_Learning_for_Fine-Grained_Image_Recognition_CVPR_2019_paper.pdf"
    },

    # --- 2. The Foundation: Lightweight Architectures ---
    {
        "title": "[4] MobileNetV2 Inverted Residuals and Linear Bottlenecks",
        "url": "https://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf"
    },
    {
        "title": "[5] EfficientNet Rethinking Model Scaling for Convolutional Neural Networks",
        "url": "https://arxiv.org/pdf/1905.11946.pdf"
    },

    # --- 3. Methodology A: Data Augmentation ---
    {
        "title": "[6] Random Erasing Data Augmentation",
        "url": "https://arxiv.org/pdf/1708.04896.pdf"
    },
    {
        "title": "[7] GridMask Data Augmentation",
        "url": "https://arxiv.org/pdf/2001.04086.pdf"
    },
    {
        "title": "[8] mixup Beyond Empirical Risk Minimization",
        "url": "https://arxiv.org/pdf/1710.09412.pdf"
    },

    # --- 4. Methodology B: Architecture Modules ---
    {
        "title": "[9] ECA-Net Efficient Channel Attention for Deep CNNs",
        "url": "https://arxiv.org/pdf/1910.03151.pdf"
    },
    {
        "title": "[10] Be Your Own Teacher Improve Performance via Self Distillation",
        "url": "https://arxiv.org/pdf/1905.08094.pdf"
    },
    {
        "title": "[11] Attention Is All You Need",
        "url": "https://arxiv.org/pdf/1706.03762.pdf"
    },

    # --- 5. Implementation: Pose Datasets ---
    {
        "title": "[12] Who Left the Dogs Out 3D Animal Reconstruction (StanfordExtra)",
        "url": "https://arxiv.org/pdf/2008.09464.pdf"
    },
    {
        "title": "[13] Cross-Domain Adaptation for Animal Pose Estimation",
        "url": "https://arxiv.org/pdf/1908.10069.pdf"
    },
    {
        "title": "[14] AP-10K A Benchmark for Animal Pose Estimation",
        "url": "https://arxiv.org/pdf/2108.12617.pdf"
    },

    # --- 6. Benchmarks & Data ---
    {
        "title": "[15] Cats and Dogs (Oxford-IIIT Pet Dataset)",
        "url": "https://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf"
    },
    {
        "title": "[16] Novel Dataset for Fine-Grained Image Categorization (Stanford Dogs)",
        "url": "http://people.csail.mit.edu/khosla/papers/fgvc2011.pdf"
    },
    {
        "title": "[17] Microsoft COCO Common Objects in Context",
        "url": "https://arxiv.org/pdf/1405.0312.pdf"
    },
    # [18] is Public Domain Sources (Websites), skipping PDF download.
    # [19] is Ultralytics YOLO (GitHub), skipping PDF download.

    # --- 7. Tools & Visualization ---
    {
        "title": "[20] Grad-CAM Visual Explanations from Deep Networks",
        "url": "https://arxiv.org/pdf/1610.02391.pdf"
    },

    # --- 8. Theoretical Basis: Curriculum Learning (NEW) ---
    {
        "title": "[21] Curriculum Learning (Bengio 2009)",
        "url": "https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf"
    },
    {
        "title": "[22] Curriculum by Smoothing (NeurIPS 2020)",
        "url": "https://arxiv.org/pdf/2003.01367.pdf"
    },

    # --- 9. Advanced Augmentation Comparisons (NEW) ---
    {
        "title": "[23] CutMix Regularization Strategy",
        "url": "https://arxiv.org/pdf/1905.04899.pdf"
    },
    {
        "title": "[24] SaliencyMix A Saliency-Guided Data Augmentation",
        "url": "https://arxiv.org/pdf/2006.01791.pdf"
    },

    # --- 10. Heavyweight & Modern Baselines (NEW) ---
    {
        "title": "[25] TransFG A Transformer Architecture for FGVC",
        "url": "https://ojs.aaai.org/index.php/AAAI/article/view/19965/19724"
    },
    {
        "title": "[26] MobileViT Light-weight General-purpose Vision Transformer",
        "url": "https://arxiv.org/pdf/2110.02178.pdf"
    },

    # --- 11. 2024-2025 Cutting Edge (NEW) ---
    {
        "title": "[27] MobileNetV4 Universal Models for the Mobile Ecosystem",
        "url": "https://arxiv.org/pdf/2404.10518.pdf"
    },
    {
        "title": "[28] YOLOv10 Real-Time End-to-End Object Detection",
        "url": "https://arxiv.org/pdf/2405.14458.pdf"
    },
    {
        "title": "[29] VMamba Visual State Space Models",
        "url": "https://arxiv.org/pdf/2401.10166.pdf"
    }
]


def sanitize_filename(name):
    """Remove illegal characters for filenames."""
    return re.sub(r'[\\/*?:"<>|]', "", name)


def download_file(url, output_path):
    """Download file with progress bar and headers."""
    # Headers to mimic a browser (prevents 403 Forbidden on arXiv/CVF)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f, tqdm(
                desc=output_path.name[:35]+"...", # Truncate name for cleaner display
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
        return True
    except Exception as e:
        print(f"❌ Failed to download: {url}")
        print(f"   Error: {e}")
        # Remove partial file if failed
        if output_path.exists():
            os.remove(output_path)
        return False


def main():
    parser = argparse.ArgumentParser(description="Download PetNet References")
    parser.add_argument("--output", type=str, default="report/papers", help="Directory to save PDFs")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📚 Starting download of {len(PAPERS)} papers to '{output_dir}/'...\n")

    success_count = 0

    for paper in PAPERS:
        safe_title = sanitize_filename(paper["title"])
        file_path = output_dir / f"{safe_title}.pdf"


        if file_path.exists():

            if file_path.stat().st_size > 0:
                print(f"⏩ Skipping (already exists): {paper['title']}")
                success_count += 1
                continue
            else:
                print(f"⚠️  Found empty file, re-downloading: {paper['title']}")
                os.remove(file_path)

        print(f"⬇️  Downloading: {paper['title']}")
        if download_file(paper["url"], file_path):
            success_count += 1
        print("-" * 50)

    print(f"\n✅ Completed! Have {success_count}/{len(PAPERS)} papers ready.")


if __name__ == "__main__":
    main()