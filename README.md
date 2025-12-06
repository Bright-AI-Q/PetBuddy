# PetBuddy: One-Click Pet Health & Feeding Consultant

[![English Web Demo](https://img.shields.io/badge/🌐-English%20Web%20Demo-blue)](https://your-demo-link.com)
[![YOLOv8](https://img.shields.io/badge/🖼️-YOLOv8-orange)](https://github.com/ultralytics/ultralytics)
[![Mini-PetNet](https://img.shields.io/badge/🧠-Mini--PetNet-green)](https://github.com/your-username/PetBuddy)
[![1.1MB LLM](https://img.shields.io/badge/💬-1.1MB%20LLM-purple)](https://huggingface.co/Qwen/Qwen-1.8B)

> Upload any pet photo → detect all cats & dogs → classify breed → generate feeding/grooming text → continue chatting.

## 🎯 Goal

- **Ultra-lightweight**: Whole pipeline ≤ 1.1 MB (1.05M vision + 1.1M LLM)
- **Real-time performance**: End-to-end < 3 seconds on RTX-3060
- **Comprehensive analysis**: Breed classification + health consultation

## 🚀 Pipeline

### 🖼️ Image Processing
```
YOLOv8 → Mini-PetNet (3 ablatable modules) → JSON output
```

### 💬 Text Generation
```
Qwen-1.8B + QLoRA 4-bit → 5k Q-A jsonl (team-verified) → Gradio chat interface
```

## 🔬 Innovations (Each Module Ablatable)

| Innovation | Description | Performance Gain |
|------------|-------------|------------------|
| **LDRE** | GridMask top-score ear/eye regions from YOLO heat-map | +1.8% Top-1 |
| **Dual-Attention** | ECA + 2-D relative position encoding | +1.3% |
| **Progressive Self-KD** | EMA teacher on 3 stages, zero inference cost | +2.1% |

## 📊 Performance Metrics

- **Accuracy**: 88.7% Top-1
- **Model Size**: 1.05M parameters
- **Computational Cost**: 111M FLOPs
- **Inference Speed**: 7ms

## 🛠️ Installation

```bash
git clone https://github.com/your-username/PetBuddy.git
cd PetBuddy
pip install -r requirements.txt
```

## 🎮 Quick Start

```bash
# Use multi_pet_inference.py for pet detection and classification
python tools/multi_pet_inference.py \
    --image "your_pet_photo.jpg" \
    --classifier "path/to/your/petnet/weights.pt" \
    --detector "yolov8n.pt" \
    --det_conf 0.2 \
    --cls_conf 0.1 \
    --save
```

Parameter Description:
- `--image`: Input image path
- `--classifier`: PetNet classifier weights file path (.pt)
- `--detector`: YOLO detector weights file path (default: yolov8n.pt)
- `--det_conf`: YOLO detection confidence threshold (default: 0.2)
- `--cls_conf`: PetNet classification confidence threshold (default: 0.1)
- `--save`: Save visualization result image
- `--output_dir`: Output directory (default: current directory)
- `--img_size`: Inference image size (default: 224)

## 📁 Project Structure

```
PetBuddy/
├── configs/            # 4 YAML: full / no-LDRE / no-Attn / no-KD
├── models/
│   ├── petnet.py
│   └── modules/        # LDRE · Dual-Attention · Self-KD plug-ins
├── tools/
│   ├── train.py        # trains & logs csv
│   ├── test.py
│   └── latex_table.py  # outputs ablation_table.tex
├── llm/
│   ├── pet_knowledge/  # 5k Q-A jsonl (AI-gen → team-verified)
│   └── qlora/          # 4-bit QLoRA weights & script
├── runs/               # weights + ablation_results.csv
├── report/             # ready for LaTeX: figures + tables
├── app/                # Frontend and backend application
├── data/               # Dataset and training data
├── utils/              # Utility functions
└── requirements.txt    # Python dependencies
```

## 📚 Citation

If you use PetBuddy in your research, please cite:

```bibtex
@software{PetBuddy2025,
  title = {PetBuddy: One-Click Pet Health & Feeding Consultant},
  author = {Your Name and Team},
  year = {2025},
  url = {https://github.com/your-username/PetBuddy}
}
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [Qwen](https://huggingface.co/Qwen) for the base language model
- [Gradio](https://www.gradio.app/) for the web interface

---

⭐ **Star this repo if you find it useful!**