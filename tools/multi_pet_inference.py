#!/usr/bin/env python3
# multi_pet_inference.py
import torch
import cv2
import numpy as np
import sys
from pathlib import Path
from torchvision import transforms
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yaml


project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.petnet import PetNet

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

class MultiPetDetector:
    def __init__(self, detector_path, classifier_path, conf_threshold=0.5):
        """
        初始化多宠物检测分类器

        Args:
            detector_path: YOLO检测器模型路径
            classifier_path: PetNet分类器模型路径
            conf_threshold: 置信度阈值
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold

        # 加载YOLO检测器，优先使用本地预训练模型
        detector_path = self._get_detector_path(detector_path)
        self.detector = YOLO(detector_path)
        self.detector.to(self.device)

        # 更鲁棒的模型加载 - 使用与训练时相同的配置结构
        checkpoint = torch.load(classifier_path, map_location=self.device)
        model_config = checkpoint['config']

        # 获取完整的训练配置（确保与训练时一致）
        config_name = Path(classifier_path).parent.name
        config_path = Path(__file__).parent.parent / "configs" / f"{config_name}.yaml"

        if not config_path.exists():
            print(f"⚠️ Warning: Config file {config_path} not found, using default base config")
            config_path = Path(__file__).parent.parent / "configs" / "petnet_base.yaml"

        full_config = load_config(config_path)

        # 初始化模型（使用完整的配置确保结构一致性）
        self.classifier = PetNet(
            num_classes=model_config['num_classes'],
            stage_repeats=model_config['stage_repeats'],
            ldre_cfg=full_config['data'].get('ldre_cfg'),  # 从完整配置的data部分获取
            attn_cfg=model_config['attn_cfg'],
            selfkd_cfg=model_config['selfkd_cfg'],
            max_pets_per_image=model_config.get('max_pets_per_image', 10)
        )

        # 处理state_dict不匹配问题
        model_state_dict = self.classifier.state_dict()
        pretrained_dict = {
            k: v for k, v in checkpoint['model_state_dict'].items()
            if k in model_state_dict and v.size() == model_state_dict[k].size()
        }

        # 加载匹配的参数
        model_state_dict.update(pretrained_dict)
        self.classifier.load_state_dict(model_state_dict)

        # 打印加载状态
        print(f"成功加载 {len(pretrained_dict)}/{len(model_state_dict)} 个参数")
        if len(pretrained_dict) != len(model_state_dict):
            print(f"警告: {len(model_state_dict)-len(pretrained_dict)} 个参数使用随机初始化")

        self.classifier.to(self.device)
        self.classifier.eval()

        # 固定预处理参数
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # 从模型配置加载类别映射
        self.class_names = self._load_class_names(model_config['num_classes'])

    def _get_detector_path(self, detector_path):
        """获取检测器模型路径，优先使用预训练模型"""
        # 强制使用预训练的YOLOv8n模型
        if Path('./yolov8n.pt').exists():
            print("✅ 使用预训练的YOLOv8n模型")
            return './yolov8n.pt'
        elif Path('./yolov8n-pose.pt').exists():
            print("✅ 使用预训练的YOLOv8n-pose模型")
            return './yolov8n-pose.pt'
        else:
            print("⚠️  使用默认的YOLOv8n模型 (将自动下载)")
            return 'yolov8n.pt'
    def _load_class_names(self, num_classes):
        """加载类别名称映射"""
        try:
            # 从模型配置中获取数据集名称
            from utils.data_utils import get_class_names

            return get_class_names("pet_cls_training")
        except Exception as e:
            print(f"警告: 加载类别映射失败: {e}")
            # 失败时使用序号作为类别名
            return {i: f"class_{i}" for i in range(num_classes)}

    def detect_pets(self, image_path):
        """
        检测图像中的宠物并进行分类

        Args:
            image_path: 输入图像路径

        Returns:
            list: 检测结果列表，每个元素包含bbox、类别、置信度
        """
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"错误: 无法读取图像 {image_path}")
            return []

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = []

        # 第一阶段：YOLO检测
        results = self.detector(img_rgb, conf=0.1, iou=0.2, agnostic_nms=True)

        if results and len(results) > 0:
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # 提取检测框信息
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls_id = int(box.cls[0].cpu().numpy())

                        # 第一阶段：YOLO检测结果为主
                        class_name_map = {0: 'cat', 1: 'dog'}
                        detected_class = class_name_map.get(cls_id, f'unknown({cls_id})')
                        print(f"YOLO检测到对象: class_id={cls_id} ({detected_class}), 置信度: {conf:.3f}")

                        # 处理所有置信度达标的检测结果
                        # YOLOv8 COCO数据集：15=cat, 16=dog
                        if conf > self.conf_threshold and cls_id in [0, 1, 15, 16]:
                            # 第二阶段：PetNet精细分类
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            crop_img = img_rgb[y1:y2, x1:x2]

                            if crop_img.size > 0:  # 确保裁剪区域有效
                                print(f"处理检测框: ({x1}, {y1}, {x2}, {y2}), 大小: {crop_img.shape}")

                                # PetNet精细分类
                                input_tensor = self.transform(crop_img).unsqueeze(0).to(self.device)
                                with torch.no_grad():
                                    output = self.classifier(input_tensor)
                                    logits = output[0] if isinstance(output, tuple) else output
                                    probabilities = torch.softmax(logits, dim=1)
                                    class_conf, class_id = probabilities.max(1)

                                # 输出Top-5预测结果
                                k = min(5, probabilities.size(1))
                                top_conf, top_idx = torch.topk(probabilities, k)
                                print(f"\nPetNet精细分类 (Top-{k}):")
                                for i, (idx, c) in enumerate(zip(top_idx[0], top_conf[0])):
                                    print(f"{i+1}. {self.class_names.get(idx.item(), f'class_{idx}')}: {c.item():.4f}")

                                detections.append({
                                    'bbox': [x1, y1, x2, y2],
                                    'yolo_class_id': cls_id,
                                    'yolo_class_name': 'cat' if cls_id in [0, 15] else 'dog',
                                    'detection_confidence': float(conf),  # 统一使用detection_confidence
                                    'class_id': class_id.item(),
                                    'class_confidence': class_conf.item(),
                                    'class_name': self.class_names.get(class_id.item(), f"class_{class_id}")
                                })
                            else:
                                print("警告: 裁剪区域无效")
                        else:
                            print(f"跳过低置信度检测: {conf:.3f} < {self.conf_threshold:.3f}")
                else:
                    print("警告: 未找到检测框")
        else:
            print("未检测到任何对象")

        return detections
    # 移除此方法，功能已整合到detect_pets中

    def visualize_results(self, image_path, detections, output_path=None):
        """可视化检测结果"""
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        ax = plt.gca()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            width = x2 - x1
            height = y2 - y1

            # 绘制边界框
            rect = Rectangle((x1, y1), width, height, linewidth=2,
                           edgecolor='red', facecolor='none')
            ax.add_patch(rect)

            # 添加标签
            label = f"{det['class_name']} ({det['class_confidence']:.2f})"
            plt.text(x1, y1 - 10, label, color='red', fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.7))

        plt.axis('off')

        if output_path:
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            print(f"结果保存至: {output_path}")

        plt.show()

def main():
    import argparse

    parser = argparse.ArgumentParser(description='多宠物检测与分类')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--detector', type=str, default='./yolov8n.pt', help='YOLO检测器路径 (默认使用预训练模型)')
    parser.add_argument('--classifier', type=str, default='runs/petnet_cls/best.pt',
                       help='PetNet分类器路径')
    parser.add_argument('--output', type=str, help='输出图像路径')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')

    args = parser.parse_args()

    # 初始化检测器
    detector = MultiPetDetector(args.detector, args.classifier, args.conf)

    # 检测宠物
    print(f"正在检测图像: {args.image}")
    detections = detector.detect_pets(args.image)
    # 输出结果
    print(f"\n检测到 {len(detections)} 只宠物:")
    for i, det in enumerate(detections, 1):
        print(f"{i}. {det['class_name']} - 精细分类置信度: {det['class_confidence']:.3f}")
        print(f"   检测框: {det['bbox']}")
        print(f"   YOLO检测: {det['yolo_class_name']}, 置信度: {det.get('detection_confidence', det.get('yolo_confidence', 0)):.3f}")
        print()

    # 可视化结果
    if detections:
        detector.visualize_results(args.image, detections, args.output)
    else:
        print("未检测到宠物")

if __name__ == "__main__":
    main()