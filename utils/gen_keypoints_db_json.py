import json
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np


def generate_keypoints_json(
        model_path='runs/pose/train3/weights/best.pt',
        data_root='data/pet_cls_training',  # 你的分类数据集根目录
        output_path='data/keypoints_db.json'
):
    print(f"🚀 Loading model from: {model_path}")
    model = YOLO(model_path)

    # 查找所有图片 (支持 jpg, jpeg, png)
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    root_path = Path(data_root)

    print(f"📂 Scanning images in: {root_path}")
    for ext in extensions:
        image_files.extend(list(root_path.rglob(ext)))

    print(f"✨ Found {len(image_files)} images. Starting inference...")

    database = {}

    # 批量推理 (或者单张推理，这里用单张以确保逻辑清晰且不爆显存)
    for img_path in tqdm(image_files):
        # 运行预测
        # conf=0.25 过滤低置信度，save=False 不保存图片
        results = model.predict(str(img_path), conf=0.1, verbose=False)
        result = results[0]  # 获取第一张图的结果

        entry_list = []

        # 检查是否检测到了目标
        if result.boxes:
            # 遍历每一个检测到的对象 (例如一张图里有多只狗)
            for i in range(len(result.boxes)):
                box = result.boxes[i]

                # 1. 提取 BBox [x1, y1, x2, y2]
                # .cpu().numpy() 转为 numpy 数组, .tolist() 转为 Python 列表以便存 JSON
                bbox = box.xyxy.cpu().numpy()[0].tolist()
                conf = float(box.conf.cpu().numpy()[0])
                cls_id = int(box.cls.cpu().numpy()[0])

                # 2. 提取关键点
                # result.keypoints.data 的形状通常是 (N, 5, 3) -> (num_dets, num_kpts, [x,y,conf])
                if result.keypoints is not None:
                    # 获取当前对象的关键点
                    kpts_data = result.keypoints.data[i].cpu().numpy()  # shape: (5, 3)

                    keypoints_list = []
                    for idx, (x, y, score) in enumerate(kpts_data):
                        keypoints_list.append({
                            "x": float(x),
                            "y": float(y),
                            "score": float(score),
                            "type": f"keypoint_{idx}"  # 对应 JSON 中的格式
                        })
                else:
                    keypoints_list = []

                # 构建该对象的记录
                entry_list.append({
                    "bbox": bbox,
                    "confidence": conf,
                    "class_id": cls_id,
                    "keypoints": keypoints_list
                })

        # 计算相对路径作为 Key (例如 "test/pets_0120.../xxx.jpg")
        rel_path = str(img_path.relative_to(root_path))

        # 存入字典
        database[rel_path] = entry_list

    # 保存为 JSON
    print(f"💾 Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(database, f, indent=2)

    print("✅ Done!")


if __name__ == "__main__":
    # 请根据你的实际路径修改这里
    generate_keypoints_json(
        model_path='runs/pose/train3/weights/best.pt',  #
        data_root='data/pet_cls_training',  #
    )