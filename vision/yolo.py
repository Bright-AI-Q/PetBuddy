from ultralytics import YOLO

# Reference: from https://github.com/ultralytics/ultralytics README

model = YOLO("yolov8n-cls.pt")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="../data/yolo_stanford_dogs",  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=256,  # Image size for training
    device="mps",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)
