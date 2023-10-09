import torch
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # Load the pretrained model

# Specify your training data configuration in a YAML file
data_yaml = r"C:\Users\saad.ayaz\Desktop\My TASK\NEW TASK\Coco\saadnew.yaml"



# Train the model
results = model.train(
    data=data_yaml,  # Path to your data configuration YAML file
    epochs=2,       # Number of training epochs
    imgsz=640        # Image size used for training
)