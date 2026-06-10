from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='data.yaml',
    epochs=30,
    imgsz=640,
    batch=8,
    name='carrito_control_model',
    patience=10
)