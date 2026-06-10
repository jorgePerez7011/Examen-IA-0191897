import cv2
from ultralytics import YOLO

# Debe decir esto
model = YOLO('runs/detect/carrito_control_model-2/weights/best.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.10)
    annotated = results[0].plot()

    cv2.imshow('Deteccion en Tiempo Real', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()