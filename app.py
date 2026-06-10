from flask import Flask, request, render_template
from ultralytics import YOLO
import cv2
import os
import uuid

app = Flask(__name__)

# Cargar el modelo entrenado
model = YOLO('runs/detect/carrito_control_model-2/weights/best.pt')

# Carpeta donde se guardan los resultados
os.makedirs('static/results', exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error='No se subió ninguna imagen.')

    file = request.files['image']

    if file.filename == '':
        return render_template('index.html', error='No se seleccionó ningún archivo.')

    # Guardar imagen subida
    filename = f"{uuid.uuid4()}.jpg"
    input_path = os.path.join('static/results', filename)
    file.save(input_path)

    # Hacer detección
    results = model(input_path, conf=0.25, iou=0.45)

    # Guardar imagen con detecciones dibujadas
    results[0].save(input_path)

    # Obtener lista de detecciones
    detections = []
    for box in results[0].boxes:
        label = model.names[int(box.cls)]
        confidence = round(float(box.conf) * 100, 1)
        detections.append({
            'label': label,
            'confidence': confidence
        })

    return render_template(
        'index.html',
        result_image=filename,
        detections=detections,
        total=len(detections)
    )

if __name__ == '__main__':
    app.run(debug=True)