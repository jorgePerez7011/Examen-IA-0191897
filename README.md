# Examen Final – Detección de Objetos con YOLO (YOLOv8)

Aplicación de visión por computadora para **detección en tiempo real** y **detección por imágenes**, entrenada con un **dataset propio** etiquetado en formato YOLO.

---

## 1. Objetivo del proyecto
Desarrollar un sistema capaz de detectar al menos **dos (2) categorías** de objetos en imágenes y/o video usando **YOLOv8**, aplicando el flujo completo:

1. Construcción del dataset (recolección, etiquetado y división)
2. Entrenamiento del modelo
3. Evaluación con métricas (Precision, Recall, mAP@50, mAP@50-95)
4. Implementación de una aplicación funcional (Flask)

---

## 2. Categorías detectadas
El modelo está entrenado para **2 clases**:

- **Carrito**
- **ControlPlay4**

Configuración de clases (ver `data.yaml`):

- `nc: 2`
- `names: ['Carrito', 'ControlPlay4']`

---

## 3. Construcción del Dataset

### 3.1 Dataset y etiquetado
- Las imágenes fueron anotadas en **formato YOLOv8/YOLO** (archivos `.txt` con coordenadas normalizadas en formato: `class x_center y_center width height`).
- La estructura del dataset se organiza en:
  - `train/`
  - `valid/`
  - `test/`

### 3.2 División del dataset
Se usa la configuración del `data.yaml`:

- `train: ../train/images`
- `val: ../valid/images`
- `test: ../test/images`

### 3.3 Evidencias de entrenamiento / dataset
Se incluyen evidencias del proceso de entrenamiento y métricas en la carpeta **`Evidencias/`**.

---

## 4. Entrenamiento del Modelo

### 4.1 Modelo base
Se utilizó YOLOv8 como modelo base:

- `yolov8n.pt`

### 4.2 Script de entrenamiento
El entrenamiento se realiza con `train.py`:

- Dataset: `data='data.yaml'`
- Épocas: `epochs=30`
- Tamaño de imagen: `imgsz=640`
- Batch: `batch=8`
- Nombre de experimento: `name='carrito_control_model'`
- Early stopping: `patience=10`

### 4.3 Evidencias del entrenamiento
Capturas incluidas en `Evidencias/`, por ejemplo:

- **Comienzo de entrenamiento**: `Evidencias/Comienzo de entrenamiento.png`
- Curvas/Gráficas del entrenamiento:
  - `Evidencias/BoxF1_curve.png`
  - `Evidencias/BoxP_curve.png`
  - `Evidencias/BoxR_curve.png`
  - `Evidencias/BoxPR_curve.png`
- Curvas por épocas (si están disponibles en la carpeta):
  - `Evidencias/epoca 1.png`, `Evidencias/epoca 2.png`, `Evidencias/epoca 6 7 8.png`, `Evidencias/epoca 12.png`, `Evidencias/epoca 30.png`

---

## 5. Evaluación del Modelo

Durante la evaluación se analizan métricas típicas de YOLO:

### 5.1 Precision
**Precision** mide qué proporción de las detecciones realizadas por el modelo son correctas:

- Alta Precision ⇒ pocas falsas alarmas (detecciones incorrectas).

### 5.2 Recall
**Recall** mide qué proporción de los objetos reales fueron detectados por el modelo:

- Alta Recall ⇒ pocos objetos reales se pierden.

### 5.3 mAP@50
**mAP@50** es el promedio del **Average Precision** sobre todas las clases usando:
- umbral de IoU = 0.50

### 5.4 mAP@50-95
**mAP@50-95** es el promedio del AP para múltiples umbrales de IoU:
- IoU en el rango [0.50, 0.95] (paso típico 0.05)

Esto es más estricto que mAP@50 y evalúa mejor la calidad de localización.

### 5.5 Evidencias de métricas
Se incluyen figuras en `Evidencias/`:

- **Curvas de Precision/Recall/F1**:
  - `Evidencias/BoxP_curve.png`
  - `Evidencias/BoxR_curve.png`
  - `Evidencias/BoxF1_curve.png`
  - `Evidencias/BoxPR_curve.png`

- **Matriz de confusión**:
  - `Evidencias/confusion_matrix_normalized.png`
  - `Evidencias/confusion_matrix.png`

> Nota: Las cifras numéricas exactas (valores de Precision/Recall/mAP) dependen del reporte generado por Ultralytics para la corrida específica. Las evidencias visuales incluidas sustentan el análisis requerido.

---

## 6. Implementación de la Aplicación (Flask)

Se implementa una aplicación web con **Flask** que:

1. Permite cargar una imagen.
2. Ejecuta inferencia con el modelo entrenado.
3. Dibuja/guarda la salida y muestra las detecciones (clase + confianza).

### 6.1 Script principal
- `app.py`

Características:
- Carga del modelo entrenado:
  - `YOLO('runs/detect/carrito_control_model-2/weights/best.pt')`
- Endpoint de predicción:
  - `POST /predict`
- Render de la salida:
  - `templates/index.html`
- Almacenamiento de resultados:
  - `static/results/`

### 6.2 Frontend
- `templates/index.html`

---

## 7. Instrucciones de ejecución

### 7.1 Requisitos
Ver `requirements.txt`.

Archivo `requirements.txt`:
- `ultralytics`
- `opencv-python`
- `flask`

### 7.2 Instalación (entorno Python)
```bash
pip install -r requirements.txt
```

### 7.3 Ejecutar la aplicación Flask
```bash
python app.py
```

Abrir el navegador en:
- `http://127.0.0.1:5000/`

### 7.4 Demostración de categorías
Durante la sustentación:
1. Ejecutar `python app.py`
2. Cargar una imagen de prueba donde aparezcan **Carrito** y/o **ControlPlay4**.
3. Ver las detecciones mostradas en pantalla (lista de clases y porcentaje de confianza).

---

## 8. Estructura del proyecto
- `train.py` → script de entrenamiento
- `app.py` → aplicación Flask de inferencia por imagen
- `camara.py` → detección en tiempo real con OpenCV (opcional para demostración)
- `data.yaml` → configuración del dataset y clases
- `Evidencias/` → capturas de curvas y matrices de evaluación
- `runs/` → carpeta de salidas de entrenamiento e inferencia (incluye `best.pt`)
- `templates/` → interfaz web
- `static/results/` → imágenes con detecciones resultantes

---

## 9. Sustentación técnica (qué decir en la demostración en vivo)

1. **Ejecución del sistema**
   - Iniciar la app con `python app.py`.

2. **Detección de las categorías entrenadas**
   - Mostrar que la app detecta **Carrito** y **ControlPlay4**.

3. **Explicar el dataset utilizado**
   - Se etiquetó en formato YOLO.
   - Se dividió en `train/`, `valid/`, `test/` usando `data.yaml`.

4. **Explicar métricas obtenidas**
   - Definir y diferenciar Precision, Recall, mAP@50 y mAP@50-95.
   - Apoyar el análisis con capturas de curvas y matriz de confusión:
     - `BoxP_curve`, `BoxR_curve`, `BoxF1_curve`, `BoxPR_curve`
     - `confusion_matrix(_normalized)`

5. **Explicación general de la arquitectura**
   - Se usa YOLOv8 (backbone + neck + head para detección).
   - El modelo aprende a predecir bounding boxes y clases.
   - En inferencia, Flask carga `best.pt` y ejecuta `model(image, conf=..., iou=...)`.

6. **Robustez del sistema**
   - Mostrar resultados en varias imágenes del conjunto `test/` o imágenes similares.

---

## 10. Modelo entrenado
El modelo final se encuentra en:

- `runs/detect/carrito_control_model-2/weights/best.pt`

---

## 11. Archivos importantes (para el jurado)
- `README.md` (este documento)
- `data.yaml`
- `train.py`
- `app.py`
- `camara.py` (opcional para tiempo real)
- `requirements.txt`
- Evidencias:
  - `Evidencias/*`


