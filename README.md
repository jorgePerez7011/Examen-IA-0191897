# Examen Final – Detección de Objetos con YOLO (Jorge Perez-0191897)

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

Configuración de clases  `data.yaml`

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

En esta sección se describe cómo se evaluó el desempeño del modelo entrenado para **2 clases**: **Carrito** y **ControlPlay4**, usando el reporte de métricas generado por **Ultralytics (YOLOv8)**.
 umbral de IoU (Intersección sobre Unión) son valores límite (entre 0 y 1) utilizados para definir si la predicción del modelo es correcta o no.

### 5.1 Precision (Carrito / ControlPlay4)
**Precision** mide qué proporción de las detecciones realizadas por el modelo son correctas para cada clase:

- Alta Precision ⇒ pocas falsas alarmas (detecciones de **Carrito** o **ControlPlay4** que en realidad no corresponden).

### 5.2 Recall (Carrito / ControlPlay4)
**Recall** mide qué proporción de los objetos reales fueron detectados por el modelo para cada clase:

- Alta Recall ⇒ pocos objetos reales se pierden (p. ej., que instancias reales de **Carrito** no sean detectadas).

### 5.3 mAP@50
**mAP@50** resume el desempeño de detección promediando el **Average Precision/Precisión promedio (AP)** de las clases con:
- umbral de IoU = **0.50**

### 5.4 mAP@50-95
**mAP@50-95** es más estricto y promedia el AP para múltiples umbrales de IoU:
- IoU en el rango **[0.50, 0.95]** (paso típico 0.05)

Esto evalúa mejor la **calidad de localización (bounding box)**: no solo que “detecte”, sino que las cajas se ajusten correctamente a **Carrito** y **ControlPlay4**.

### 5.5 Evidencias de métricas (del entrenamiento realizado)
Se incluyen evidencias visuales en `Evidencias/` para sustentar el análisis:

- **Curvas de Precision/Recall/F1** (evolución por épocas):
  - `Evidencias/BoxP_curve.png`
  - `Evidencias/BoxR_curve.png`
  - `Evidencias/BoxF1_curve.png`
  - `Evidencias/BoxPR_curve.png`

- **Matriz de confusión** (cómo se confunden Carrito vs ControlPlay4):
  - `Evidencias/confusion_matrix_normalized.png`
  - `Evidencias/confusion_matrix.png`

> Nota: Los valores numéricos exactos (Precision/Recall/mAP) se toman del reporte de Ultralytics correspondiente a la corrida. Las gráficas y la matriz incluidas en `Evidencias/` respaldan la interpretación para el modelo entrenado en este proyecto.


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

## 9. Modelo entrenado
El modelo final se encuentra en:

- `runs/detect/carrito_control_model-2/weights/best.pt`

---

## Conclusiones

En base a los resultados podemos decir que una buena toma a las fotos y los distintos angulos y luces de los mismos nos pueden dar mejores resultados, ademas tambien a la hora de empezar a entrenar el modelo las epocas deben ser las correctas ya que en medio de la realizacion del examen en una fase puse 50 epocas y el resultado fue el mismo, se llegaba a un sobrenetrenamiento, luego de ver los labels creados por el entrfenamiento de roboflow se logra evidenciar que el numero de instancias de el carrito es mayor que el del control, es decir para subir mas la precision del modelo se deben agregar mas fotos al dataset. 

El desarrollo de este sistema de detección de objetos con YOLOv8 demostró que es posible construir un modelo funcional y preciso a partir de un dataset propio, siguiendo el flujo completo de recolección, etiquetado, entrenamiento y evaluación. Las métricas obtenidas (Precision, Recall, mAP@50 y mAP@50-95) evidencian que el modelo logra identificar correctamente las clases Carrito y ControlPlay4, con una buena capacidad de localización de los objetos en la imagen.

La integración del modelo entrenado en una aplicación web con Flask valida la viabilidad de llevar soluciones de visión por computadora a entornos reales de uso, permitiendo realizar inferencias de forma sencilla a través de una interfaz accesible. Esto confirma que YOLOv8, incluso en su variante ligera (yolov8n), representa una opción eficiente y práctica para proyectos de detección en tiempo real con recursos computacionales limitados.

