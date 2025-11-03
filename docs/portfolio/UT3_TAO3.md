---
title: "Comparativa YOLO y Tracking"
date: 2025-10-31
---

# Comparativa YOLO y Tracking

## Contexto

En esta actividad se realizó una comparación práctica entre modelos de detección basados en Ultralytics YOLO (versiones v8 y v11) y una demo de tracking usando Norfair. El objetivo fue experimentar con entrenamiento rápido sobre un dataset de detección de frutas, analizar trade-offs entre velocidad y precisión, y producir visualizaciones y métricas comparativas.

## Objetivos

- Evaluar y comparar el desempeño de varias configuraciones de YOLO sobre el dataset fruit-detection de Kaggle.
- Mostrar resultados visuales (detecciones con bounding boxes) y preparar métricas básicas de tracking/benchmark para futuros experimentos.

## Actividades

- Preparación del entorno e instalación de dependencias
- Descarga y verificación del dataset
- Entrenamiento rápido de modelos base (YOLOv8n, YOLOv8s, YOLOv11n)
- Entrenamiento extendido (más épocas y modelos adicionales)
- Evaluación, benchmarking de inferencia (20 imágenes) y recolección de métricas
- Visualización comparativa de detecciones (3 imágenes) y gráficos de análisis

## Desarrollo

En las primeras celdas del notebook se instalaron y cargaron las dependencias necesarias, como `ultralytics`, `opencv-python`, `matplotlib`, `numpy`, `pandas`, `seaborn`, `gdown`, `norfair` y `motmetrics`, dejando además una nota opcional para instalar `detectron2` en entornos con CUDA compatible. A continuación se verificó la disponibilidad de GPU con PyTorch y se registró el dispositivo, y se fijó la semilla global (`SEED = 42`) para que los experimentos sean reproducibles.

El dataset utilizado fue `lakshaytyagi01/fruit-detection` de Kaggle, y el notebook automatiza su descarga mediante la CLI de Kaggle cuando `kaggle.json` está disponible. Tras descomprimir el paquete se busca un `data.yaml` existente; si no se encuentra, se genera uno con las rutas de `train` y `val`, el número de clases y nombres de etiquetas predeterminados (por ejemplo `apple`, `banana`, `grapes`, `orange`, `pineapple`, `watermelon`). También se incluye lógica para corregir rutas absolutas convirtiéndolas en relativas y crear un `data_fixed.yaml` cuando hace falta, lo que evita problemas al alimentar Ultralytics.

Antes de los entrenamientos se ejecutaron funciones rápidas para contar imágenes por split, de modo de conocer cuántas muestras había en `train` y `val` y ajustar parámetros como `fraction` y `batch` para los experimentos rápidos. En la fase inicial de pruebas (Trabajo 1) se entrenaron modelos ligeros con vistas a obtener resultados indicativos en poco tiempo: se probaron `YOLOv8n`, `YOLOv8s` y `YOLOv11n` usando 5 épocas, tamaño de imagen 416, batch 16 y `fraction=0.25` (es decir, 25% del dataset). Para cada modelo se midió el tiempo de entrenamiento y se ejecutó la validación con `model.val()` para recoger `map50`, `map5095`, `precision` y `recall`, almacenando los resultados en un `DataFrame` para comparaciones rápidas.

Posteriormente se llevó a cabo una comparación extendida con parámetros más exigentes para observar diferencias más reales entre arquitecturas: en esta segunda fase se evaluaron `YOLOv8n`, `YOLOv8s`, `YOLOv8m` y `YOLOv11n` con 10 épocas, tamaño de imagen 640 y `fraction=0.75`. Además de las métricas agregadas, se recolectaron métricas por clase (`maps`), el tamaño del archivo de pesos `best.pt`, el tiempo de inferencia promedio por imagen sobre una muestra de hasta 20 imágenes y el uso máximo de memoria GPU cuando estaba disponible. Todos estos resultados se consolidaron en `df_ext` para facilitar la comparación de `map50`, `map5095`, `precision`, `recall`, `train_time_min`, `infer_ms`, `weights_mb` y `gpu_mem_max_mb`.

Con los modelos entrenados se procedió a una inspección cualitativa: se cargaron los `best.pt` cuando estaban presentes y se aplicaron a tres imágenes de validación escogidas aleatoriamente, generando visualizaciones con los bounding boxes y anotando el número de detecciones por modelo para detectar falsos positivos, falsos negativos y problemas de solapamiento o de IoU. Complementariamente, se generaron gráficos de "Speed vs Accuracy" (inferencia ms vs mAP@0.5) y de mAP por clase para el mejor modelo, lo que permitió identificar clases con rendimiento especialmente bajo y priorizar intervenciones en el dataset.

Aunque el foco fue la comparación de detectores, el notebook también incluye una breve demo de tracking basada en Norfair y sugiere el uso de `motmetrics` si se dispone de ground-truths temporales, con la intención de medir MOTA/MOTP/IDF1 en futuros pasos de evaluación. Por último, se documentaron consideraciones de reproducibilidad y despliegue, indicando que el flujo se puede ejecutar tanto en entornos con GPU como en CPU (ajustando `fraction`, batch y número de épocas) y dejando recursos opcionales (como Detectron2) para ejecuciones en máquinas compatibles.

## Reflexión

- Durante la actividad aprendí a automatizar la preparación de un dataset en formato YOLO, incluida la creación/validación de `data.yaml` y la corrección de rutas, lo que facilita repetir experimentos con Ultralytics sin intervención manual.

- El uso de `fraction` para entrenamientos rápidos fue muy útil para experimentar con múltiples arquitecturas en tiempos razonables; sin embargo, los resultados a 25% del dataset son indicativos y deben confirmarse con más datos (por ejemplo, `fraction=0.75`) antes de tomar decisiones de producción.

- Medir tanto el tiempo de entrenamiento como el tiempo de inferencia (ms por imagen) permitió visualizar claramente el trade-off entre velocidad y precisión. Estas métricas son esenciales cuando el despliegue real requiere inferencia en tiempo cercano a real.

- La comparación por clases (mAP por clase) mostró que algunas clases tienen mAP mucho más baja que otras; esto indica necesidad de más muestras o limpieza de anotaciones para esas categorías específicas.

- Integrar visualizaciones (detecciones en imágenes de validación) facilitó detectar errores cualitativos que las métricas agregadas no captan, por ejemplo detecciones fragmentadas o falsos positivos en fondos complejos.

- Próximos pasos recomendados:
  1. Ejecutar los entrenamientos finales sin `fraction` (o con `fraction=1.0`) y hacer fine-tuning del mejor backbone seleccionado.
  2. Recolectar más datos o balancear clases con baja mAP y evaluar si el re-etiquetado mejora significativamente la mAP por clase.
  3. Integrar un pipeline reproducible de evaluación de tracking con ground-truths y `motmetrics` para obtener MOTA/MOTP/IDF1.
  4. Evaluar modelos ligeros (p. ej. versiones `n` o `s` optimizadas) para despliegue en edge si la inferencia en tiempo real es un requisito.

---

Archivo generado a partir del notebook `UT3_TAO3.ipynb`. Si querés, puedo:
- Añadir ejemplos concretos de salidas (tablas `df`/`df_ext`) si ejecutás las celdas y me compartís los resultados.
- Generar figuras PNG/SVG con los gráficos ya renderizados si ejecutás el notebook en tu entorno y subís las imágenes.



## Referencias

- https://colab.research.google.com/drive/1Zbq7zS8CdEXs6lMuqNtkKocAdWMumVh_?usp=sharing