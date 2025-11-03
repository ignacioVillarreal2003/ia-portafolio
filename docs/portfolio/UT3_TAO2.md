---
title: "Food-101"
date: 2025-10-31
---

# Food-101

## Contexto

En esta actividad se trabajó con el dataset Food-101, que contiene imágenes de 101 diferentes clases de alimentos. El objetivo principal fue implementar técnicas avanzadas de deep learning para clasificación de imágenes, incluyendo data augmentation, transfer learning y métodos de explicabilidad de modelos.

## Objetivos

- Implementar un modelo de clasificación de imágenes utilizando transfer learning con EfficientNetB0 alcanzando una precisión de validación superior al 70%
- Aplicar técnicas de data augmentation para mejorar la robustez del modelo
- Implementar y visualizar métodos de explicabilidad (GradCAM, Integrated Gradients y LIME) para entender las decisiones del modelo

## Actividades

- Configuración inicial y carga del dataset Food-101
- Implementación de pipelines de data augmentation
- Creación y entrenamiento del modelo con transfer learning
- Implementación de técnicas de explicabilidad
- Análisis y visualización de resultados

## Desarrollo

### Configuración Inicial y Preparación del Dataset

Para el manejo de imágenes, establecimos un tamaño estándar de 224x224 píxeles y un tamaño de batch de 32. El dataset Food-101 se cargó utilizando TensorFlow Datasets, que contiene un total de 101 clases diferentes de alimentos. Para optimizar el tiempo de desarrollo, trabajamos con un subset del dataset: 20,000 imágenes para entrenamiento y 5,000 para validación.

### Implementación de Data Augmentation

Se desarrollo dos pipelines diferentes para el procesamiento de datos:

El pipeline baseline se encargó de:

- Redimensionar las imágenes a 224x224
- Aplicar normalización específica para EfficientNet
- Mantener las imágenes en el rango [0, 255] antes de la normalización

El pipeline aumentado incluyó transformaciones más sofisticadas:

- Volteos horizontales y verticales aleatorios para aumentar la variabilidad
- Rotaciones de hasta 45 grados
- Zoom aleatorio de hasta 20%
- Traslaciones de hasta 10% en ambas direcciones
- Ajustes de contraste y brillo
- Normalización final para EfficientNet

Estas transformaciones se visualizaron para verificar su correcta implementación y asegurar que mantenían la integridad de las imágenes.

### Arquitectura del Modelo y Proceso de Entrenamiento

Se implemento un modelo basado en transfer learning utilizando EfficientNetB0. La arquitectura final consistió en:

- EfficientNetB0 pre-entrenado en ImageNet (con pesos congelados)
- Una capa de Global Average Pooling para reducir la dimensionalidad
- Una capa de Dropout con tasa de 0.2 para prevenir el overfitting
- Una capa densa final con 101 neuronas (una por clase) y activación softmax

El modelo se compiló con:

- Optimizador: Adam
- Función de pérdida: Sparse Categorical Crossentropy
- Métrica: Accuracy

El entrenamiento se realizó durante 6 épocas utilizando el pipeline aumentado para el conjunto de entrenamiento y el pipeline baseline para validación. Los resultados mostraron una mejora progresiva en la precisión tanto en entrenamiento como en validación.

### 4. Técnicas de Explicabilidad

Se implemento tres técnicas diferentes de explicabilidad para entender mejor las decisiones del modelo:

1. GradCAM:

Se identifico automáticamente la última capa convolucional del modelo y generamos mapas de calor que resaltan las regiones más importantes para la predicción. Esta técnica nos permitió visualizar qué partes de la imagen el modelo consideraba más relevantes para su decisión.

2. Integrated Gradients (IG):

Se implemento IG con 50 pasos de interpolación entre una imagen base (negra) y la imagen de entrada. Esta técnica nos proporcionó una vista más granular de la contribución de cada píxel a la predicción final.

3. LIME (Local Interpretable Model-agnostic Explanations):

Se configuro LIME para generar explicaciones locales de las predicciones, utilizando 1000 muestras por imagen y visualizando las 10 regiones más importantes que influyen en la clasificación.

### 5. Mejoras en la Robustez

Se implemento Test-Time Augmentation (TTA) para mejorar la robustez del modelo durante la inferencia. Este proceso consistió en:

- Aplicar 8 transformaciones diferentes a cada imagen de prueba
- Generar predicciones para cada versión transformada
- Promediar las predicciones para obtener un resultado más robusto

## Reflexión

Durante el proceso de implementación, observé cómo el data augmentation contribuyó significativamente a la capacidad del modelo para generalizar mejor, especialmente en un dataset tan diverso como Food-101.

Los resultados del entrenamiento mostraron una evolución positiva en la precisión del modelo. GradCAM mostró que el modelo efectivamente se enfocaba en las características distintivas de cada plato, mientras que Integrated Gradients proporcionó una visión más detallada de cómo cada región de la imagen contribuía a la clasificación final.

La implementación de Test-Time Augmentation demostró ser efectiva para mejorar la robustez de las predicciones, aunque añadió un costo computacional adicional.

Las técnicas de explicabilidad implementadas no solo ayudaron a entender mejor las decisiones del modelo, sino que también proporcionaron una base para la confianza en sus predicciones.

## Referencias

- https://colab.research.google.com/drive/1DYzDoEOcwinQ_m1kBIq3n23bkOs-kob_?usp=sharing