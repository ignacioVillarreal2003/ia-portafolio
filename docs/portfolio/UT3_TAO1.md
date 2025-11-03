---
title: "Cat vs Dogs"
date: 2025-10-31
---

# Cat vs Dogs

## Contexto

Este trabajo aborda la clasificación binaria de imágenes (gatos vs. perros) con TensorFlow/Keras, poniendo el foco en comparar una red convolucional construida desde cero con un enfoque de Transfer Learning basado en ResNet50. 

El flujo completo incluyó la preparación del conjunto de datos con `tensorflow_datasets`, la normalización de imágenes al rango [0, 1] y el reescalado a 224×224 píxeles, la definición de arquitecturas, la configuración de entrenamiento con callbacks para regularización dinámica y, finalmente, la evaluación cuantitativa y cualitativa.

## Objetivos

El objetivo principal fue alcanzar un desempeño de referencia en test igual o superior al 70% de exactitud y, en paralelo, comparar una CNN desde cero frente a un modelo con aprendizaje por transferencia. 

Además, se buscó analizar métricas por clase para entender asimetrías en el rendimiento entre “Cat” y “Dog”, y explorar rápidamente arquitecturas livianas (p. ej., MobileNetV2 y EfficientNetB0) que pudieran ofrecer mejor relación precisión/parámetros.

## Actividades

- Preparación y preprocesamiento del dataset
- Implementación CNN simple y compilación
- Implementación Transfer Learning (ResNet50)
- Entrenamiento con callbacks (ambos modelos)
- Evaluación, gráficas y reportes
- Experimentos con arquitecturas livianas

## Desarrollo

La preparación del dataset se realizó dividiendo aproximadamente 80/20 para entrenamiento y test, y mapeando un preprocesamiento que normaliza y redimensiona las imágenes. Se limitaron 4,000 ejemplos para entrenamiento y 1,000 para test con el fin de acelerar el entrenamiento, manteniendo un balance razonable entre clases (aprox. 2,048 “Cat” y 1,952 “Dog”). Se utilizó codificación one-hot y un tamaño de lote de 32 imágenes por iteración.

La primera arquitectura fue una CNN desde cero con cuatro bloques convolucionales y de pooling, normalización por lotes y activaciones ReLU. Para favorecer la estabilidad y evitar un número excesivo de parámetros, la red reemplaza el `Flatten` por un `GlobalAveragePooling2D`, seguido de capas densas con `Dropout`. Esta configuración suma 423,490 parámetros y busca un buen equilibrio entre capacidad representacional y regularización para imágenes de 224×224.

El segundo enfoque recurrió a Transfer Learning con ResNet50 pre-entrenada en ImageNet con el feature extractor congelado. Sobre esa base se añadieron `GlobalAveragePooling2D` y un clasificador con capas densas, `BatchNormalization` y `Dropout`. El modelo tiene 24,146,434 parámetros, de los cuales 558,210 son entrenables en esta primera fase.

El entrenamiento de ambos modelos uso `Adam` y callbacks de `EarlyStopping` (monitorizando `val_accuracy` con paciencia 3) y `ReduceLROnPlateau` (reducción de la tasa de aprendizaje al detectar estancamiento en `val_loss`). En las primeras épocas, la CNN simple se estabilizó alrededor de 0.72–0.73 de `val_accuracy`, mientras que el modelo con ResNet50, partiendo más bajo (~0.62), progresó hasta ~0.71 en validación. 

En test, la CNN desde cero alcanzó 73.30% de exactitud y el modelo con Transfer Learning obtuvo 70.80%, para una diferencia de -2.50 puntos porcentuales a favor del enfoque desde cero. 

El análisis por clase mostró que la CNN simple presentó mejor recall para “Cat” (0.82) a costa de un menor recall para “Dog” (0.65), mientras que el Transfer Learning invirtió esa tendencia (recall “Dog” 0.82 y “Cat” 0.59). En términos de precisión, la CNN simple favoreció “Dog” (0.79) y el TL favoreció “Cat” (0.76). 

Además, se ejecutó un experimento breve con arquitecturas livianas. En pocas épocas, MobileNetV2 obtuvo 98.70% de exactitud (≈2.62M de parámetros), superando claramente a EfficientNetB0 (48.90%, ≈4.41M) y a la propia ResNet50 en este régimen de entrenamiento corto (51.50%). Dado lo atípico del 98.70% en un escenario acelerado, este resultado debería validarse con múltiples semillas, mayor número de épocas y controles estrictos para descartar cualquier fuga de información o efectos de sobreajuste inadvertido.

## Reflexión

Este ejercicio refuerza que la calidad del preprocesamiento, la elección de arquitectura y la estrategia de entrenamiento inciden de forma decisiva en el desempeño final. El hecho de que la CNN desde cero superara a ResNet50 congelada sugiere que, para este dataset y bajo estas condiciones, la arquitectura diseñada resultó más adecuada que reutilizar representaciones genéricas sin ajuste fino.

Para mejorar, sería conveniente introducir una política de data augmentation más agresiva (volteos, rotaciones, jitter de color, recortes aleatorios) que incremente la diversidad del entrenamiento. En el modelo de Transfer Learning, el siguiente paso natural es habilitar fine-tuning parcial descongelando un subconjunto de capas finales de la base con un learning rate reducido. 

## Referencias

https://colab.research.google.com/drive/1qfOTqIPp7WUWTobZki3J1o2PLvgsX1ji?usp=sharing