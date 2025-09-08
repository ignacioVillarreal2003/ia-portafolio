---
title: "Práctica 4: Regresión Lineal y Logística"
date: 2025-08-22
---

# Práctica 4: Regresión Lineal y Logística

## Contexto
En esta práctica vamos a trabajar con dos algoritmos la regresión lineal y la regresión logística.

Ambos modelos se usan para tareas distintas, pero comparten la misma lógica de entrenamiento, ajustando parámetros a partir de datos para predecir resultados.  

Primero aplicaremos regresión lineal al dataset de Boston Housing para estimar precios de viviendas. Luego aplicaremos regresión logística al dataset de diagnóstico de cáncer de mama para clasificar tumores como benignos o malignos.  

Esto para ver la diferencia entre predecir un valor numérico (regresión) y predecir una categoría (clasificación).  

## Objetivos
- Implementar un modelo de regresión lineal.  
- Implementar un modelo de regresión logística.  
- Evaluar los modelos con métricas.  
- Comparar las diferencias entre ambos enfoques.  

## Actividades (con tiempos estimados)
- Parte 1: Regresión Lineal - Predecir Precios de Casas
    - Setup Inicial
    - Cargar Dataset de Boston Housing
    - Entrenar Regresión Lineal
- Parte 2: Regresión Logística - Diagnóstico Médico
    - Cargar Datos Médicos
    - Entrenar Regresión Logística
    - Preguntas de Reflexión
- Parte 3: Actividad Final - Compara los Dos Modelos
    - Comparación Simple
    - Reflexión Final

## Desarrollo

### 1. Setup Inicial

Importamos librerías para manipulación de datos, visualización y machine learning.

```python linenums="1"
# Importar librerías que vamos a usar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Para los modelos de machine learning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.datasets import load_breast_cancer

print("✅ Setup completo!")
```

### 2. Cargar Dataset de Boston Housing

```python linenums="1"
# === CARGAR DATOS DE CASAS EN BOSTON ===

# 1. Cargar el dataset desde una URL
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
boston_data = pd.read_csv(url)

print("🏠 DATASET: Boston Housing")
print(f"   📊 Forma: {boston_data.shape}")
print(f"   📋 Columnas: {list(boston_data.columns)}")

# 2. Explorar los datos básicamente
print("\n🔍 Primeras 5 filas:")
print(boston_data.head())

# 3. Preparar X (variables independientes) e y (variable dependiente)
# La columna 'medv' es el precio de la casa que queremos predecir
X = boston_data.drop('medv', axis=1)  # Todas las columnas EXCEPTO la que queremos predecir
y = boston_data['medv']                # Solo la columna que queremos predecir

print(f"\n📊 X tiene forma: {X.shape}")
print(f"📊 y tiene forma: {y.shape}")
print(f"🎯 Queremos predecir: Precio de casas en miles de USD")
print(f"📈 Precio mínimo: ${y.min():.1f}k, Precio máximo: ${y.max():.1f}k")
```

```python linenums="1"
🏠 DATASET: Boston Housing
   📊 Forma: (506, 14)
   📋 Columnas: ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat', 'medv']

🔍 Primeras 5 filas:
      crim    zn  indus  chas    nox     rm   age     dis  rad  tax  ptratio  \
0  0.00632  18.0   2.31     0  0.538  6.575  65.2  4.0900    1  296     15.3   
1  0.02731   0.0   7.07     0  0.469  6.421  78.9  4.9671    2  242     17.8   
2  0.02729   0.0   7.07     0  0.469  7.185  61.1  4.9671    2  242     17.8   
3  0.03237   0.0   2.18     0  0.458  6.998  45.8  6.0622    3  222     18.7   
4  0.06905   0.0   2.18     0  0.458  7.147  54.2  6.0622    3  222     18.7   

        b  lstat  medv  
0  396.90   4.98  24.0  
1  396.90   9.14  21.6  
2  392.83   4.03  34.7  
3  394.63   2.94  33.4  
4  396.90   5.33  36.2  

📊 X tiene forma: (506, 13)
📊 y tiene forma: (506,)
🎯 Queremos predecir: Precio de casas en miles de USD
📈 Precio mínimo: $5.0k, Precio máximo: $50.0k
```

### 3. Entrenar Regresión Lineal

```python linenums="1"
# === ENTRENAR MODELO DE REGRESIÓN LINEAL ===

# 1. Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Datos de entrenamiento: {X_train.shape[0]} casas")
print(f"📊 Datos de prueba: {X_test.shape[0]} casas")

# 2. Crear y entrenar el modelo
modelo_regresion = LinearRegression()
modelo_regresion.fit(X_train, y_train)

print("✅ Modelo entrenado!")

# 3. Hacer predicciones
predicciones = modelo_regresion.predict(X_test)

print(f"\n🔮 Predicciones hechas para {len(predicciones)} casas")

# 4. Evaluar qué tan bueno es el modelo con MÚLTIPLES MÉTRICAS
mae = mean_absolute_error(y_test, predicciones)
mse = mean_squared_error(y_test, predicciones)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predicciones)

# Calcular MAPE manualmente
mape = np.mean(np.abs((y_test - predicciones) / y_test)) * 100

print(f"\n📈 MÉTRICAS DE EVALUACIÓN:")
print(f"   📊 MAE (Error Absoluto Medio): ${mae:.2f}k")
print(f"   📊 MSE (Error Cuadrático Medio): {mse:.2f}")
print(f"   📊 RMSE (Raíz del Error Cuadrático): ${rmse:.2f}k")
print(f"   📊 R² (Coeficiente de determinación): {r2:.3f}")
print(f"   📊 MAPE (Error Porcentual Absoluto): {mape:.1f}%")

print(f"\n🔍 INTERPRETACIÓN:")
print(f"   💰 En promedio nos equivocamos por ${mae:.2f}k (MAE)")
print(f"   📈 El modelo explica {r2*100:.1f}% de la variabilidad (R²)")
print(f"   📊 Error porcentual promedio: {mape:.1f}% (MAPE)")

# 5. Comparar algunas predicciones reales vs predichas
print(f"\n🔍 EJEMPLOS (Real vs Predicho):")
for i in range(5):
    real = y_test.iloc[i]
    predicho = predicciones[i]
    print(f"   Casa {i+1}: Real ${real:.1f}k vs Predicho ${predicho:.1f}k")
```

```python linenums="1"
📊 Datos de entrenamiento: 404 casas
📊 Datos de prueba: 102 casas
✅ Modelo entrenado!

🔮 Predicciones hechas para 102 casas

📈 MÉTRICAS DE EVALUACIÓN:
   📊 MAE (Error Absoluto Medio): $3.19k
   📊 MSE (Error Cuadrático Medio): 24.29
   📊 RMSE (Raíz del Error Cuadrático): $4.93k
   📊 R² (Coeficiente de determinación): 0.669
   📊 MAPE (Error Porcentual Absoluto): 16.9%

🔍 INTERPRETACIÓN:
   💰 En promedio nos equivocamos por $3.19k (MAE)
   📈 El modelo explica 66.9% de la variabilidad (R²)
   📊 Error porcentual promedio: 16.9% (MAPE)

🔍 EJEMPLOS (Real vs Predicho):
   Casa 1: Real $23.6k vs Predicho $29.0k
   Casa 2: Real $32.4k vs Predicho $36.0k
   Casa 3: Real $13.6k vs Predicho $14.8k
   Casa 4: Real $22.8k vs Predicho $25.0k
   Casa 5: Real $16.1k vs Predicho $18.8k
```

- MAE (Mean Absolute Error): Promedio de los errores absolutos sin importar si son positivos o negativos.
- MSE (Mean Squared Error): Promedio de los errores al cuadrado, penaliza más los errores grandes.
- RMSE: Raíz cuadrada del MSE, vuelve a las unidades originales del problema.
- R²: Indica qué porcentaje de la variabilidad es explicada por el modelo (0-1, donde 1 es perfecto).
- MAPE: Error porcentual promedio, útil para comparar modelos con diferentes escalas.

### 4. Cargar Datos Médicos

```python linenums="1"
# === CARGAR DATOS DE DIAGNÓSTICO DE CÁNCER ===

# 1. Cargar el dataset de cáncer de mama (que viene con sklearn)
cancer_data = load_breast_cancer()

# 2. Convertir a DataFrame para verlo mejor
X_cancer = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
y_cancer = cancer_data.target  # 0 = maligno, 1 = benigno

print("🏥 DATASET: Breast Cancer (Diagnóstico)")
print(f"   📊 Pacientes: {X_cancer.shape[0]}")
print(f"   📊 Características: {X_cancer.shape[1]}")
print(f"   🎯 Objetivo: Predecir si tumor es benigno (1) o maligno (0)")

# 3. Ver balance de clases
casos_malignos = (y_cancer == 0).sum()
casos_benignos = (y_cancer == 1).sum()

print(f"\n📊 DISTRIBUCIÓN:")
print(f"   ❌ Casos malignos: {casos_malignos}")
print(f"   ✅ Casos benignos: {casos_benignos}")
```

```python linenums="1"
🏥 DATASET: Breast Cancer (Diagnóstico)
   📊 Pacientes: 569
   📊 Características: 30
   🎯 Objetivo: Predecir si tumor es benigno (1) o maligno (0)

📊 DISTRIBUCIÓN:
   ❌ Casos malignos: 212
   ✅ Casos benignos: 357
```

### 5. Entrenamiento y Evaluación

```python linenums="1"
# === ENTRENAR MODELO DE CLASIFICACIÓN ===

# 1. Dividir datos en entrenamiento y prueba
X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(
    X_cancer, y_cancer, test_size=0.2, random_state=42
)

print(f"📊 Datos de entrenamiento: {X_train_cancer.shape[0]} pacientes")
print(f"📊 Datos de prueba: {X_test_cancer.shape[0]} pacientes")

# 2. Crear y entrenar modelo de regresión logística
modelo_clasificacion = LogisticRegression(max_iter=5000, random_state=42)
modelo_clasificacion.fit(X_train_cancer, y_train_cancer)

print("✅ Modelo de clasificación entrenado!")

# 3. Hacer predicciones
predicciones_cancer = modelo_clasificacion.predict(X_test_cancer)

# 4. Evaluar con MÚLTIPLES MÉTRICAS de clasificación
exactitud = accuracy_score(y_test_cancer, predicciones_cancer)
precision = precision_score(y_test_cancer, predicciones_cancer)
recall = recall_score(y_test_cancer, predicciones_cancer)
f1 = f1_score(y_test_cancer, predicciones_cancer)

print(f"\n📈 MÉTRICAS DE CLASIFICACIÓN:")
print(f"   🎯 Exactitud (Accuracy): {exactitud:.3f} ({exactitud*100:.1f}%)")
print(f"   🎯 Precisión (Precision): {precision:.3f} ({precision*100:.1f}%)")
print(f"   🎯 Recall (Sensibilidad): {recall:.3f} ({recall*100:.1f}%)")
print(f"   🎯 F1-Score: {f1:.3f}")

# Mostrar matriz de confusión de forma simple
matriz_confusion = confusion_matrix(y_test_cancer, predicciones_cancer)
print(f"\n🔢 MATRIZ DE CONFUSIÓN:")
print(f"   📊 {matriz_confusion}")
print(f"   📋 [Verdaderos Negativos, Falsos Positivos]")
print(f"   📋 [Falsos Negativos, Verdaderos Positivos]")

# Reporte detallado
print(f"\n📋 REPORTE DETALLADO:")
print(classification_report(y_test_cancer, predicciones_cancer, target_names=['Maligno', 'Benigno']))

print(f"\n🔍 INTERPRETACIÓN MÉDICA:")
print(f"   🩺 Precision: De los casos que predecimos como benignos, {precision*100:.1f}% lo son realmente")
print(f"   🩺 Recall: De todos los casos benignos reales, detectamos {recall*100:.1f}%")
print(f"   🩺 F1-Score: Balance general entre precision y recall: {f1:.3f}")

# 5. Ver ejemplos específicos
print(f"\n🔍 EJEMPLOS (Real vs Predicho):")
for i in range(5):
    real = "Benigno" if y_test_cancer[i] == 1 else "Maligno"
    predicho = "Benigno" if predicciones_cancer[i] == 1 else "Maligno"
    print(f"   Paciente {i+1}: Real: {real} vs Predicho: {predicho}")
```

```python linenums="1"
📊 Datos de entrenamiento: 455 pacientes
📊 Datos de prueba: 114 pacientes
✅ Modelo de clasificación entrenado!

📈 MÉTRICAS DE CLASIFICACIÓN:
   🎯 Exactitud (Accuracy): 0.956 (95.6%)
   🎯 Precisión (Precision): 0.946 (94.6%)
   🎯 Recall (Sensibilidad): 0.986 (98.6%)
   🎯 F1-Score: 0.966

🔢 MATRIZ DE CONFUSIÓN:
   📊 [[39  4]
 [ 1 70]]
   📋 [Verdaderos Negativos, Falsos Positivos]
   📋 [Falsos Negativos, Verdaderos Positivos]

📋 REPORTE DETALLADO:
              precision    recall  f1-score   support

     Maligno       0.97      0.91      0.94        43
     Benigno       0.95      0.99      0.97        71

    accuracy                           0.96       114
   macro avg       0.96      0.95      0.95       114
weighted avg       0.96      0.96      0.96       114


🔍 INTERPRETACIÓN MÉDICA:
   🩺 Precision: De los casos que predecimos como benignos, 94.6% lo son realmente
   🩺 Recall: De todos los casos benignos reales, detectamos 98.6%
   🩺 F1-Score: Balance general entre precision y recall: 0.966

🔍 EJEMPLOS (Real vs Predicho):
   Paciente 1: Real: Benigno vs Predicho: Benigno
   Paciente 2: Real: Maligno vs Predicho: Maligno
   Paciente 3: Real: Maligno vs Predicho: Maligno
   Paciente 4: Real: Benigno vs Predicho: Benigno
   Paciente 5: Real: Benigno vs Predicho: Benigno
```

- Accuracy: Porcentaje de predicciones correctas sobre el total.
- Precision: De todas las predicciones positivas, ¿cuántas fueron realmente correctas?
- Recall (Sensibilidad): De todos los casos positivos reales, ¿cuántos detectamos?
- F1-Score: Promedio armónico entre precision y recall.
- Matriz de Confusión: Tabla que muestra predicciones vs valores reales.

### 6. Preguntas de Reflexión

#### ¿Cuál es la diferencia principal entre regresión lineal y logística?
La regresión lineal predice valores numéricos continuos (ej: precios de casas). La regresión logística predice categorías (ej: benigno/maligno).

#### ¿Por qué dividimos los datos en entrenamiento y prueba?
Para evaluar al modelo en datos que nunca vio y evitar sobreajuste.

#### ¿Qué significa una exactitud del 95%?
Que de cada 100 pacientes, el modelo clasifica correctamente a 95.

#### ¿Cuál es más peligroso: predecir "benigno" cuando es "maligno", o al revés?
Un benigno cuando en realidad es maligno, ya que podría no detectarse un cáncer real.

### 7. Comparación Simple

| Aspecto           | Regresión Lineal  | Regresión Logística              |
| ----------------- | ----------------- | -------------------------------- |
| Qué predice       | Números continuos | Categorías                       |
| Ejemplo de uso    | Precio de casas   | Diagnóstico médico (benigno/mal) |
| Rango de salida   | Valores reales    | 0 a 1 (probabilidad)             |
| Métrica principal | MAE, RMSE, R²     | Accuracy, Precision, Recall, F1  |

### 8. Reflexión Final

#### ¿Cuál modelo usarías para predecir el salario de un empleado?
Regresión lineal por ser continuo.

#### ¿Cuál modelo usarías para predecir si un email es spam?
Regresión logística para clasificar si un email es spam o no spam.

#### ¿Por qué es importante separar datos de entrenamiento y prueba?
Para tener una estimación realista del rendimiento del modelo.

## Reflexión
La regresión lineal y la logística son modelos simples, una permite estimar valores continuos y la otra clasificar categorías. 

Lo importante no es solo entrenar el modelo, sino también elegir las métricas correctas para evaluar su desempeño.

Esta tarea nos muestra cómo con modelos muy simples podemos resolver problemas reales de predicción y clasificación, y lo importante que es elegir la métrica adecuada según el tipo de problema.

## Referencias
- Dataset: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Dataset: https://www.kaggle.com/datasets/altavish/boston-housing-dataset
- Scikit-learn: https://scikit-learn.org/stable/index.html