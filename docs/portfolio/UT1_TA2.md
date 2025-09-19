---
title: "Práctica 2: Feature Engineering simple + Modelo base"
date: 2025-08-21
---

# Práctica 2: Feature Engineering simple + Modelo base

## Contexto
En esta práctica realizaremos feature engineering para mejorar la representación de los datos y construiremos un modelo de Regresión Logística con Scikit-learn.  

También compararemos su rendimiento con un baseline para entender si nuestro modelo realmente aporta valor.

## Objetivos
- Comprender la importancia del Feature Engineering para mejorar la calidad de los datos para el modelo.
- Crear nuevas variables derivadas del dataset original.
- Entrenar un modelo de Regresión Logística y evaluarlo.
- Comparar el rendimiento con un baseline.
- Analizar resultados mediante métricas de clasificación y la matriz de confusión.

## Actividades (con tiempos estimados)
- Investigación de Scikit-learn (10 min)
- Preprocesamiento y features (15 min)
- Modelo base y baseline (20 min)

## Desarrollo

### 1. Investigación de Scikit-learn

#### LogisticRegression
- Resuelve problemas de clasificación binaria y multiclase.
- Parámetros importantes:
  - `penalty`: tipo de regularización (l1, l2).
  - `C`: controla la fuerza de regularización.
  - `solver`: método numérico de optimización.
- `solver='liblinear'`: útil para datasets pequeños y soporta `l1`.  

#### DummyClassifier
- Genera un modelo baseline muy simple.  
- Estrategias: `most_frequent`, `stratified`, `uniform`.  
- Si un modelo complejo no supera al baseline, no es mejor que adivinar.

#### train_test_split
- `stratify=y`: mantiene la proporción de clases en train/test.  
- `random_state`: garantiza reproducibilidad.  
- Tamaño de test: común usar 20%.

#### Métricas de evaluación
- `classification_report`: incluye `precision`, `recall`, `f1-score`, `support`.
- Matriz de confusión: muestra errores en FP y FN.
- `accuracy`: buena si las clases están balanceadas, si no, conviene usar `f1` o `recall`.

## 2. Preprocesamiento y Feature Engineering
Feature Engineering es transformar y crear nuevas variables que hagan más fácil para el modelo encontrar patrones.

```python linenums="1"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette('deep')

!pip -q install kaggle
from google.colab import files
files.upload()  # Subí tu archivo kaggle.json descargado
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c titanic -p data
!unzip -o data/titanic.zip -d data

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

from pathlib import Path
try:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT = Path('/content/drive/MyDrive/IA-UT1')
except Exception:
    ROOT = Path.cwd() / 'IA-UT1'

DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
for d in (DATA_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
print('Outputs →', ROOT)
```

```python linenums="1"
df = train.copy()

# 🚫 PASO 1: Manejar valores faltantes (imputación)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Valor más común
df['Fare'] = df['Fare'].fillna(df['Fare'].median())              # Mediana
df['Age'] = df['Age'].fillna(df.groupby(['Sex','Pclass'])['Age'].transform('median'))

# 🆕 PASO 2: Crear nuevas features útiles
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

df['Title'] = df['Name'].str.extract(',\s*([^\.]+)\.')
rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
df['Title'] = df['Title'].replace(rare_titles, 'Rare')

# 🔄 PASO 3: Preparar datos para el modelo
features = ['Pclass','Sex','Age','Fare','Embarked','FamilySize','IsAlone','Title','SibSp','Parch']
X = pd.get_dummies(df[features], drop_first=True)
y = df['Survived']

X.shape, y.shape
```

```python linenums="1"
((891, 14), (891,))
```

### 3. Modelo base y baseline
```python linenums="1"
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)
baseline_pred = dummy.predict(X_test)

lr = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
lr.fit(X_train, y_train)
pred = lr.predict(X_test)

print('Baseline acc:', accuracy_score(y_test, baseline_pred))
print('LogReg acc  :', accuracy_score(y_test, pred))

print('\nClassification report (LogReg):')
print(classification_report(y_test, pred))

print('\nConfusion matrix (LogReg):')
print(confusion_matrix(y_test, pred))
```

```python linenums="1"
Baseline acc: 0.6145251396648045
LogReg acc  : 0.8156424581005587

Classification report (LogReg):
              precision    recall  f1-score   support

           0       0.82      0.89      0.86       110
           1       0.80      0.70      0.74        69

    accuracy                           0.82       179
   macro avg       0.81      0.79      0.80       179
weighted avg       0.81      0.82      0.81       179


Confusion matrix (LogReg):
[[98 12]
 [21 48]]
```

## Preguntas

### ¿En qué casos se equivoca más el modelo: cuando predice que una persona sobrevivió y no lo hizo, o al revés?
Se equivoca más al predecir que una persona sobrevivió y no lo hizo (FN - 21 casos) que al predecir que no sobrevivió cuando sí lo hizo (FP - 12 casos), según la matriz de confusión.

### ¿El modelo acierta más con los que sobrevivieron o con los que no sobrevivieron?
Acierta más con los que no sobrevivieron (98 aciertos) que con los que sí sobrevivieron (48 aciertos).

### ¿La Regresión Logística obtiene más aciertos que el modelo que siempre predice la clase más común?
Sí, el baseline tenía una exactitud de ~61% mientras que la Regresión Logística logra ~82%. Esto demuestra que el modelo aprende patrones útiles y supera el baseline.

### ¿Cuál de los dos tipos de error creés que es más grave para este problema?
Los falsos negativos (predijo que no sobrevivió y sí lo hizo) pueden ser más graves si estuvieras usando el modelo para priorizar rescates, porque podrían dejar sin atención a alguien que necesitaba ayuda.

### Mirando las gráficas y números, ¿qué patrones interesantes encontraste sobre la supervivencia?
Respuesta en UT1_TA1

### ¿Qué nueva columna (feature) se te ocurre que podría ayudar a que el modelo acierte más?
- Extraer la letra de la cabina (`C23` → `C`) para reflejar la ubicación en el barco.
- Categorizar edades en rangos (`Child`, `Adult`, `Senior`) para capturar diferencias de supervivencia.

## Reflexión
En esta práctica en ciertos casos es importante crear features derivados para ayudat al modelo a captar relaciones no obvias.

La comparación con el baseline es clave para asegura que el modelo realmente aporta valor.

Finalmente, la Regresión Logística es simple, interpretable y efectivo en datasets pequeños.


## Referencias
- Dataset: https://www.kaggle.com/competitions/titanic/data/
- Kaggle: https://www.kaggle.com/
- Scikit-learn: https://scikit-learn.org/stable/
https://colab.research.google.com/drive/1ll1-Xful-SHlzc6wazeKC5QGFEbXb0O6?usp=sharing