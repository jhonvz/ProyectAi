# Clasificación de Solicitudes a Mesa de Ayuda
Naïve Bayes desde cero | Primer Semestre 2026

## Descripción

Este proyecto implementa un sistema de clasificación automática de tickets de soporte utilizando el algoritmo Naïve Bayes Multinomial construido desde cero.

El sistema permite clasificar solicitudes de clientes en diferentes categorías como:
- cancel_order
- change_order
- get_refund
- place_order
- complaint
- track_order
- entre otras

Incluye:
- Motor de inferencia en Python
- Evaluación con K-Folds Cross Validation
- Métricas (Accuracy, Precision, Recall, F1)
- Matriz de confusión
- Aplicación web con Flask para predicción en tiempo real

---

## Estructura del Proyecto


project/
│
├── data/
│ └── dataset.csv
│
├── model/
│ ├── naive_bayes.py
│ ├── preprocess.py
│ └── utils.py
│
├── app/
│ └── app.py
│
├── saved_model/
│ └── model.pkl
│
├── main.py
└── README.md


---

## Requisitos

- Python 3.10+
- pip

---

## Instalación

1. Clonar el repositorio:


git clone <URL_DEL_REPOSITORIO>
cd project


2. Crear entorno virtual:


python -m venv .venv


3. Activar entorno virtual:

Windows (PowerShell):

.venv\Scripts\Activate


4. Instalar dependencias:


pip install pandas nltk flask openpyxl


5. Descargar recursos de NLTK:

Abrir Python e ingresar:


import nltk
nltk.download('punkt')
nltk.download('stopwords')


---

## Dataset

Se utiliza el dataset:

Bitext Customer Support Dataset

Debe colocarse en:


data/dataset.csv


Columnas requeridas:
- instruction (texto)
- intent (categoría)

---

## Entrenamiento del Modelo

Ejecutar:


python main.py


Esto realizará:
- Preprocesamiento de texto
- Construcción de vocabulario
- Entrenamiento del modelo Naïve Bayes
- Evaluación con K-Folds (K=5)
- Cálculo de métricas
- Generación de matrices de confusión (Excel)
- Guardado del modelo en:


saved_model/model.pkl


---

## Ejecución de la Aplicación Web

Ejecutar:


python app/app.py


Luego abrir en el navegador:


http://127.0.0.1:5000


---

## Uso

1. Ingresar un texto de solicitud (ticket)
2. Enviar el formulario
3. El sistema mostrará la categoría predicha

Ejemplo:


Input:
"I want a refund for my order"

Output:
get_refund


---

## Tecnologías Utilizadas

- Python
- Flask
- NLTK
- Pandas

---

## Funcionalidades Implementadas

- Preprocesamiento de texto
- Bag of Words
- Naïve Bayes Multinomial (desde cero)
- Laplace Smoothing
- Suma de logaritmos
- K-Folds Cross Validation
- Métricas por clase
- Matriz de confusión
- Persistencia del modelo
- Aplicación web

---

## Notas Importantes

- No se utilizaron librerías de machine learning como scikit-learn
- El modelo fue implementado completamente desde cero
- El dataset solo se usa en entrenamiento

---

## Autor
Jhonatan Velasquez Fuentes - 1137521
Proyecto académico - Ingeniería en Informática
