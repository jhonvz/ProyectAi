import pandas as pd
import pickle


from model.preprocess import clean_text
from model.utils import build_vocab, vectorize
from model.naive_bayes import NaiveBayes
from model.kfold import run_kfold

# 1. cargar dataset
df = pd.read_csv("data/dataset.csv")

# validación básica
if "instruction" not in df.columns or "intent" not in df.columns:
    raise Exception("El dataset no tiene las columnas necesarias")

texts = df["instruction"].astype(str)
labels = df["intent"].astype(str)

# K-FOLDS
print("\n=== K-FOLDS ===")
run_kfold(texts, labels, k=5)

# 2. preprocesar
docs = [clean_text(t) for t in texts]

# 3. vocabulario
vocab = build_vocab(docs)

# 4. vectorizar
X = [vectorize(doc, vocab) for doc in docs]
y = list(labels)

# 5. entrenar
model = NaiveBayes()
model.train(X, y)

print("Modelo entrenado")

# 6. guardar modelo
with open("saved_model/model.pkl", "wb") as f:
    pickle.dump((model, vocab), f)

print("Modelo guardado")

# 7. prueba rápida (obligatoria)
test_text = "I want a refund for my order"

tokens = clean_text(test_text)
vector = vectorize(tokens, vocab)

prediction = model.predict(vector)

print("Test:", test_text)
print("Predicción:", prediction)
