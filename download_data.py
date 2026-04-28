from datasets import load_dataset
import pandas as pd

# cargar dataset
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

# convertir a pandas
df = dataset["train"].to_pandas()

# guardar en tu carpeta data
df.to_csv("data/dataset.csv", index=False)

print("Dataset descargado y guardado")