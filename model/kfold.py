import random
from collections import defaultdict
import pandas as pd

from model.preprocess import clean_text
from model.utils import build_vocab, vectorize
from model.naive_bayes import NaiveBayes


# =========================
# SPLIT K-FOLDS
# =========================
def k_folds_split(X, y, k=5):
    data = list(zip(X, y))
    random.shuffle(data)

    fold_size = len(data) // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else len(data)
        folds.append(data[start:end])

    return folds


# =========================
# ACCURACY
# =========================
def evaluate(y_true, y_pred):
    correct = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i])
    return correct / len(y_true)


# =========================
# MATRIZ DE CONFUSIÓN
# =========================
def build_confusion_matrix(y_true, y_pred):
    classes = sorted(list(set(y_true)))
    matrix = {c: {c2: 0 for c2 in classes} for c in classes}

    for real, pred in zip(y_true, y_pred):
        matrix[real][pred] += 1

    return matrix, classes


# =========================
# MÉTRICAS
# =========================
def compute_metrics(matrix, classes):
    metrics = {}

    for c in classes:
        TP = matrix[c][c]
        FP = sum(matrix[other][c] for other in classes if other != c)
        FN = sum(matrix[c][other] for other in classes if other != c)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    return metrics


def print_metrics(metrics):
    print("\nMetrics por clase:")
    print(f"{'Clase':25} {'Precision':10} {'Recall':10} {'F1':10}")

    for c, m in metrics.items():
        print(f"{c:25} {m['precision']:.4f}     {m['recall']:.4f}     {m['f1']:.4f}")


# =========================
# GUARDAR EXCEL
# =========================
def save_all_matrices(matrices, filename="confusion_matrices.xlsx"):
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        for i, (matrix, classes) in enumerate(matrices):
            df = pd.DataFrame(matrix).T
            df.to_excel(writer, sheet_name=f"Fold_{i+1}")

    print(f"\nMatrices guardadas en {filename}")


# =========================
# K-FOLDS
# =========================
def run_kfold(texts, labels, k=5):
    folds = k_folds_split(texts, labels, k)

    accuracies = []
    all_matrices = []

    # acumulador métricas
    metrics_acum = defaultdict(lambda: {"precision": 0, "recall": 0, "f1": 0})

    for i in range(k):
        print(f"\n====================")
        print(f"Fold {i+1}")
        print(f"====================")

        test_data = folds[i]
        train_data = [item for j in range(k) if j != i for item in folds[j]]

        train_texts = [x for x, y in train_data]
        train_labels = [y for x, y in train_data]

        test_texts = [x for x, y in test_data]
        test_labels = [y for x, y in test_data]

        # PREPROCESAMIENTO
        train_docs = [clean_text(t) for t in train_texts]
        test_docs = [clean_text(t) for t in test_texts]

        # VOCABULARIO (solo train)
        vocab = build_vocab(train_docs)

        # VECTORIZACIÓN
        X_train = [vectorize(doc, vocab) for doc in train_docs]
        X_test = [vectorize(doc, vocab) for doc in test_docs]

        # MODELO
        model = NaiveBayes()
        model.train(X_train, train_labels)

        # PREDICCIÓN
        predictions = [model.predict(x) for x in X_test]

        # ACCURACY
        acc = evaluate(test_labels, predictions)
        accuracies.append(acc)
        print("Accuracy:", acc)

        # MATRIZ
        matrix, classes = build_confusion_matrix(test_labels, predictions)
        all_matrices.append((matrix, classes))

        # MÉTRICAS
        metrics = compute_metrics(matrix, classes)
        print_metrics(metrics)

        # ACUMULAR
        for c in metrics:
            metrics_acum[c]["precision"] += metrics[c]["precision"]
            metrics_acum[c]["recall"] += metrics[c]["recall"]
            metrics_acum[c]["f1"] += metrics[c]["f1"]

    # =========================
    # RESULTADOS FINALES
    # =========================
    avg_acc = sum(accuracies) / k

    print("\n====================")
    print("Accuracy promedio:", avg_acc)
    print("====================")

    # promedio métricas por clase
    final_metrics = {}

    for c in metrics_acum:
        final_metrics[c] = {
            "precision": metrics_acum[c]["precision"] / k,
            "recall": metrics_acum[c]["recall"] / k,
            "f1": metrics_acum[c]["f1"] / k
        }

    print("\n====================")
    print("MÉTRICAS PROMEDIO POR CLASE")
    print("====================")
    print_metrics(final_metrics)

    # Macro F1
    macro_f1 = sum(m["f1"] for m in final_metrics.values()) / len(final_metrics)
    print("\nMacro F1:", macro_f1)

    # guardar matrices
    save_all_matrices(all_matrices)

    return avg_acc, final_metrics, macro_f1