import math
from collections import defaultdict

class NaiveBayes:

    def __init__(self):
        self.class_probs = {}
        self.word_probs = {}
        self.vocab_size = 0
        self.classes = []

    def train(self, X, y):
        self.classes = set(y)
        self.vocab_size = len(X[0])

        class_counts = defaultdict(int)
        word_counts = {c: [0]*self.vocab_size for c in self.classes}
        total_words = {c: 0 for c in self.classes}

        # contar
        for i in range(len(X)):
            c = y[i]
            class_counts[c] += 1

            for j in range(self.vocab_size):
                word_counts[c][j] += X[i][j]
                total_words[c] += X[i][j]

        total_docs = len(y)

        # prior P(clase)
        for c in self.classes:
            self.class_probs[c] = math.log(class_counts[c] / total_docs)

        # likelihood con Laplace
        for c in self.classes:
            self.word_probs[c] = []
            for j in range(self.vocab_size):
                prob = (word_counts[c][j] + 1) / (total_words[c] + self.vocab_size)
                self.word_probs[c].append(math.log(prob))

    def predict(self, x):
        scores = {}

        for c in self.classes:
            score = self.class_probs[c]

            for i in range(len(x)):
                score += x[i] * self.word_probs[c][i]

            scores[c] = score

        return max(scores, key=scores.get)