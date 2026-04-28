from collections import Counter

def build_vocab(docs):
    vocab = set()
    for doc in docs:
        vocab.update(doc)
    return list(vocab)

def vectorize(doc, vocab):
    counter = Counter(doc)
    return [counter[word] for word in vocab]