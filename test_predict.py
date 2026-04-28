from model.load_model import load_model
from model.preprocess import clean_text
from model.utils import vectorize

model, vocab = load_model()

tests = [
    "I want a refund",
    "Where is my order?",
    "Update my billing info",
    "I can't log into my account"
]

for t in tests:
    tokens = clean_text(t)
    vector = vectorize(tokens, vocab)
    print(t, "→", model.predict(vector))