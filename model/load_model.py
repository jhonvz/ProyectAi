import pickle

def load_model():
    with open("saved_model/model.pkl", "rb") as f:
        model, vocab = pickle.load(f)
    return model, vocab