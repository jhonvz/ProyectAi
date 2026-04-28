import sys
import os
import pickle
import uuid

from flask import Flask, render_template, request

#(root del proyecto)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.preprocess import clean_text
from model.utils import vectorize

app = Flask(__name__)

#ruta correcta al modelo
model_path = os.path.join(os.path.dirname(__file__), '..', 'saved_model', 'model.pkl')

with open(model_path, "rb") as f:
    model, vocab = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    ticket_id = None

    if request.method == "POST":
        subject = request.form.get("subject")
        description = request.form.get("description")

        full_text = f"{subject} {description}"

        tokens = clean_text(full_text)
        vector = vectorize(tokens, vocab)
        
        prediction = model.predict(vector)
        
        ticket_id = str(uuid.uuid4())[:8]

    return render_template(
        "index.html",
        prediction=prediction,
        ticket_id=ticket_id
    )


if __name__ == "__main__":
    app.run(debug=True)