import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    
    # eliminar placeholders tipo {{...}}
    text = re.sub(r"\{\{.*?\}\}", "", text)
    
    # eliminar caracteres especiales
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    tokens = text.split()
    
    # eliminar stopwords + stemming
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return tokens