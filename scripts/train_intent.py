from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pickle


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from data.training_data import TRAINING_DATA

TRAIN_DATA = TRAINING_DATA

def main():
    texts = [text for text, intent in TRAIN_DATA]
    labels = [intent for text, intent in TRAIN_DATA]

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,2))

    X = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, labels)

    test_inputs = [
        ("hello", "greeting"),
        ("thanks", "thanks"),
        ("who are you", "identity"),
        ("tell me a fun fact", "knowledge"),
        ("what is a foul ball", "rules_question")
    ]

    for text, expected in test_inputs:
        X_test = vectorizer.transform([text])
        prediction = model.predict(X_test)[0]
        print(f"{text!r} -> predicted: {prediction}, expected: {expected}")
    
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    with open(models_dir / "intent_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    with open(models_dir/"intent_classifier.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Saved intent classifer model.")

if __name__ == "__main__":
    main()