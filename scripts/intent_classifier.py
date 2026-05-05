import pickle
import random
import re
from pathlib import Path

base_dir = Path(__file__).resolve().parents[1]
intent_vectorizer_path = base_dir/"models"/"intent_vectorizer.pkl"
intent_model_path = base_dir/"models"/"intent_classifier.pkl"
min_confidence = 0.45

intent_responses = {
    "greeting": [
        "Hello! Ask me anything about MLB rules",
        "Hi there. What baseball rule can I help you with today?",
        "Hey! I am ready for your baseball rules question."
    ],
    "goodbye": [
        "Goodbye! Have a great day!",
        "see you next time!",
        "Bye! I hope the next call goes your way."
    ],
    "thanks": [
        "You're welcome.",
        "Happy to help.",
        "Anytime."
    ],
    "help": [
        "You can ask me baseball rules questions, like 'what is a balk?', or 'what is a foul ball?'"
    ],
    "identity": [
        "I am an MBL rules chatbot. I search the rulebook, and glossary, and commentaries to help answer baseball rules quesitons. "
    ],
    "knowledge": [
        "Here's a baseball fact: the modern MLB rulebook separates official rules, definitions, and comments so calls can be interpreted consistently.",
        "Fun baseball fact: a foul tip is treated differently from a foul ball when it is caught by the catcher"
    ]
}

intent_patterns = {
    "greeting": [
        r"\b(hello|hi|hey|howdy)\b",
        r"\bgood (morning|afternoon|evening)\b",
        r"\bwhat'?s up\b"
    ],
    "goodbye": [
        r"\b(goodbye|bye|see ya| see you|later)\b",
        r"\btalk to you later\b"
    ],
    "thanks": [
        r"\b(thanks|thank you|appreciate it|much appreciated)\b"
    ],
    "help": [
        r"\b(help|what can you do|how do i use this)\b"
    ],
    "identity": [
        r"\b(who are you|what are you|your name)\b"
    ],
    "knowledge": [
        r"\b(fun fact|tell me something|baseball fact)\b"
    ]
}

try:
    with open(intent_vectorizer_path, "rb") as f:
        intent_vectorizer = pickle.load(f)
    with open(intent_model_path, "rb") as f:
        intent_model = pickle.load(f)
except (FileNotFoundError, ImportError, ModuleNotFoundError, pickle.UnpicklingError):
    intent_vectorizer = None
    intent_model = None

def normalize_text(text):
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)

def predict_intent(user_input):
    normalized = normalize_text(user_input)

    if not normalized:
        return "empty"
    
    for intent, patterns in intent_patterns.items():
        if any(re.search(pattern, normalized) for pattern in patterns):
            return intent
    if intent_vectorizer is not None and intent_model is not None:
        query_vec = intent_vectorizer.transform([normalized])
        prediction = intent_model.predict(query_vec)[0]
        confidence = max(intent_model.predict_proba(query_vec)[0])
        if confidence >= min_confidence:
            return prediction
    return "rules_question"

def get_intent_response(intent):
    responses = intent_responses.get(intent)
    if not responses:
        return ""
    return random.choice(responses)