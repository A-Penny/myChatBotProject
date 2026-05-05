import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

chunks_path = Path("data/processed/chunks.jsonl")
vectorizer_path = Path("models/tfidf_vectorizer.pkl")
matrix_path = Path("models/tfidf_matrix.pkl")
metadata_path = Path("models/chunk_metadata.pkl")

texts = []
metadata = []

with open(chunks_path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        section_title = obj.get("section_title", "")
        body_text = obj.get("text", "")
        combined_text = f"{section_title} {body_text}".strip()

        texts.append(combined_text)
        metadata.append({
            "chunk_id": obj["chunk_id"],
            "source": obj.get("source"),
            "rule_ref": obj.get("rule_ref"),
            "section_type": obj.get("section_type"),
            "section_title": obj.get("section_title"),
            "page_start": obj.get("page_start")
        })

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))

tfidf_matrix = vectorizer.fit_transform(texts)

vectorizer_path.parent.mkdir(parents=True, exist_ok=True)

with open(vectorizer_path, "wb") as f:
    pickle.dump(vectorizer, f)

with open(matrix_path, "wb") as f:
    pickle.dump(tfidf_matrix, f)

with open(metadata_path, "wb") as f:
    pickle.dump(metadata, f)

print("built the tf-idf retrieval idnex")
print(f"number of chunks: {len(texts)}")
print(f"matricx shape: {tfidf_matrix.shape}")

