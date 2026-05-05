import pickle
import numpy as np
import json
import re
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

vectorizer_path = Path("models/tfidf_vectorizer.pkl")
matrix_path = Path("models/tfidf_matrix.pkl")
metadata_path = Path("models/chunk_metadata.pkl")
chunks_path = Path("data/processed/chunks.jsonl")

def normalize_query(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def load_chunks_by_id(chunks_path):
    chunk_lookup = {}
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunk_lookup[obj["chunk_id"]] = obj
    return chunk_lookup

def load_index():
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(matrix_path, "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return vectorizer, tfidf_matrix, metadata

def retrieve(query, vectorizer, tfidf_matrix, metadata, chunk_lookup, top_k=5):
    query = normalize_query(query)
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    scored_results = []

    for idx, raw_score in enumerate(scores):
        meta = metadata[idx]
        chunk_id = meta["chunk_id"]
        full_chunk = chunk_lookup.get(chunk_id, {})

        adjusted_score = float(raw_score)
        source = meta.get("source")
        section_type = meta.get("section_type")
        section_title = (meta.get("section_title") or "").lower()
        text = (full_chunk.get("text") or "").lower()
        full_text = f"{section_title} {text}".lower()

        #prefer glossary termsn for direct definition style quesion:
        if query.startswith("what is"):
            if source == "glossary":
                adjusted_score += 0.03
            if section_type == "term":
                adjusted_score += 0.02
            
        # small preference for scenario supporting chunks
        if "what happens if" in query or "what happens when" in query:
            if section_type in {"comment", "approved_ruling"}:
                adjusted_score += 0.01
            if "running the bases" in section_title:
                adjusted_score += 0.03
        
        #small boost when important query words appear in section title
        query_words = [w for w in query.split() if len(w) > 3]
        title_hits = sum(1 for w in query_words if w in section_title)
        adjusted_score += min(title_hits * 0.01, 0.03)

        # Specific phrase based boost for two-runners on one base scenario
        mentions_two_people = ("two runners" in query) or ("two players" in query)
        mentions_shared_base = ( ("same base" in query) or ("occupy a base" in query) or ("occupy the same base" in query))

        if mentions_shared_base and mentions_two_people:
            if "two runners may not occupy a base" in full_text:
                adjusted_score += 0.1
            if "occupying the base" in full_text:
                adjusted_score += 0.1
            if "running the bases" in section_title:
                adjusted_score += 0.1
            if "following runner shall be out when tagged" in full_text:
                adjusted_score += 0.1

        #direct foul ball phrase boost
        if "foul ball" in query: 
            if "foul ball" in section_title:
                adjusted_score += 0.1
        
        #phrae boost for defensive players count question
        if "how many players" in query and "defense" in query:
            if "nine players" in full_text:
                adjusted_score += 0.1
            if "defensive team" in full_text:
                adjusted_score += 0.1
        
        scored_results.append({
            "score": float(raw_score),
            "adjusted_score": adjusted_score,
            "chunk_id": chunk_id,
            "source": source,
            "rule_ref": meta.get("rule_ref"),
            "section_type": section_type,
            "section_title": meta.get("section_title"),
            "page_start": meta.get("page_start"),
            "text": full_chunk.get("text", "")
        })

    scored_results.sort(key=lambda x: x["adjusted_score"], reverse=True)
    return scored_results[:top_k]

def print_results(query, results):
    print("\n" + "=" * 50)
    print("QUERY:", query)
    print("=" * 50)

    for rank, r in enumerate(results, start=1):
        print(f"\n[Rank {rank}]")
        print(f"raw score: {r['score']:.4f}")
        print(f" adjusted score: {r['adjusted_score']:.4f}")
        print(f"chunk id: {r['chunk_id']}")
        print(f"source: {r['source']}")
        print(f"rule ref: {r['rule_ref']}")
        print(f"section type: {r['section_type']}")
        print(f"section title: {r['section_title']}")
        print(f"page start: {r['page_start']}")
        print("-" * 50)
        print(r["text"][:1500])
        print("-" * 50)

def main():
    vectorizer, tfidf_matrix, metadata = load_index()
    chunk_lookup = load_chunks_by_id(chunks_path)

    print("tfidf retrival system ready.")
    print("type a baseball rules question, or type 'quit' to exit.\n")

    while True:
        query = input("question: ").strip()
        if query.lower() in {"quit", "exit", "q"}:
            print("goodbye.")
            break
        if not query:
            continue

        results = retrieve(
            query=query,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            metadata=metadata,
            chunk_lookup=chunk_lookup, 
            top_k=5
        )

        print_results(query, results)

if __name__ == "__main__":
    main()