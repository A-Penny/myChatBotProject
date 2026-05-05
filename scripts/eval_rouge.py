import json
from rouge import Rouge

from retrieve import load_chunks_by_id, load_index, retrieve

chunks_path = "data/processed/chunks.jsonl"
eval_path = "tests/eval_questions.json"

ROUGE = Rouge()

def shorten(text, max_chars=1000):
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."

def get_candidate_answer(question, vectorizer, tfidf_matrix, metadata, chunk_lookup):
    results = retrieve(
        query=question,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        metadata=metadata,
        chunk_lookup=chunk_lookup,
        top_k=5
    )

    if not results:
        return ""
    
    best = results[0]
    return shorten(best.get("text", ""), 1000)

def main():
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    
    vectorizer, tfidf_matrix, metadata = load_index()
    chunk_lookup = load_chunks_by_id(chunks_path)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores= []

    print("\nrunning rouge evaluations...\n")

    for i, item in enumerate(eval_data, start=1):
        question = item["question"]
        reference = item["reference_answer"]
        candidate = get_candidate_answer(
            question,
            vectorizer,
            tfidf_matrix,
            metadata,
            chunk_lookup
        )

        if not candidate.strip():
            r1 = r2 = rL = 0.0
        else:
            scores = ROUGE.get_scores(candidate, reference)[0]
            r1 = scores["rouge-1"]["f"]
            r2 = scores["rouge-2"]["f"]
            rL = scores["rouge-l"]["f"]

        rouge1_scores.append(r1)
        rouge2_scores.append(r2)
        rougeL_scores.append(rL)

        print(f"Question {i}: {question}")
        print(f"ROUGE-1: {r1:.4f}")
        print(f"Rouge-2: {r2:.4f}")
        print(f"Rouge-L: {rL:.4f}")
        print("-" * 50)

    avg_r1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_r2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rL = sum(rougeL_scores) / len(rougeL_scores)

    print("\nOverall average scores;")
    print(f"Average rouge-1: {avg_r1:.4f}")
    print(f"Averge rouge-2: {avg_r2:.4f}")
    print(f"Average rouge-L: {avg_rL:.4f}")

if __name__ == "__main__":
    main()