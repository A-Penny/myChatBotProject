import time
from pathlib import Path

from intent_classifier import get_intent_response, predict_intent
from retrieve import load_index, load_chunks_by_id, retrieve

chunks_path = Path("data/processed/chunks.jsonl")

min_answer_score = 0.12

def format_citation(meta: dict) -> str:
    parts = []

    src = meta.get("source")
    if src:
        parts.append(src)
    
    rule_ref = meta.get("rule_ref")
    if rule_ref:
        parts.append(f"Rule {rule_ref}")
    
    section_type = meta.get("section_type")
    if section_type and section_type not in {"rule", "term"}:
        parts.append(section_type.replace("_", " "))

    page = meta.get("page_start")
    if page is not None:
        parts.append(f"p. {page}")

    section_title = meta.get("section_title")
    if section_title:
        parts.append(f"'{section_title}")
    
    return " | ".join(parts) if parts else "source unknown"

def shorten(text: str, max_chars: int=900) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."

def answer_from_top1(query, vectorizer, tfidf_matrix, metadata, chunk_lookup, show_debug=False):
    top = retrieve(
        query=query,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        metadata=metadata,
        chunk_lookup=chunk_lookup,
        top_k=5
    )

    best = top[0]

    score_for_confidence = best["adjusted_score"]
    if score_for_confidence >= 0.35:
        confidence = "High"
    elif score_for_confidence >= 0.20:
        confidence = "Medium"
    else:
        confidence = "Low"

    result = {
        "answer_text": shorten(best.get("text", ""), 1000),
        "citation":format_citation(best),
        "raw_score": best["score"],
        "score": best["adjusted_score"],
        "confidence": confidence,
        "chunk_id": best["chunk_id"],
        "rule_ref": best.get("rule_ref"),
        "source": best.get("source"),
        "page_start": best.get("page_start"),
        "section_title": best.get("section_title"),
        "topk": top
    }

    if show_debug:
        result["full_text"] = best.get("text", "")
    
    return result

def build_rule_line(result):
    if result.get("rule_ref"):
        return f"Rule {result['rule_ref']}"
    elif result.get("section_title"):
        return result["section_title"]
    else:
        return "Rule reference unavailable"
    
def print_chatbot_answer(query: str, result: dict, show_topk=False):
    print("Chatbot: ", result["answer_text"])

    print("\nRule:")
    print(build_rule_line(result))

    print("\nSource;")
    source_parts = []
    if result.get("source"):
        source_parts.append(str(result["source"]))
    if result.get("page_start") is not None:
        source_parts.append(f"page {result['page_start']}")
    if result.get("section_title"):
        source_parts.append(result["section_title"])
    if source_parts:
        print(", ".join(source_parts))
    else:
        print("Unknown source")
    
    print(
        f"Confidence: {result['confidence']} "
        f"(adjusted_similartiy={result['score']:.4f}, raw={result['raw_score']:.4f})"
    )
    print("chunk id: ", result["chunk_id"])

    if show_topk:
        print("\nTop matches:")
        for rank, r in enumerate(result["topk"], start=1):
            print(
                f" {rank}. adjusted={r['adjusted_score']:.4f} "
                f"raw={r['score']:.4f} {format_citation(r)} [{r.get('chunk_id')}]"
            )
    print("=" * 50 + "\n")

def thinking():
    print("\nThinking", end="", flush=True)
    for _ in range(3):
        time.sleep(0.5)
        print(".", end="", flush=True)
    print("\n")

def main():
    vectorizer, tfidf_matrix, metadata = load_index()
    chunk_lookup = load_chunks_by_id(chunks_path)

    print("MLB Rules Chatbot")
    print("type a question. Type 'quit' to exit.")
    print("Commands: /debug on | /debug off | /topk on | /topk off\n")

    debug = False
    show_topk = False

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break
        if query.startswith("/debug"):
            debug = query.lower().endswith("on")
            print(f"Debug mode: {'ON' if debug else 'OFF'} \n")
            continue
        if query.startswith("/topk"):
            show_topk = query.lower().endswith("on")
            print(f"top-k display: {'ON' if show_topk else 'OFF'}\n")
        
        intent = predict_intent(query)
        response = get_intent_response(intent)
        if response:
            thinking()
            print(f"Chatbot: {response}\n")
            continue
        result = answer_from_top1(
            query=query, 
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            metadata=metadata,
            chunk_lookup=chunk_lookup,
            show_debug=debug
        )

        thinking()
        if result["score"] < min_answer_score:
            print("Chatbot: I'm sorry I don't have a rule for that question.")
        else:
            print_chatbot_answer(query, result, show_topk=show_topk)

if __name__ == "__main__":
    main()