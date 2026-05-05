import json
from pathlib import Path

rules_path = Path("data/processed/chunks_rules.jsonl")
glossary_path = Path("data/processed/chunks_glossary.jsonl")
comments_path = Path("data/processed/chunks_comments.jsonl")
output_path = Path("data/processed/chunks.jsonl")

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

rules = read_jsonl(rules_path)
glossary = read_jsonl(glossary_path)
comments = read_jsonl(comments_path)

all_chunks = rules + glossary + comments

write_jsonl(output_path, all_chunks)

print(f"wrote {len(all_chunks)} chunks to {output_path}")
print(f"rule book chunksl: {len(rules)}")
print(f"glossary chunks: {len(glossary)}")
print(f"comment chunks: {len(comments)}")