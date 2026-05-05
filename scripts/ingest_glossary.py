import re
import json
from pathlib import Path

input_path = Path("data/raw/glossary.txt")
output_path = Path("data/processed/chunks_glossary.jsonl")

def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

text = normalize_text(text)
entries = re.split(r"\n=+\n", text)

chunks = []
chunk_num = 1

for entry in entries:
    entry = entry.strip()
    if not entry:
        continue

    term_match = re.search(r"TERM:\s*(.+)", entry)
    url_match = re.search(r"URL:\s*(.+)", entry)

    term = term_match.group(1).strip() if term_match else f"unknown_{chunk_num}"
    url = url_match.group(1).strip() if url_match else ""

    body = re.sub(r"TERM:\s*.+\n?", "", entry)
    body = re.sub(r"URL:\s*.+\n?", "", body)
    body = normalize_text(body)

    chunks.append({
        "chunk_id": f"glossary_{chunk_num:04d}",
        "source": "glossary",
        "rule_ref": term, 
        "section_title": term,
        "section_type": "term",
        "text": body,
        "url": url
    })

    chunk_num += 1

write_jsonl(output_path, chunks)
print(f"wrote {len(chunks)} chunks to {output_path}")