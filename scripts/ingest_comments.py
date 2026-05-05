import json
import re
from pathlib import Path


input_path = Path("data/raw/comments.txt")
output_path = Path("data/processed/chunks_comments.jsonl")

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

source_match = re.search(r"Source PDF:\s*(.+)", text)
source_url = source_match.group(1).strip() if source_match else ""

block_pat = re.compile(
    r"RULE:\s*(?P<rule_ref>[^\n]+)\n"
    r"COMMENT:\n"
    r"(?P<comment>.*?)(?=\nRULE:\s*|\Z)",
    re.DOTALL
)

chunks = []

for chunk_num, match in enumerate(block_pat.finditer(text), start=1):
    rule_ref = match.group("rule_ref").strip()
    comment = normalize_text(match.group("comment"))

    if not comment:
        continue

    chunks.append({
        "chunk_id": f"comment_{chunk_num:05d}",
        "source": "comments",
        "rule_ref": rule_ref,
        "section_title": "Comment",
        "section_type": "comment",
        "text": comment,
        "url": source_url
    })

write_jsonl(output_path, chunks)
print(f"wrote {len(chunks)} chunks to {output_path}")