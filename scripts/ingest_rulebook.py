import json
import re

from pathlib import Path

input_path = Path("data/raw/rulebook.txt")
output_path = Path("data/processed/chunks_rules.jsonl")

page_pat = re.compile(r"^=== PAGE (\d+) ===$")
chapter_pat = re.compile(r"^(\d+\.00)[--](.+)$")
main_rule_pat = re.compile(r"^(\d+\.\d+(?:\([a-z0-9]+\))*)\s+(.+)$")
subsection_pat = re.compile(r"^\(([a-z0-9]+)\)\s+(.+)$")
comment_pat = re.compile(r"^COMMENT:$")
approved_pat = re.compile(r"^APPROVED RULING:\s*(.*)$")
page_header_pat = re.compile(r"^Rule\s+.+$")

def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

with open(input_path, "r", encoding="utf-8") as f:
    lines = [line.rstrip() for line in f]

chunks = []
chunk_counter = 1

current_page = None
current_chapter_ref = None
current_chapter_title = None
current_rule_ref = None
current_rule_title = None

current_chunk = None

def flush_chunk():
    global current_chunk, chunk_counter
    if current_chunk and current_chunk["text"].strip():
        current_chunk["chunk_id"] = f"rulebook_{chunk_counter:05d}"
        current_chunk["text"] = normalize_text(current_chunk["text"])
        chunks.append(current_chunk)
        chunk_counter += 1
    current_chunk = None

for line in lines:
    line = line.strip()

    if not line:
        if current_chunk is not None:
            current_chunk["text"] += "\n"
        continue
    m = page_pat.match(line)
    if m:
        current_page = int(m.group(1))
        continue
    if current_page is None or current_page < 13:
        continue
    if page_header_pat.match(line):
        continue
    m = chapter_pat.match(line)
    if m:
        current_chapter_ref = m.group(1)
        current_chapter_title = m.group(2).strip()
        continue

    if comment_pat.match(line):
        flush_chunk()
        current_chunk = {
            "source": "rulebook",
            "page_start": current_page,
            "chapter_ref": current_chapter_ref,
            "chapter_title": current_chapter_title,
            "rule_ref": current_rule_ref,
            "section_title": "Comment",
            "section_type": "comment",
            "text": ""
        }
        continue

    m = approved_pat.match(line)
    if m:
        flush_chunk()
        current_chunk = {
            "source": "rulebook",
            "page_start": current_page,
            "chapter_ref": current_chapter_ref,
            "chapter_title": current_chapter_title,
            "rule_ref": current_rule_ref,
            "section_title": "Approved Ruling",
            "section_type": "approved_ruling",
            "text": m.group(1).strip()
        }
        continue
    m = main_rule_pat.match(line)
    if m:
        rule_ref = m.group(1)
        rule_title = m.group(2).strip()

        if len(rule_ref) > 3:
            flush_chunk()
            current_rule_ref = rule_ref
            current_rule_title = rule_title
            current_chunk = {
                "source": "rulebook",
                "page_start": current_page,
                "chapter_ref": current_chapter_ref,
                "chapter_title": current_chapter_title,
                "rule_ref": current_rule_ref,
                "section_title": current_rule_title,
                "section_type": "rule",
                "text": ""
            }
            continue
    if current_chunk is None:
        continue

    current_chunk["text"] += line + " "

flush_chunk()

output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(json.dumps(chunk) + "\n")

print(f"wrote {len(chunks)} chunks to {output_path}")