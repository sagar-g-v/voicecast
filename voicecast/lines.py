"""Reading the lines to voice.

Accepted files (by extension):
  .jsonl / .json   objects with "text" and optional "id", "speaker", "hint",
                   "mood", "priority"
  .csv / .tsv      a header row with the same column names
  .txt             one line per row; "Speaker: text" sets the speaker
A line's "hint" is optional extra text used only to detect its mood (for
example the original-language line when the text is a translation).
"""
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Line:
    id: str
    speaker: str
    text: str
    hint: str = ""
    mood: str = ""
    priority: float = 0.0
    n: int = 0                    # position in the input


def text_hash(speaker, text):
    return hashlib.md5(f"{speaker}\x1f{text}".encode("utf-8")).hexdigest()[:16]


def _rows(path):
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".jsonl":
        for raw in p.read_text(encoding="utf-8-sig").splitlines():
            if raw.strip():
                yield json.loads(raw)
    elif ext == ".json":
        data = json.loads(p.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):                       # {id: text} or {id: {...}}
            for k, v in data.items():
                yield dict(v, id=k) if isinstance(v, dict) else {"id": k, "text": v}
        else:
            yield from data
    elif ext in (".csv", ".tsv"):
        with open(p, encoding="utf-8-sig", newline="") as fh:
            yield from csv.DictReader(fh, delimiter="\t" if ext == ".tsv" else ",")
    else:
        for raw in p.read_text(encoding="utf-8-sig").splitlines():
            if not raw.strip():
                continue
            m = re.match(r"^\s*([^:]{1,40}):\s+(.*)$", raw)
            yield {"speaker": m.group(1).strip(), "text": m.group(2)} if m else {"text": raw}


def read(path, default_speaker="narrator", dedupe=True):
    """-> [Line]. Without an "id", a line's id is a hash of speaker + text, so
    identical lines are voiced once (dedupe) and ids are stable across runs."""
    out, seen = [], set()
    for i, row in enumerate(_rows(path)):
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        speaker = str(row.get("speaker") or default_speaker).strip()
        lid = str(row.get("id") or text_hash(speaker, text))
        if dedupe and lid in seen:
            continue
        seen.add(lid)
        try:
            prio = float(row.get("priority") or 0)
        except ValueError:
            prio = 0.0
        out.append(Line(lid, speaker, text, str(row.get("hint") or ""), str(row.get("mood") or ""), prio, i))
    return out
