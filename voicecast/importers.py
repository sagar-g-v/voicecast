"""Turn scripts and stories into a voiceable line list (JSONL).

  screenplay  Fountain (.fountain) or plain-text screenplays: CHARACTER cues
              in capitals, dialogue under them, (parentheticals) become the
              line's delivery hint, action paragraphs become narration
  subtitles   .srt / .vtt: one line per cue, with start / end times; speakers
              from VTT <v Name> tags or "NAME:" prefixes
  prose       novels / short stories: quoted speech attributed from nearby
              "said Alice" / "Bob asked" phrases, the rest is narration
  dialogue    "Name: text" transcripts, chat logs, radio plays ("[00:01] Name: text")

Every line gets an "id" in reading order (stable for the same input), a
"speaker", the "text", and where known a "hint" (parenthetical, stage
direction) that drives the tone, "kind" (dialogue / narration) and timing.
"""
import json
import re
from pathlib import Path

NARRATOR = "Narrator"

# ---------------------------------------------------------------- screenplay
_SCENE = re.compile(r"^(?:\.(?!\.)|(?:INT|EXT|EST|INT\./EXT|INT/EXT|I/E)[ .])", re.I)
_TRANSITION = re.compile(r"^(?:>|[A-Z\s]+TO:$|FADE (?:IN|OUT)[.:]?$|CUT TO BLACK\.?$)")
_EXTENSION = re.compile(r"\s*\((?:V\.?O\.?|O\.?S\.?|O\.?C\.?|CONT'?D|CONT\.|CONTINUED|PRE-?LAP|FILTERED|ON PHONE|INTO PHONE|"
                        r"OVER|SUBTITLE[SD]?)\)\s*", re.I)


def _is_cue(line, nxt):
    s = line.strip()
    if not s or not nxt.strip():
        return False
    if s.startswith("@"):
        return True                                     # Fountain forced character
    name = _EXTENSION.sub("", s).rstrip("^ ").strip()
    if not name or _SCENE.match(s) or _TRANSITION.match(s) or s.startswith(("!", "#", "=", "~", ">")):
        return False
    return bool(re.search(r"[A-Z]", name)) and name == name.upper() and len(name) <= 40 \
        and not re.search(r"[.!?]$", name)


def screenplay(text, *, narration=True, scene_headings=False):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)  # Fountain boneyard
    text = re.sub(r"\[\[.*?\]\]", "", text, flags=re.S)  # Fountain notes
    rows = text.replace("\r\n", "\n").split("\n")
    # skip a Fountain title page (key: value lines before the first blank line)
    if rows and re.match(r"^\s*(Title|Credit|Author|Source|Draft date|Contact)\s*:", rows[0], re.I):
        while rows and rows[0].strip():
            rows.pop(0)
    out, i, para = [], 0, []

    def flush_action():
        if para and narration:
            t = " ".join(p.strip() for p in para).strip()
            if t:
                out.append({"speaker": NARRATOR, "text": t, "kind": "narration"})
        para.clear()

    while i < len(rows):
        line = rows[i].rstrip()
        nxt = rows[i + 1] if i + 1 < len(rows) else ""
        s = line.strip()
        if not s:
            flush_action()
        elif _SCENE.match(s):
            flush_action()
            if scene_headings and narration:
                out.append({"speaker": NARRATOR, "text": s.lstrip("."), "kind": "scene"})
        elif _TRANSITION.match(s) or s.startswith(("#", "=", ">")):
            flush_action()
        elif _is_cue(line, nxt):
            flush_action()
            speaker = _EXTENSION.sub(" ", s.lstrip("@")).rstrip("^ ").strip()
            speaker = re.sub(r"\s+", " ", speaker).title()
            i += 1
            hint, said = "", []
            while i < len(rows) and rows[i].strip():
                d = rows[i].strip()
                if re.fullmatch(r"\(.*\)", d):          # parenthetical: how it is said
                    if said:
                        out.append({"speaker": speaker, "text": " ".join(said), "hint": hint, "kind": "dialogue"})
                        said = []
                    hint = d.strip("()")
                else:
                    said.append(d.lstrip("~").strip())
                i += 1
            if said:
                out.append({"speaker": speaker, "text": " ".join(said), "hint": hint, "kind": "dialogue"})
            continue
        else:
            para.append(s.lstrip("!"))
        i += 1
    flush_action()
    return out


# ---------------------------------------------------------------- subtitles
_TIME = r"(\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{1,3}"


def subtitles(text):
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip())
    out, last = [], None
    for b in blocks:
        ls = [x for x in b.split("\n") if x.strip()]
        k = next((j for j, x in enumerate(ls) if re.search(_TIME + r"\s*-->\s*" + _TIME, x)), None)
        if k is None:
            continue
        start, end = (t.strip().split(" ")[0] for t in ls[k].split("-->"))
        body = " ".join(ls[k + 1:])
        speaker = None
        m = re.match(r"^\s*<v(?:\.[\w.]+)?\s+([^>]+)>", body)
        if m:
            speaker = m.group(1).strip()
        body = re.sub(r"<[^>]+>", "", body)             # styling tags
        m = re.match(r"^\s*-?\s*([A-Z][\w .'-]{0,30}):\s+(.*)$", body)
        if not speaker and m:
            speaker, body = m.group(1).strip(), m.group(2)
        body = re.sub(r"^\s*-\s*", "", body).strip()
        if not body:
            continue
        hint = ""
        m = re.match(r"^\[([^\]]+)\]\s*(.*)$", body)   # [whispering] Hello
        if m:
            hint, body = m.group(1), m.group(2)
        speaker = speaker or last or NARRATOR
        last = speaker
        out.append({"speaker": speaker, "text": body, "hint": hint, "start": start.replace(",", "."),
                    "end": end.replace(",", "."), "kind": "dialogue"})
    return out


# ---------------------------------------------------------------- prose
SAID = (r"said|says|asked|asks|replied|replies|answered|whispered|shouted|yelled|cried|called|"
        r"muttered|murmured|added|continued|began|explained|exclaimed|laughed|sighed|snapped|"
        r"growled|hissed|insisted|admitted|agreed|warned|repeated|pleaded|demanded|told|breathed|"
        r"interrupted|protested|teased|joked|groaned|stammered|whimpered")
_QUOTE = re.compile(r"[\"“]([^\"”]+)[\"”]|(?<![A-Za-z])['‘]([^'’]{2,}?)['’](?![A-Za-z])")
_NAME = r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)"
_AFTER = re.compile(rf"^\s*,?\s*(?:{_NAME}\s+(?:{SAID})|(?:{SAID})\s+{_NAME})\b", re.U)
_BEFORE = re.compile(rf"(?:{_NAME}\s+(?:{SAID})|(?:{SAID})\s+{_NAME})\s*(?:,|:)?\s*$", re.U)
_NOT_NAMES = {"He", "She", "They", "It", "I", "We", "You", "The", "Then", "But", "And", "His", "Her"}
_VERB = re.compile(rf"\b(whispered|shouted|yelled|cried|muttered|murmured|laughed|sighed|snapped|growled|hissed|"
                   rf"pleaded|groaned|stammered|whimpered|breathed|teased|joked)\b", re.I)


def _who(m):
    if not m:
        return None
    name = next((g for g in m.groups() if g), None)
    return None if name in _NOT_NAMES else name


def prose(text, *, narration=True, unknown="Unknown"):
    """Paragraph by paragraph: quotes -> the attributed speaker (the paragraph's
    named speaker, else the last speaker of a back-and-forth), the rest -> narration.
    Attribution is heuristic: check the result with `voicecast cast`."""
    out, recent = [], []
    for para in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        para = " ".join(para.split())
        if not para:
            continue
        quotes = list(_QUOTE.finditer(para))
        if not quotes:
            if narration:
                out.append({"speaker": NARRATOR, "text": para, "kind": "narration"})
            continue
        speaker = None
        for q in quotes:
            speaker = speaker or _who(_AFTER.search(para[q.end():q.end() + 60])) \
                or _who(_BEFORE.search(para[max(0, q.start() - 60):q.start()]))
        if speaker is None and len(recent) >= 2 and not any(_who(_AFTER.search(para[q.end():])) for q in quotes):
            speaker = recent[-2]                        # alternating exchange: the other person
        speaker = speaker or unknown
        pos = 0
        for q in quotes:
            between = para[pos:q.start()].strip(" ,.;:—–-")
            if narration and between and len(re.findall(r"\w+", between)) > 3:
                out.append({"speaker": NARRATOR, "text": between, "kind": "narration"})
            tail = para[q.end():q.end() + 60]
            v = _VERB.search(tail) or _VERB.search(para[max(0, q.start() - 40):q.start()])
            out.append({"speaker": speaker, "text": (q.group(1) or q.group(2)).strip(),
                        "hint": v.group(1) if v else "", "kind": "dialogue"})
            pos = q.end()
        rest = para[pos:].strip(" ,.;:—–-")
        if narration and rest and len(re.findall(r"\w+", rest)) > 3 and not _AFTER.match(para[pos:]):
            out.append({"speaker": NARRATOR, "text": rest, "kind": "narration"})
        if speaker != unknown:
            if not recent or recent[-1] != speaker:
                recent.append(speaker)
    return out


# ---------------------------------------------------------------- transcripts
def dialogue(text, default=NARRATOR):
    out = []
    for raw in text.replace("\r\n", "\n").split("\n"):
        s = raw.strip()
        if not s:
            continue
        s = re.sub(r"^\[?\(?\d{1,2}:\d{2}(?::\d{2})?\]?\)?\s*", "", s)          # [00:01]
        m = re.match(r"^([^:]{1,40}):\s+(.*)$", s)
        speaker, body = (m.group(1).strip(), m.group(2)) if m else (default, s)
        hint = ""
        m2 = re.match(r"^\(([^)]{1,40})\)\s*(.*)$", body)   # Alice: (quietly) hello
        if m2:
            hint, body = m2.group(1), m2.group(2)
        out.append({"speaker": speaker, "text": body, "hint": hint,
                    "kind": "narration" if speaker == default and not m else "dialogue"})
    return out


FORMATS = {"screenplay": screenplay, "subtitles": subtitles, "prose": prose, "dialogue": dialogue}


def detect(path, text):
    ext = Path(path).suffix.lower()
    if ext in (".fountain", ".spmd"):
        return "screenplay"
    if ext in (".srt", ".vtt"):
        return "subtitles"
    sample = text[:20000]
    rows = [r for r in sample.split("\n") if r.strip()]
    cues = sum(1 for a, b in zip(rows, rows[1:]) if _is_cue(a, b))
    named = sum(1 for r in rows if re.match(r"^\s*(\[[^\]]*\]\s*)?[^:\n]{1,40}:\s+\S", r))
    quotes = len(_QUOTE.findall(sample))
    if rows and named / len(rows) > 0.5:
        return "dialogue"
    if rows and (cues / len(rows) > 0.08 or _SCENE.search(sample)):
        return "screenplay"
    return "prose" if quotes else "dialogue"


def convert(path, fmt="auto", **opts):
    """-> (format used, [line dicts with ids])."""
    text = Path(path).read_text(encoding="utf-8-sig")
    fmt = detect(path, text) if fmt == "auto" else fmt
    fn = FORMATS[fmt]
    allowed = fn.__code__.co_varnames[: fn.__code__.co_argcount + fn.__code__.co_kwonlyargcount]
    rows = fn(text, **{k: v for k, v in opts.items() if k in allowed and v is not None})
    width = max(4, len(str(len(rows))))
    for n, r in enumerate(rows, 1):
        r["id"] = f"{Path(path).stem}-{n:0{width}d}"
        if not r.get("hint"):
            r.pop("hint", None)
    return fmt, rows


def write_jsonl(rows, out):
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            keys = ["id", "speaker", "text"] + [k for k in r if k not in ("id", "speaker", "text")]
            fh.write(json.dumps({k: r[k] for k in keys}, ensure_ascii=False) + "\n")
