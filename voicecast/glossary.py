"""Glossaries: plain-text replacement rules applied to every line before it
is spoken. Use them for pronunciation fixes, preferred terms or wording,
expanding abbreviations, transliterating names, or any word-level change.

File format (UTF-8, one rule per line, applied top to bottom):

  # comment
  term => replacement                 whole word / phrase, any script
  term => replacement @ condition     only in lines whose text or hint matches the regex
  re: pattern => replacement          a regular expression (\\1 back-references)
  stem~ => newstem~                   the start of a word, keeping its ending
  [speakers: Alice, Bob]              the rules below apply to these speakers only
  [speakers: *]                       ... and from here on to everyone again
  [options: case]                     the rules below match case-sensitively
  [options: nocase]                   ... or not (the default)

"Whole word" means the match is not touching another letter or mark on
either side, which works for scripts with combining vowel signs too (where
regex \\b does not).
"""
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


def _wordish(ch):
    return bool(ch) and unicodedata.category(ch)[0] in "LMN"


@dataclass
class Rule:
    pattern: re.Pattern
    replace: str
    whole: bool                 # whole-word check on both sides
    stem: bool                  # stem~ rule: whole at the start only
    speakers: frozenset = None  # None = everyone
    when: re.Pattern = None
    source: str = ""            # "file:line" for messages

    def apply(self, text):
        if not (self.whole or self.stem):
            return self.pattern.sub(self.replace, text)

        def one(m):
            before = text[m.start() - 1] if m.start() else ""
            after = text[m.end()] if m.end() < len(text) else ""
            if _wordish(before) or (self.whole and _wordish(after)):
                return m.group()
            return self.replace
        return self.pattern.sub(one, text)


def parse(text, source="glossary"):
    rules, speakers, flags = [], None, re.I
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"\[\s*(speakers|options)\s*:\s*(.*?)\s*\]", line, re.I)
        if m:
            if m.group(1).lower() == "speakers":
                names = [s.strip() for s in m.group(2).split(",") if s.strip()]
                speakers = None if names in ([], ["*"]) else frozenset(s.lower() for s in names)
            else:
                opts = {o.strip().lower() for o in m.group(2).split(",")}
                flags = 0 if "case" in opts else re.I
            continue
        if "=>" not in line:
            raise ValueError(f"{source}:{n}: expected 'term => replacement'")
        left, right = (s.strip() for s in line.split("=>", 1))
        when = None
        m = re.match(r"(.*?)\s+@\s+(.+)$", right)
        if m:
            right, when = m.group(1).strip(), re.compile(m.group(2), re.I)
        where = f"{source}:{n}"
        if left.lower().startswith("re:"):
            rules.append(Rule(re.compile(left[3:].strip(), flags), right, False, False, speakers, when, where))
        elif left.endswith("~") and right.endswith("~"):
            # only the stem is replaced; the rest of the word stays as it is
            rules.append(Rule(re.compile(re.escape(left[:-1]), flags), right[:-1], False, True,
                              speakers, when, where))
        else:
            rules.append(Rule(re.compile(re.escape(left), flags), right, True, False, speakers, when, where))
    return rules


def load(paths, root="."):
    rules = []
    for p in ([paths] if isinstance(paths, str) else paths or []):
        path = Path(root) / p
        if not path.exists():
            raise FileNotFoundError(f"glossary not found: {path}")
        rules += parse(path.read_text(encoding="utf-8-sig"), path.name)
    return rules


def apply(rules, text, speaker="", context=""):
    spk = speaker.lower()
    for r in rules:
        if r.speakers is not None and spk not in r.speakers:
            continue
        if r.when is not None and not r.when.search(context or text):
            continue
        text = r.apply(text)
    return text
