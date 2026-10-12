"""From a line's text to what the voice says.

Order: glossary + substitutions -> markup stripped -> stage cues (sound map / speak / drop)
-> left-over placeholders dropped -> numbers to words -> punctuation tidied.
"""
import re
import unicodedata

from . import glossary
from .config import cue_key


def spoken_len(text):
    """Characters that take time to say: letters and combining marks, any script."""
    return sum(1 for ch in text if unicodedata.category(ch)[0] in "LM")


def _variant(sound, speaker, context):
    for v in sound.variants:
        if v.get("speakers") and speaker not in v["speakers"]:
            continue
        if v.get("when") and not re.search(v["when"], context, re.I):
            continue
        return v.get("say", sound.say), v.get("tone", sound.tone)
    return sound.say, sound.tone


def sound_for(cfg, cue, speaker, context):
    """(say, tone) for a cue in the sound map, else None."""
    s = cfg.sounds.get(cue_key(cue))
    return _variant(s, speaker, context) if s else None


def substitute(cfg, text, speaker, context):
    """Glossary files first, then the [[substitutions]] regexes from the config."""
    text = glossary.apply(cfg.glossary, text, speaker, context)
    for sub in cfg.substitutions:
        if sub.speakers is not None and speaker not in sub.speakers:
            continue
        if sub.when is not None and not sub.when.search(context):
            continue
        text = sub.pattern.sub(sub.replace, text)
    return text


def _num_to_words(text, lang):
    try:
        from num2words import num2words
    except ImportError:
        return text

    def one(m):
        try:
            return num2words(int(m.group().replace(",", "")), lang=lang)
        except (NotImplementedError, OverflowError, ValueError):
            return m.group()
    return re.sub(r"\d[\d,]*", one, text)


def prepare(cfg, line):
    """-> (speakable text, [tones of the line's sounds])."""
    t = cfg["text"]
    context = f"{line.text}\n{line.hint}"
    text = substitute(cfg, line.text, line.speaker, context)
    tones = []

    whole = sound_for(cfg, text.strip(), line.speaker, context)
    if whole is not None:                       # the line is only a sound ("Ahh~", "*sigh*")
        say, tone = whole
        if tone:
            tones.append(tone)
        text = say
    for rx in t["strip"]:
        text = re.sub(rx, " ", text)
    for rx in t["cues"]:
        def cue(m):
            inner = m.group(1) if m.groups() else m.group()
            hit = sound_for(cfg, inner, line.speaker, context)
            if hit is not None:
                say, tone = hit
                if tone:
                    tones.append(tone)
                return f" {say} " if say else " "
            return f" {inner} " if t["unknown_cues"] == "speak" else " "
        text = re.sub(rx, cue, text)
    if t["placeholders"]:
        text = re.sub(t["placeholders"], " ", text)
    if t["numbers"]:
        text = _num_to_words(text, t["numbers"])
    if t["drop_latin"]:
        text = re.sub(r"[A-Za-z][A-Za-z'’-]*", " ", text)
    for ch in "*~()[]{}<>":
        text = text.replace(ch, " ")
    text = re.sub(r"\.{2,}|…+", t["pause"], text)
    text = re.sub(r"([!?])[!?.]+", r"\1", text)
    text = re.sub(r"\s*,(\s*,)+", ",", text)
    text = re.sub(r"\s+([,.!?।])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip(" .,")
    return text, tones


def chunks(text, limit=180):
    """Split long text at sentence ends (TTS models are best under ~20 s)."""
    if len(text) <= limit:
        return [text]
    parts, cur = [], ""
    for piece in re.split(r"(?<=[.!?।…])\s+", text):
        if cur and len(cur) + len(piece) > limit:
            parts.append(cur)
            cur = piece
        else:
            cur = (cur + " " + piece).strip()
    if cur:
        parts.append(cur)
    return parts
