"""Who sounds how: a speaker's voice description, with the line's mood worked in."""
import re


def describe(cfg, speaker):
    """voices.txt: exact name, then the first matching `~ regex`, then `*`."""
    if speaker.lower() in cfg.voices:
        return cfg.voices[speaker.lower()]
    for rx, desc in cfg.patterns:
        if rx.search(speaker):
            return desc
    if cfg.default_voice:
        return cfg.default_voice
    raise KeyError(f"no voice for speaker {speaker!r}: add it to voices.txt (or a `* = ...` default)")


def detect_mood(cfg, line):
    """(name, template) of the line's mood, or (None, None)."""
    m = cfg["moods"]
    if line.mood:
        for name, _, tpl in cfg.moods:
            if name == line.mood:
                return name, tpl
        return line.mood, None
    if not m["enabled"]:
        return None, None
    probe = line.hint if (m["use_hint"] and line.hint) else line.text
    for name, rx, tpl in cfg.moods:
        if rx.search(probe):
            return name, tpl
    if "!" in probe:
        return "animated", m["exclaim_template"]
    return None, m["plain_template"]


def with_mood(cfg, desc, line, tones=()):
    """'Anu speaks at a moderate pace with a soft voice. Very clear audio.' +
    mood 'happy' -> 'Anu speaks in a happy, cheerful tone ..., at a moderate
    pace with a soft voice. Very clear audio.' Descriptions not shaped like
    '<Name> speaks ...' get the mood appended as a sentence."""
    name, tpl = detect_mood(cfg, line)
    m = re.match(r"(\w+) speaks (.*?)\.\s*(.*)$", desc, re.S)
    if tpl and m:
        who, how, rest = m.groups()
        if re.search(r"slow|fast", tpl):                # the mood sets the pace itself
            how = re.sub(r"\b(?:at a [\w-]+ pace|very fast|fast|very slowly|slowly)\b\s*,?\s*", "", how).strip(" ,")
        desc = f"{tpl.format(name=who)}{', ' + how if how else ''}. {rest}".strip()
    elif name:
        desc = desc.rstrip() + cfg["moods"]["fallback"].format(mood=name)
    for tone in dict.fromkeys(t for t in tones if t):
        desc = desc.rstrip() + f" The speaker is {tone}."
    return desc
