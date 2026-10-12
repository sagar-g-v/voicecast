"""Project configuration: voicecast.toml (settings) + voices.txt (who sounds how).

Everything has a default, so a project needs only the parts it changes.
"""
import copy
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULTS = {
    "engine": {
        "model": "ai4bharat/indic-parler-tts",   # any Parler-TTS checkpoint
        "device": "auto",                        # auto | cuda | cuda:1 | cpu
        "fp16": True,                            # half precision on the GPU
        "batch": 6,                              # lines generated together
        "attempts": 4,                           # tries for a clip whose length looks wrong
        "seed": 1000,
        "max_chars": 180,                        # longer lines are split at sentence ends
        # plausible clip length, seconds per spoken character (letters and marks)
        "min_per_char": 0.045, "max_per_char": 0.22, "ideal_per_char": 0.108,
        "min_extra": 0.15, "max_extra": 1.0, "max_extra_short": 1.6,
    },
    "text": {
        "strip": [r"\{[^}]*\}", r"<[^<>]*>"],     # markup removed before speaking
        "placeholders": r"\[[^\]]*\]",             # left-over [variables]: dropped (after substitutions)
        "cues": [r"\*([^*]{1,25})\*", r"\(([^()]{1,40})\)"],  # stage cues: mapped via sounds, else handled below
        "unknown_cues": "speak",                   # speak | drop  (cue text not in the sound map)
        "drop_latin": False,                       # drop Latin-script words the voice can't read
        "numbers": "",                             # num2words language code ("en", "kn", "hi" ...); "" = leave digits
        "pause": ", ",                             # "..." becomes this (a short pause)
    },
    "moods": {
        "enabled": True,
        "use_hint": True,                          # detect moods on the line's hint (e.g. its source text) if given
        "rules": [],                               # [{name, pattern, flags, template}] replace the defaults if set
        "exclaim_template": "{name} speaks in an animated, expressive tone",
        "plain_template": "{name} speaks in a natural conversational tone",
        "fallback": " The speaker sounds {mood}.",  # for descriptions not shaped "<Name> speaks ..."
    },
    "output": {
        "dir": "voices",
        "format": "ogg",                           # ogg | wav | flac
        "name": "{id}",                            # file name pattern: {id} {hash} {speaker} {n}
        "gap": 0.25,                               # seconds between the pieces of a split line
        "normalize": 0.0,                          # target RMS (e.g. 0.06); 0 = as generated
    },
    "order": {
        "speakers_first": [],                      # these speakers' lines first, in this order
    },
}

DEFAULT_MOODS = [
    {"name": "whisper", "pattern": r"\(whisper|whispering|\bwhispers?\b|^\s*\*whisper", "flags": "i",
     "template": "{name} whispers very softly and slowly in a hushed, breathy voice, close-sounding"},
    {"name": "anger", "pattern": r"how dare|damn you|shut up|get out|\bidiot|stupid|stop it|go away|get lost|"
                                 r"disgusting|\b[A-Z]{4,}\b.*!", "flags": "",
     "template": "{name} speaks in an angry tone, loud and sharp, expressive"},
    {"name": "fear", "pattern": r"scared|afraid|help me|don't hurt|please don't|no+,? no+|terrified", "flags": "i",
     "template": "{name} speaks in a fearful, trembling tone, slightly fast, expressive"},
    {"name": "sad", "pattern": r"\bsorry\b|\bsob|\bcry|crying|\btears?\b|\bsad\b|lonely|i miss|forgive me|\bsigh",
     "flags": "i", "template": "{name} speaks in a sad tone, slowly and softly, slightly expressive"},
    {"name": "surprise", "pattern": r"\?!|!\?|^\W*(?:what|huh|eh+|really|seriously|no way|oh my|wow|whoa)\b[^.]*[?!]",
     "flags": "i", "template": "{name} speaks in a surprised tone with a higher pitch, animated"},
    {"name": "happy", "pattern": r"\bha(?:ha)+|\bhe(?:he)+|giggle|\byay\b|\bthank|\bglad\b|\bhappy\b|wonderful|"
                                 r"love it|great|~\s*$", "flags": "i",
     "template": "{name} speaks in a happy, cheerful tone with a smile, expressive"},
]


def _merge(base, extra):
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Substitution:
    pattern: re.Pattern
    replace: str
    speakers: set = None          # None = everyone
    when: re.Pattern = None       # only lines whose text / hint matches


@dataclass
class Sound:
    say: str = ""                 # what the voice says for the cue ("" = silence)
    tone: str = ""                # added to the voice description ("moaning", "laughing" ...)
    variants: list = field(default_factory=list)   # [{when, speakers, say, tone}]


@dataclass
class Config:
    data: dict
    root: Path
    voices: dict = field(default_factory=dict)       # speaker (lower) -> description
    patterns: list = field(default_factory=list)     # [(regex, description)] for unlisted speakers
    default_voice: str = ""
    candidates: dict = field(default_factory=dict)   # speaker -> [(key, label, description)]
    glossary: list = field(default_factory=list)     # glossary.Rule, from the glossary files
    substitutions: list = field(default_factory=list)
    sounds: dict = field(default_factory=dict)       # normalised cue -> Sound
    moods: list = field(default_factory=list)        # [(name, regex, template)]

    def __getitem__(self, k):
        return self.data[k]

    @property
    def out_dir(self):
        return (self.root / self.data["output"]["dir"]).resolve()


def cue_key(s):
    """'(Lick..Lick)' and '(lick...lick)' are the same cue."""
    s = re.sub(r"\.{2,}|…", "..", s.strip().lower())
    return re.sub(r"\s+", " ", s)


def _flags(s):
    f = 0
    for ch in s or "":
        f |= {"i": re.I, "m": re.M, "s": re.S}.get(ch, 0)
    return f


def read_voices(path):
    """voices.txt: `Speaker = description` lines; `~ regex = description`
    for any speaker matching the regex; `* = description` is the default;
    `#` comments. Returns (voices, patterns, default)."""
    voices, patterns, default = {}, [], ""
    if not path or not Path(path).exists():
        return voices, patterns, default
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, desc = (s.strip() for s in line.split("=", 1))
        if name == "*":
            default = desc
        elif name.startswith("~"):
            patterns.append((re.compile(name[1:].strip(), re.I), desc))
        else:
            voices[name.lower()] = desc
    return voices, patterns, default


def load(path="voicecast.toml"):
    path = Path(path)
    raw = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data = _merge(DEFAULTS, raw)
    root = path.parent.resolve()
    cfg = Config(data, root)
    vfile = raw.get("voices_file", "voices.txt")
    cfg.voices, cfg.patterns, cfg.default_voice = read_voices(root / vfile)
    from . import glossary
    cfg.glossary = glossary.load(raw.get("glossary", []), root)
    for spk, items in (raw.get("candidates") or {}).items():
        cfg.candidates[spk] = [(c["key"], c.get("label", c["key"]), c["description"]) for c in items]
    for s in raw.get("substitutions", []):
        cfg.substitutions.append(Substitution(
            re.compile(s["pattern"], _flags(s.get("flags", ""))), s.get("replace", ""),
            set(s["speakers"]) if s.get("speakers") else None,
            re.compile(s["when"], re.I) if s.get("when") else None))
    for cue, v in (raw.get("sounds") or {}).items():
        v = v if isinstance(v, dict) else {"say": v}
        cfg.sounds[cue_key(cue)] = Sound(v.get("say", ""), v.get("tone", ""), v.get("variants", []))
    rules = data["moods"]["rules"] or DEFAULT_MOODS
    cfg.moods = [(r["name"], re.compile(r["pattern"], _flags(r.get("flags", "i"))), r["template"]) for r in rules]
    return cfg
