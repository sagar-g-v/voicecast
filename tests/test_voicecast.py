"""Runs without a TTS model: a fake engine makes tones whose length follows the text."""
import json

import numpy as np
import pytest

from voicecast import config, lines, run, samples
from voicecast.engine import Plausibility
from voicecast.text import chunks, prepare, spoken_len
from voicecast.voices import describe, detect_mood, with_mood

TOML = '''
[output]
dir = "out"
format = "wav"
[order]
speakers_first = ["Bob"]
[[candidates.Alice]]
key = "A1"
label = "warm"
description = "Mary speaks at a moderate pace with a warm voice. Very clear audio."
[[candidates.Alice]]
key = "A2"
label = "bright"
description = "Mary speaks fast with a bright voice. Very clear audio."
[[substitutions]]
pattern = "\\\\[name\\\\]"
replace = "friend"
speakers = ["Alice"]
[sounds]
"sigh" = { say = "haah", tone = "sighing" }
"door opens" = ""
"moan" = { say = "mm", tone = "soft", variants = [{ when = "night", say = "mmmh", tone = "breathy" }] }
'''
VOICES = '''# comment
Alice = Mary speaks at a moderate pace with a warm voice. Very clear audio.
~ ^guard = Thoma speaks slowly with a deep voice. Very clear audio.
* = Mary speaks with a clear voice. Very clear audio.
'''


@pytest.fixture
def project(tmp_path):
    (tmp_path / "voicecast.toml").write_text(TOML, encoding="utf-8")
    (tmp_path / "voices.txt").write_text(VOICES, encoding="utf-8")
    (tmp_path / "lines.jsonl").write_text("\n".join(json.dumps(x) for x in [
        {"speaker": "Alice", "text": "Hi [name]! I didn't see you there."},
        {"speaker": "Guard 2", "text": "HALT! Who goes there?!"},
        {"speaker": "Bob", "text": "(sigh) Sorry... I'm late."},
        {"speaker": "Alice", "text": "(door opens)"},
        {"speaker": "Alice", "text": "Hi [name]! I didn't see you there."},      # duplicate
        {"speaker": "Eve", "text": "We have 3 rooms.", "hint": "Original: we have three rooms, hahaha"},
    ]), encoding="utf-8")
    return tmp_path


class FakeEngine:
    rate = 8000

    def __init__(self):
        self.check = Plausibility(config.DEFAULTS["engine"])
        self.calls = 0

    def generate(self, batch, seed=None):
        self.calls += 1
        return [np.sin(np.arange(int(self.check.ideal(t) * self.rate)) / 5).astype(np.float32) * 0.3
                for _, t in batch]

    def best_of(self, desc, text, first=None, seed=0):
        return self.generate([(desc, text)])[0]


def test_config_and_voices(project):
    cfg = config.load(project / "voicecast.toml")
    assert describe(cfg, "alice").startswith("Mary")
    assert describe(cfg, "Guard 2").startswith("Thoma")
    assert describe(cfg, "Stranger").startswith("Mary speaks with a clear")
    assert [k for k, _, _ in cfg.candidates["Alice"]] == ["A1", "A2"]


def test_lines_dedupe_and_formats(project, tmp_path):
    got = lines.read(project / "lines.jsonl")
    assert len(got) == 5 and got[0].id == lines.text_hash("Alice", got[0].text)
    (tmp_path / "s.csv").write_text("id,speaker,text\nx1,Bob,Hello\n", encoding="utf-8")
    assert lines.read(tmp_path / "s.csv")[0].id == "x1"
    (tmp_path / "s.txt").write_text("Bob: Hi there\njust narration\n", encoding="utf-8")
    t = lines.read(tmp_path / "s.txt")
    assert (t[0].speaker, t[1].speaker) == ("Bob", "narrator")


def test_prepare_text(project):
    cfg = config.load(project / "voicecast.toml")
    ls = lines.read(project / "lines.jsonl")
    alice, guard, bob, door = ls[0], ls[1], ls[2], ls[3]
    assert prepare(cfg, alice)[0] == "Hi friend! I didn't see you there"
    text, tones = prepare(cfg, bob)
    assert text.startswith("haah Sorry,") and tones == ["sighing"]
    assert spoken_len(prepare(cfg, door)[0]) == 0                   # silence: nothing to voice
    assert prepare(cfg, guard)[0] == "HALT! Who goes there?"


def test_sound_variants(project):
    cfg = config.load(project / "voicecast.toml")
    day = lines.Line("a", "Alice", "(moan) yes")
    night = lines.Line("b", "Alice", "(moan) yes", hint="at night")
    assert prepare(cfg, day) == ("mm yes", ["soft"])
    assert prepare(cfg, night) == ("mmmh yes", ["breathy"])


def test_moods(project):
    cfg = config.load(project / "voicecast.toml")
    ls = lines.read(project / "lines.jsonl")
    assert detect_mood(cfg, ls[1])[0] == "anger"                    # caps + !
    assert detect_mood(cfg, ls[2])[0] == "sad"
    assert detect_mood(cfg, ls[4])[0] == "happy"                    # from the hint
    d = with_mood(cfg, describe(cfg, "Alice"), ls[2], ["sighing"])
    assert d.startswith("Mary speaks in a sad tone") and "at a moderate pace" not in d
    assert d.endswith("The speaker is sighing.")
    odd = with_mood(cfg, "A gentle voice.", ls[1])
    assert odd.endswith("The speaker sounds anger.")


def test_chunks():
    text = "One sentence here. " * 30
    parts = chunks(text.strip(), 100)
    assert len(parts) > 1 and all(len(p) <= 100 for p in parts) and " ".join(parts) == text.strip()


def test_plausibility():
    p = Plausibility(config.DEFAULTS["engine"])
    assert p.ok(1.2, "a ten letter line") and not p.ok(0.05, "a ten letter line") and not p.ok(30, "short")


def test_plan_and_run_resume_and_shards(project):
    cfg = config.load(project / "voicecast.toml")
    ls = lines.read(project / "lines.jsonl")
    jobs, skipped = run.plan(cfg, ls)
    assert jobs[0].line.speaker == "Bob" and skipped == {"nothing to say": 1}
    eng = FakeEngine()
    assert run.voice(cfg, eng, jobs, log=lambda m: None) == 4
    assert all(j.path.exists() for j in jobs)
    assert len((cfg.out_dir / "index.tsv").read_text(encoding="utf-8").splitlines()) == 4
    again, skipped = run.plan(cfg, ls)
    assert again == [] and skipped["done"] == 4
    a, _ = run.plan(cfg, ls, shard=(0, 2), redo=True)
    b, _ = run.plan(cfg, ls, shard=(1, 2), redo=True)
    assert len(a) + len(b) == 4 and not {j.line.id for j in a} & {j.line.id for j in b}


def test_samples_page_and_pick(project):
    cfg = config.load(project / "voicecast.toml")
    ls = lines.read(project / "lines.jsonl")
    out = samples.make(cfg, FakeEngine(), ls, ["Alice"], per=2, out=str(project / "s.html"), log=lambda m: None)
    page = (project / "s.html").read_text(encoding="utf-8")
    assert page.count("<audio") == 2 and "A2: bright" in page and out
    new = samples.pick(cfg, "Alice", "a2", project / "voices.txt")
    assert "bright" in new
    assert config.load(project / "voicecast.toml").voices["alice"].startswith("Mary speaks fast")
