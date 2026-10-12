import json

from voicecast import config, importers, lines
from voicecast.voices import detect_mood

SCREENPLAY = """Title: The Lighthouse
Author: Someone

INT. LIGHTHOUSE - NIGHT

Rain hammers the glass. MAYA (30s) climbs the last steps.

MAYA
(out of breath)
Is anyone up here?

OLD TOM (O.S.)
Who's asking?

MAYA
It's me. Maya.
(beat)
I came back.

OLD TOM (CONT'D)
(angrily)
You shouldn't have.

CUT TO:

EXT. CLIFF - DAWN

The storm is over.
"""

SRT = """1
00:00:01,000 --> 00:00:03,500
MAYA: Is anyone up here?

2
00:00:04,000 --> 00:00:05,000
[whispering] Who's asking?

3
00:00:06,000 --> 00:00:08,000
- I came back.
"""

VTT = """WEBVTT

00:01.000 --> 00:03.000
<v Maya>Is anyone up here?</v>

00:04.000 --> 00:05.000
<v Old Tom><i>Who's asking?</i>
"""

STORY = """The rain had not stopped for three days and the lighthouse stood dark against the sky.

"Is anyone up here?" Maya asked.

"Who's asking?" a voice growled from the dark.

"It's me," whispered Maya. "I came back."

Tom said, "You shouldn't have."
"""

TRANSCRIPT = """[00:01] Maya: Is anyone up here?
[00:04] Tom: (gruffly) Who's asking?
The wind howls.
"""


def _w(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_screenplay(tmp_path):
    fmt, rows = importers.convert(_w(tmp_path, "light.fountain", SCREENPLAY))
    assert fmt == "screenplay"
    said = [(r["speaker"], r["text"], r.get("hint", "")) for r in rows if r["kind"] == "dialogue"]
    assert said == [("Maya", "Is anyone up here?", "out of breath"),
                    ("Old Tom", "Who's asking?", ""),
                    ("Maya", "It's me. Maya.", ""),
                    ("Maya", "I came back.", "beat"),
                    ("Old Tom", "You shouldn't have.", "angrily")]
    narr = [r["text"] for r in rows if r["kind"] == "narration"]
    assert narr[0].startswith("Rain hammers") and "The storm is over." in narr
    assert not any("LIGHTHOUSE" in r["text"] or "CUT TO" in r["text"] for r in rows)
    assert rows[0]["id"] == "light-0001"


def test_screenplay_detected_from_txt_and_options(tmp_path):
    fmt, rows = importers.convert(_w(tmp_path, "s.txt", SCREENPLAY), narration=False)
    assert fmt == "screenplay" and all(r["kind"] == "dialogue" for r in rows)
    _, rows = importers.convert(_w(tmp_path, "s2.txt", SCREENPLAY), scene_headings=True)
    assert any(r.get("kind") == "scene" for r in rows)


def test_subtitles(tmp_path):
    _, rows = importers.convert(_w(tmp_path, "a.srt", SRT))
    assert [(r["speaker"], r["text"]) for r in rows] == [("MAYA", "Is anyone up here?"),
                                                         ("MAYA", "Who's asking?"), ("MAYA", "I came back.")]
    assert rows[1]["hint"] == "whispering" and rows[0]["start"] == "00:00:01.000"
    _, rows = importers.convert(_w(tmp_path, "a.vtt", VTT))
    assert [(r["speaker"], r["text"]) for r in rows] == [("Maya", "Is anyone up here?"), ("Old Tom", "Who's asking?")]


def test_prose(tmp_path):
    fmt, rows = importers.convert(_w(tmp_path, "story.txt", STORY))
    assert fmt == "prose"
    said = [(r["speaker"], r["text"], r.get("hint", "")) for r in rows if r["kind"] == "dialogue"]
    assert said[0] == ("Maya", "Is anyone up here?", "")
    assert said[1][1] == "Who's asking?" and said[1][2] == "growled"
    assert said[2] == ("Maya", "It's me,", "whispered") and said[3][0] == "Maya"
    assert said[4] == ("Tom", "You shouldn't have.", "")
    assert rows[0]["kind"] == "narration"


def test_transcript(tmp_path):
    fmt, rows = importers.convert(_w(tmp_path, "t.txt", TRANSCRIPT))
    assert fmt == "dialogue"
    assert [(r["speaker"], r["text"]) for r in rows] == [("Maya", "Is anyone up here?"), ("Tom", "Who's asking?"),
                                                         ("Narrator", "The wind howls.")]
    assert rows[1]["hint"] == "gruffly"


def test_imported_lines_voiceable_with_moods(tmp_path):
    _, rows = importers.convert(_w(tmp_path, "light.fountain", SCREENPLAY))
    out = tmp_path / "light.lines.jsonl"
    importers.write_jsonl(rows, out)
    first = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert list(first)[:3] == ["id", "speaker", "text"]
    ls = lines.read(out)
    cfg = config.load(tmp_path / "none.toml")
    angry = next(ln for ln in ls if ln.hint == "angrily")
    assert detect_mood(cfg, angry)[0] == "anger"
