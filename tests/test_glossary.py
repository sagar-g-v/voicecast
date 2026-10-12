import pytest

from voicecast import config, glossary, lines
from voicecast.text import prepare

RULES = """
# comment
GIF => jif
Dr. => Doctor
re: \\b(\\d+)km\\b => \\1 kilometres
lead => led @ \\b(pipe|metal)\\b
कमल => पद्म
घर~ => मकान~
[speakers: Bob]
gonna => going to
[speakers: *]
[options: case]
US => U S
"""


@pytest.fixture
def rules():
    return glossary.parse(RULES)


def test_whole_words_and_case(rules):
    assert glossary.apply(rules, "A gif, GIFs and gift.") == "A jif, GIFs and gift."
    assert glossary.apply(rules, "Ask Dr. Lee.") == "Ask Doctor Lee."
    assert glossary.apply(rules, "us and US") == "us and U S"


def test_regex_and_condition(rules):
    assert glossary.apply(rules, "Run 5km today") == "Run 5 kilometres today"
    assert glossary.apply(rules, "The lead pipe") == "The led pipe"
    assert glossary.apply(rules, "Take the lead") == "Take the lead"
    assert glossary.apply(rules, "Take the lead", context="metal work") == "Take the led"


def test_scripts_with_vowel_signs(rules):
    # कमला = कमल + a vowel sign: not the same word, must stay
    assert glossary.apply(rules, "कमल और कमला") == "पद्म और कमला"
    # stem rule keeps the ending; not inside other words
    assert glossary.apply(rules, "घरों में, घर, नघर") == "मकानों में, मकान, नघर"


def test_speaker_sections(rules):
    assert glossary.apply(rules, "I'm gonna go", "Bob") == "I'm going to go"
    assert glossary.apply(rules, "I'm gonna go", "Alice") == "I'm gonna go"


def test_bad_line():
    with pytest.raises(ValueError):
        glossary.parse("no arrow here")


def test_glossary_from_config(tmp_path):
    (tmp_path / "voicecast.toml").write_text('glossary = ["g1.txt", "g2.txt"]\n', encoding="utf-8")
    (tmp_path / "voices.txt").write_text("* = Mary speaks. Clear audio.\n", encoding="utf-8")
    (tmp_path / "g1.txt").write_text("SQL => sequel\n", encoding="utf-8")
    (tmp_path / "g2.txt").write_text("sequel => SEQUEL\n", encoding="utf-8")     # files apply in order
    cfg = config.load(tmp_path / "voicecast.toml")
    assert prepare(cfg, lines.Line("x", "Bob", "SQL rocks"))[0] == "SEQUEL rocks"
