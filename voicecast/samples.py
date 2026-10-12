"""Voice selection: the same lines in several candidate voices, on one page.

Candidates come from voicecast.toml:

  [[candidates.Alice]]
  key = "A1"
  label = "warm, motherly"
  description = "Divya speaks at a moderate pace with a warm, mature voice."

The page (one HTML file, audio embedded) plays every candidate for each sample
line. `voicecast pick SPEAKER KEY` then writes the chosen description into
voices.txt.
"""
import base64
import html
import io
import re
from pathlib import Path

from . import audio
from .lines import Line
from .text import prepare, spoken_len
from .voices import detect_mood, with_mood


def pick_lines(cfg, lines, speaker, per=4, lo=20, hi=140):
    """Up to `per` of the speaker's lines, of different moods where possible."""
    pool = [ln for ln in lines if ln.speaker.lower() == speaker.lower() and lo <= len(ln.text) <= hi]
    by_mood = {}
    for ln in pool:
        by_mood.setdefault(detect_mood(cfg, ln)[0] or "plain", []).append(ln)
    out = []
    while len(out) < per and any(by_mood.values()):
        for mood in list(by_mood):
            if by_mood[mood] and len(out) < per:
                out.append(by_mood[mood].pop(0))
    return out


def make(cfg, engine, lines, speakers=None, per=4, extra=None, out="samples.html", log=print):
    """Generate the samples page. extra: {speaker: [text, ...]} sample lines
    to use instead of (or when there are no) lines from the input."""
    speakers = speakers or list(cfg.candidates)
    rows = []
    for spk in speakers:
        cands = cfg.candidates.get(spk)
        if not cands:
            log(f"no candidates for {spk!r} in voicecast.toml; skipped")
            continue
        chosen = [Line(f"sample{i}", spk, t) for i, t in enumerate((extra or {}).get(spk, []))] or \
            pick_lines(cfg, lines, spk, per)
        for ln in chosen:
            text, tones = prepare(cfg, ln)
            if not spoken_len(text):
                continue
            clips = []
            for key, label, desc in cands:
                d = with_mood(cfg, desc, ln, tones)
                w = engine.best_of(d, text, seed=cfg["engine"]["seed"])
                clips.append((key, label, d, w))
            rows.append((spk, ln, text, detect_mood(cfg, ln)[0] or "plain", clips))
            log(f"  {spk}: {ln.text[:60]}")
    Path(out).write_text(page(rows, engine.rate), encoding="utf-8")
    log(f"page: {out}")
    return out


def _data_uri(wave, rate):
    buf = io.BytesIO()
    import soundfile as sf
    sf.write(buf, audio.level(wave, 0.06), rate, format="OGG", subtype="VORBIS")
    return "data:audio/ogg;base64," + base64.b64encode(buf.getvalue()).decode()


CSS = """
:root{--bg:#fbfaf8;--fg:#1f1d1a;--mut:#6b665e;--card:#fff;--line:#e6e2db;--acc:#b4532a}
@media (prefers-color-scheme:dark){:root{--bg:#191817;--fg:#ece8e1;--mut:#a39d93;--card:#232120;--line:#38342f;--acc:#e08a5f}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px;color:var(--acc)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:8px 0}
.src{color:var(--mut);font-size:13px}.say{font-size:17px;margin:2px 0 8px}
.tag{font-size:12px;border:1px solid var(--line);border-radius:99px;padding:1px 8px;margin-left:6px;color:var(--mut)}
.v{display:grid;grid-template-columns:minmax(120px,300px) 1fr;gap:4px 10px;align-items:center;margin:4px 0}
.v b{font-size:13px}audio{width:100%;height:34px}details{grid-column:2;font-size:12px;color:var(--mut)}"""


def page(rows, rate):
    out = ["<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
           f"<title>Voice samples</title><style>{CSS}</style><main><h1>Voice samples</h1>",
           "<p class='src'>Each line in every candidate voice. Pick one per speaker: "
           "<code>voicecast pick SPEAKER KEY</code></p>"]
    last = None
    for spk, ln, text, mood, clips in rows:
        if spk != last:
            out.append(f"<h2>{html.escape(spk)}</h2>")
            last = spk
        src = f"<div class='src'>{html.escape(ln.hint or ln.text)}<span class='tag'>{mood}</span></div>"
        out.append(f"<div class='card'>{src}<div class='say'>{html.escape(text)}</div>")
        for key, label, desc, w in clips:
            clip = f"<audio controls preload='none' src='{_data_uri(w, rate)}'></audio>" if w is not None \
                else "<span class='src'>(no usable audio)</span>"
            out.append(f"<div class='v'><b>{html.escape(key)}: {html.escape(label)}</b>{clip}"
                       f"<details><summary>voice prompt</summary>{html.escape(desc)}</details></div>")
        out.append("</div>")
    out.append("</main>")
    return "\n".join(out)


def pick(cfg, speaker, key, voices_file):
    """Write candidate `key`'s description as `speaker`'s voice in voices.txt."""
    cands = cfg.candidates.get(speaker) or next((v for k, v in cfg.candidates.items()
                                                 if k.lower() == speaker.lower()), None)
    if not cands:
        raise KeyError(f"no candidates for {speaker!r}")
    desc = next((d for k, _, d in cands if k.lower() == key.lower()), None)
    if desc is None:
        raise KeyError(f"{speaker!r} has no candidate {key!r} (have: {', '.join(k for k, _, _ in cands)})")
    path = Path(voices_file)
    lines = path.read_text(encoding="utf-8-sig").splitlines() if path.exists() else []
    rx = re.compile(rf"^\s*{re.escape(speaker)}\s*=", re.I)
    new = f"{speaker} = {desc}"
    for i, ln in enumerate(lines):
        if rx.match(ln):
            lines[i] = new
            break
    else:
        lines.append(new)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return new
