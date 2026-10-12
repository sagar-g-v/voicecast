"""voicecast: voice a script of lines with TTS, one voice per speaker.

  init      [DIR]               write an example project (voicecast.toml, voices.txt, glossary.txt)
  import    SCRIPT              screenplay / subtitles / story / transcript -> a lines file
  cast      LINES               the speakers, how much each says, which voice each gets
  check     LINES               show what each line will say, in which voice (no model)
  samples   LINES               candidate voices side by side on one page, to choose from
  pick      SPEAKER KEY         use candidate KEY as SPEAKER's voice (writes voices.txt)
  run       LINES               voice every line (resumable; --shard for parallel workers)
  convert   --seed-vc DIR --reference CLIP   optional: another voice's timbre for some speakers

All commands read ./voicecast.toml (or --config PATH).
"""
import argparse
import sys
from pathlib import Path

from . import config as config_mod
from . import lines as lines_mod
from .run import plan, voice

EXAMPLE_TOML = '''# voicecast project settings. Everything is optional: these are the defaults
# unless noted. Paths are relative to this file.

voices_file = "voices.txt"
glossary = ["glossary.txt"]            # word / phrase rules, applied in order (see glossary.txt)

[engine]
model = "ai4bharat/indic-parler-tts"   # any Parler-TTS checkpoint, e.g. "parler-tts/parler-tts-mini-v1"
device = "auto"                        # auto | cuda | cpu
batch = 6                              # lower if the GPU runs out of memory
attempts = 4                           # retries for clips whose length looks wrong
max_chars = 180                        # longer lines are split at sentence ends

[text]
numbers = ""                           # e.g. "en" to say numbers as words (needs num2words)
unknown_cues = "speak"                 # "(laughs)", "*sigh*" not in [sounds]: speak | drop

[moods]
enabled = true                         # tone follows the line: happy, sad, angry, whisper ...

[output]
dir = "voices"
format = "ogg"                         # ogg | wav | flac
name = "{id}"                          # {id} {hash} {speaker} {n}
normalize = 0.0                        # target loudness (RMS), e.g. 0.06; 0 = as generated

[order]
speakers_first = []                    # e.g. ["Alice", "Bob"]

# --- voice selection: candidates per speaker for `voicecast samples` -------
[[candidates.Alice]]
key = "A1"
label = "warm, calm"
description = "Mary speaks at a moderate pace with a warm, calm and slightly low-pitched voice. Very clear audio, close up."

[[candidates.Alice]]
key = "A2"
label = "bright, lively"
description = "Mary speaks fast with a bright, cheerful and slightly high-pitched voice. Very clear audio, close up."

# --- text rules ---------------------------------------------------------------
# Regex replacements, applied after the glossary. Limit them to some
# speakers, or to lines whose text matches `when`.
[[substitutions]]
pattern = "\\\\[name\\\\]"
replace = "my friend"
speakers = ["Alice"]

# What stage cues and sound-only lines become ("" = silence). `tone` is added
# to the voice description. Variants apply to some speakers / lines only.
[sounds]
"sigh" = { say = "haah", tone = "sighing" }
"laughs" = { say = "haha", tone = "laughing" }
"door opens" = ""
'''

EXAMPLE_VOICES = '''# Who sounds how. Parler-TTS takes a description in plain English: speaker
# name (the model's named voices keep a speaker consistent), pace, pitch,
# character, recording quality. Moods are worked in per line automatically.
#
#   Speaker = description          an exact speaker name (case-insensitive)
#   ~ regex = description          any speaker whose name matches
#   * = description                everyone else

Alice = Mary speaks at a moderate pace with a warm, calm and slightly low-pitched voice. Very clear audio, close up.
Bob = Thoma speaks at a moderate pace with a confident, casual young male voice. Very clear audio, close up.
Narrator = Thoma speaks in a calm narration tone at a moderate pace. Very clear audio.
~ (guard|soldier|man)\\b = Thoma speaks slowly with a deep, low-pitched male voice. Very clear audio.
* = Mary speaks at a moderate pace with a clear, friendly voice. Very clear audio.
'''

EXAMPLE_GLOSSARY = '''# Glossary: word / phrase rules applied to every line before it is spoken,
# top to bottom. Add as many files as you like in voicecast.toml (glossary = [...]).
#
#   term => replacement                 whole word / phrase (any script), case-insensitive
#   term => replacement @ regex         only in lines whose text (or hint) matches
#   re: pattern => replacement          a regular expression
#   stem~ => newstem~                   the start of a word, keeping its ending
#   [speakers: Alice, Bob]              rules below: these speakers only ([speakers: *] = everyone)
#   [options: case]                     rules below: case-sensitive ([options: nocase] = back to default)

# pronunciation
GIF => jif
SQL => sequel
Nguyen => Win

# abbreviations
Dr. => Doctor
approx. => approximately
re: \\bSt\\.(?= [A-Z]) => Saint

# a word only in some contexts
lead => led @ \\b(pipe|metal|poison)\\b

[speakers: Bob]
gonna => going to
'''

EXAMPLE_LINES = '''{"speaker": "Alice", "text": "Hi [name]! I didn't see you there."}
{"speaker": "Alice", "text": "Oh, hi! I didn't see you there."}
{"speaker": "Bob", "text": "Sorry... I'm gonna ask Dr. Lee about the SQL thing."}
{"speaker": "Alice", "text": "(laughs) It's fine. Really, it's fine!"}
{"speaker": "Narrator", "text": "The door creaked open, and the room went quiet."}
{"speaker": "Alice", "text": "*sigh* We have 3 hours left. Let's go.", "mood": "sad"}
'''


def cmd_init(args):
    d = Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)
    for name, body in (("voicecast.toml", EXAMPLE_TOML), ("voices.txt", EXAMPLE_VOICES),
                       ("glossary.txt", EXAMPLE_GLOSSARY), ("lines.example.jsonl", EXAMPLE_LINES)):
        p = d / name
        if p.exists():
            print(f"exists, left alone: {p}")
        else:
            p.write_text(body, encoding="utf-8")
            print(f"wrote {p}")


def _load(args):
    cfg = config_mod.load(args.config)
    lines = lines_mod.read(args.lines, default_speaker=args.default_speaker) if getattr(args, "lines", None) else []
    return cfg, lines


def _engine(cfg):
    from .engine import ENGINES
    return ENGINES["parler"](cfg)


def cmd_check(args):
    cfg, lines = _load(args)
    only = {s.lower() for s in args.speaker} if args.speaker else None
    jobs, skipped = plan(cfg, lines, only=only, redo=args.redo)
    for j in jobs[: args.limit or None]:
        print(f"[{j.line.speaker}] {j.line.text}\n   says : {j.text}\n   voice: {j.description}\n")
    print(f"{len(jobs)} line(s) to voice; skipped: {skipped or 'none'}")


def cmd_run(args):
    cfg, lines = _load(args)
    if args.batch:
        cfg.data["engine"]["batch"] = args.batch
    if args.device:
        cfg.data["engine"]["device"] = args.device
    only = {s.lower() for s in args.speaker} if args.speaker else None
    i, n = (int(x) for x in args.shard.split("/"))
    jobs, skipped = plan(cfg, lines, only=only, shard=(i, n), redo=args.redo)
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} line(s) to voice -> {cfg.out_dir}  (skipped: {skipped or 'none'})", flush=True)
    if not jobs:
        return
    engine = _engine(cfg)
    print(f"model {cfg['engine']['model']} on {engine.device}", flush=True)
    done = voice(cfg, engine, jobs, log=lambda m: print(m, flush=True), stop_file=args.stop_file)
    print(f"done: {done} file(s) written", flush=True)


def cmd_samples(args):
    from . import samples
    cfg, lines = _load(args)
    extra = {}
    for item in args.text or []:
        spk, _, txt = item.partition(":")
        extra.setdefault(spk.strip(), []).append(txt.strip())
    if not cfg.candidates:
        sys.exit("no [[candidates.<speaker>]] in the config: add 2+ descriptions per speaker to compare")
    samples.make(cfg, _engine(cfg), lines, args.speaker or None, args.per, extra, args.out,
                 log=lambda m: print(m, flush=True))


def cmd_pick(args):
    from . import samples
    cfg = config_mod.load(args.config)
    raw = config_mod.tomllib.loads(Path(args.config).read_text(encoding="utf-8")) if Path(args.config).exists() else {}
    vfile = Path(args.config).parent / raw.get("voices_file", "voices.txt")
    print(samples.pick(cfg, args.speaker, args.key, vfile))


def cmd_import(args):
    from . import importers
    opts = {"narration": not args.no_narration, "scene_headings": args.scene_headings}
    fmt, rows = importers.convert(args.source, args.format, **opts)
    out = args.out or str(Path(args.source).with_suffix(".lines.jsonl"))
    importers.write_jsonl(rows, out)
    speakers = {}
    for r in rows:
        speakers[r["speaker"]] = speakers.get(r["speaker"], 0) + 1
    print(f"{fmt}: {len(rows)} line(s), {len(speakers)} speaker(s) -> {out}")
    for s, n in sorted(speakers.items(), key=lambda x: -x[1])[:15]:
        print(f"  {n:6d}  {s}")


def cmd_cast(args):
    from .voices import describe
    cfg, lines = _load(args)
    counts, chars = {}, {}
    for ln in lines:
        counts[ln.speaker] = counts.get(ln.speaker, 0) + 1
        chars[ln.speaker] = chars.get(ln.speaker, 0) + len(ln.text)
    missing = []
    for spk in sorted(counts, key=lambda s: -chars[s]):
        if spk.lower() in cfg.voices:
            how = "own voice"
        elif any(rx.search(spk) for rx, _ in cfg.patterns):
            how = "pattern"
        elif cfg.default_voice:
            how = "default (*)"
            missing.append(spk)
        else:
            how = "NO VOICE"
            missing.append(spk)
        print(f"{counts[spk]:6d} lines {chars[spk]:8d} chars  {how:12}  {spk}")
    if args.add_missing and missing:
        raw = config_mod.tomllib.loads(Path(args.config).read_text(encoding="utf-8")) if Path(args.config).exists() else {}
        vfile = Path(args.config).parent / raw.get("voices_file", "voices.txt")
        stub = cfg.default_voice or "Mary speaks at a moderate pace with a clear voice. Very clear audio."
        with open(vfile, "a", encoding="utf-8") as fh:
            fh.write("\n# added by `voicecast cast --add-missing`: give each one its own description\n")
            for spk in missing:
                fh.write(f"{spk} = {stub}\n")
        print(f"added {len(missing)} speaker(s) to {vfile}")


def cmd_convert(args):
    from . import convert
    cfg = config_mod.load(args.config)
    n = convert.run(cfg, args.seed_vc, args.reference, args.speaker or None, args.device,
                    log=lambda m: print(m, flush=True))
    print(f"done: {n} clip(s) converted")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="voicecast", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="voicecast.toml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="write an example project")
    p.add_argument("dir", nargs="?", default=".")
    p.set_defaults(func=cmd_init)

    def lines_args(p):
        p.add_argument("lines", help="lines file: .jsonl / .json / .csv / .tsv / .txt")
        p.add_argument("--default-speaker", default="narrator", help="for lines without a speaker")
        p.add_argument("--speaker", action="append", help="only this speaker (repeatable)")

    p = sub.add_parser("import", help="a screenplay / subtitles / story / transcript -> a lines file")
    p.add_argument("source", help=".fountain / screenplay .txt, .srt / .vtt, a story (.txt / .md), a transcript")
    p.add_argument("-o", "--out", default=None, help="output .jsonl (default: <source>.lines.jsonl)")
    p.add_argument("--format", default="auto", choices=["auto", "screenplay", "subtitles", "prose", "dialogue"])
    p.add_argument("--no-narration", action="store_true", help="dialogue only (no action / narration lines)")
    p.add_argument("--scene-headings", action="store_true", help="screenplays: also voice the scene headings")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("cast", help="the speakers in a lines file and which voice each gets")
    lines_args(p)
    p.add_argument("--add-missing", action="store_true",
                   help="add speakers without their own voice to voices.txt (to fill in)")
    p.set_defaults(func=cmd_cast)

    p = sub.add_parser("check", help="dry run: text and voice per line, no model")
    lines_args(p)
    p.add_argument("--limit", type=int, default=20, help="lines to show (0 = all)")
    p.add_argument("--redo", action="store_true")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("run", help="voice the lines")
    lines_args(p)
    p.add_argument("--shard", default="0/1", help="I/N: this worker's share, for N parallel workers")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch", type=int, default=0, help="override [engine] batch")
    p.add_argument("--device", default="", help="override [engine] device")
    p.add_argument("--redo", action="store_true", help="voice again lines that already have audio / failed")
    p.add_argument("--stop-file", default=None, help="stop when this file appears (default: <out>/.stop)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("samples", help="candidate voices side by side (HTML page)")
    p.add_argument("lines", nargs="?", default=None, help="lines to take samples from (optional with --text)")
    p.add_argument("--default-speaker", default="narrator")
    p.add_argument("--speaker", action="append", help="only these speakers (default: all with candidates)")
    p.add_argument("--per", type=int, default=4, help="sample lines per speaker")
    p.add_argument("--text", action="append", help='"Speaker: a sample line" (repeatable)')
    p.add_argument("-o", "--out", default="samples.html")
    p.set_defaults(func=cmd_samples)

    p = sub.add_parser("pick", help="choose a candidate as a speaker's voice")
    p.add_argument("speaker")
    p.add_argument("key")
    p.set_defaults(func=cmd_pick)

    p = sub.add_parser("convert", help="optional: give some speakers another voice's timbre (Seed-VC)")
    p.add_argument("--seed-vc", required=True, help="a Seed-VC checkout")
    p.add_argument("--reference", required=True, help="10-25 s clip of the target voice")
    p.add_argument("--speaker", action="append", help="only these speakers (default: all voiced clips)")
    p.add_argument("--device", default="auto")
    p.set_defaults(func=cmd_convert)

    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
