# voicecast

Turn screenplays, stories, subtitles and dialogue transcripts into voiced
audio. Each character gets one consistent voice, the tone follows each line,
and you choose the voices by listening to candidates side by side.

- **Imports.** It reads Fountain or plain-text screenplays, `.srt` / `.vtt`
  subtitles, prose stories and novels (quoted speech is attributed from
  "said Maya" style phrases, and the rest becomes narration), and `Name: text`
  transcripts. Screenplay parentheticals such as *(angrily)* or *(whispering)*
  set how a line is said.
- **Casting.** `cast` lists every character with how much they speak and
  which voice they get. `--add-missing` adds the uncast ones to `voices.txt`
  for you to fill in.

- **Voices described in words.** voicecast uses
  [Parler-TTS](https://github.com/huggingface/parler-tts) models. A voice is a
  sentence such as *"Mary speaks at a moderate pace with a warm, calm voice.
  Very clear audio."* The default model,
  [Indic Parler-TTS](https://huggingface.co/ai4bharat/indic-parler-tts), speaks
  English and 20 Indian languages and has named speakers that stay consistent.
  Any Parler-TTS checkpoint works.
- **Voice selection.** List candidate descriptions per speaker. `samples`
  renders a few of each speaker's lines in every candidate into one HTML page
  with the audio embedded. `pick` writes your choice into `voices.txt`.
- **Tone per line.** Whispers, anger, fear, sadness, surprise and joy are
  detected from the line, or from a hint such as the source-language line of a
  translation. The mood is worked into the voice description. The rules are
  configurable, and a line can set its mood explicitly.
- **Your own glossaries.** These are plain-text files of `term => replacement`
  rules for pronunciation, preferred wording, abbreviations and names. A rule
  can match a whole word in any script (including scripts with vowel signs),
  use a regex, replace a word stem and keep its ending, or apply only to some
  speakers or to lines matching a condition. Nothing is built in: you bring
  the rules.
- **Text rules.** Markup is stripped. Stage cues like `(sighs)` or `*laughs*`
  become sounds in the speaker's voice, silence, or spoken words. Sound
  variants can depend on the speaker or the line. Regex substitutions can be
  global, per speaker or per context, for example to fill template
  placeholders. Numbers can be said as words, and long lines are split at
  sentence ends and joined with a short pause.
- **Quality checks.** Each clip's length is checked against its text: clips
  the model cut off or that ramble into noise are regenerated, and the best
  try is kept. Silence is trimmed, and loudness can be levelled.
- **Big scripts.** Runs are resumable and in priority order (speakers first,
  per-line priority). Lines are batched by length on the GPU, and parallel
  workers can split the work with `--shard I/N`. A stop file, an `index.tsv`
  and a `failed.jsonl` are kept. Output is OGG, WAV or FLAC with a
  configurable file naming.
- **Optional voice conversion.** `convert` gives some speakers the timbre of
  a reference recording while keeping the words and pronunciation, via
  [Seed-VC](https://github.com/Plachtaa/seed-vc).

## Install

```bash
pip install -r requirements.txt     # install a CUDA build of torch first for GPU use
```

The model downloads from Hugging Face on first use, about 3.7 GB for Indic Parler-TTS.

## Quick start

From a script to audio:

```bash
python -m voicecast init myproject                  # voicecast.toml, voices.txt, glossary.txt
cd myproject
python -m voicecast import ../my_film.fountain      # -> my_film.lines.jsonl (also .srt, .vtt, stories, transcripts)
python -m voicecast cast my_film.lines.jsonl --add-missing    # every character -> voices.txt, describe each voice
python -m voicecast samples my_film.lines.jsonl     # compare candidate voices (see [[candidates.*]])
python -m voicecast run my_film.lines.jsonl         # voice everything into voices/
```

With the bundled example lines:

```bash
python -m voicecast init myproject          # example voicecast.toml, voices.txt, glossary.txt, lines
cd myproject
python -m voicecast check lines.example.jsonl            # what each line will say, in which voice
python -m voicecast samples lines.example.jsonl          # samples.html: candidates side by side
python -m voicecast pick Alice A2                        # use candidate A2 for Alice
python -m voicecast run lines.example.jsonl              # voice everything into voices/
```

## Importing scripts and stories

`voicecast import SOURCE [-o lines.jsonl] [--format auto|screenplay|subtitles|prose|dialogue]`

| source | what becomes what |
|---|---|
| screenplay (`.fountain`, or `.txt` with CHARACTER cues in capitals) | Dialogue goes under its character. Extensions such as `(V.O.)` and `(CONT'D)` are removed, and `(parentheticals)` become the line's delivery hint. Action paragraphs become Narrator lines. Scene headings, transitions, notes and the title page are skipped; `--scene-headings` voices the headings too. |
| subtitles (`.srt`, `.vtt`) | Each cue becomes a line with its `start` and `end` times. The speaker comes from `<v Name>`, a `NAME:` prefix or the previous cue. `[whispering]` becomes the hint. |
| prose (novel or story text) | Quotes go to the speaker named nearby (`"…," Maya said` or `said Maya`). Unnamed quotes in a back-and-forth alternate between the two speakers, and anything else goes to `Unknown`. Speech verbs (whispered, shouted…) become the hint. The rest is narration. Prose attribution is a heuristic, so check it with `cast`. |
| transcripts | `Name: text`, optionally with `[00:01]` timestamps and `(quietly)` hints. Lines without a name become narration. |

`--no-narration` keeps dialogue only. Line ids follow reading order
(`my_film-0001`, …), so the output files sort in story order.

## Lines

| format | content |
|---|---|
| `.jsonl` / `.json` | objects with `text` and optional `id`, `speaker`, `hint`, `mood`, `priority` (a JSON object `{id: text}` also works) |
| `.csv` / `.tsv` | a header row with the same column names |
| `.txt` | one line each; `Speaker: text` sets the speaker |

Without an `id`, a line's id is a hash of its speaker and text. Ids stay
stable across runs, and duplicate lines are voiced once.

## voices.txt

```
Alice = Mary speaks at a moderate pace with a warm, calm voice. Very clear audio, close up.
~ (guard|soldier)\b = Thoma speaks slowly with a deep, low-pitched voice. Very clear audio.
* = Mary speaks at a moderate pace with a clear, friendly voice. Very clear audio.
```

A line is an exact speaker name, `~ regex` for any matching speaker, or `*`
for everyone else. Parler-TTS descriptions work best as *"<Speaker> speaks
<pace/pitch/character>. <recording quality>."* That shape also lets voicecast
put each line's mood in the right place.

## Glossaries

List the files in `voicecast.toml` as `glossary = ["glossary.txt", "names.txt"]`.
They are applied in that order, rule by rule from top to bottom, before
anything else happens to a line.

```
# whole word / phrase, case-insensitive, any script
GIF => jif
Dr. => Doctor
# a regular expression
re: \b(\d+)km\b => \1 kilometres
# only in lines whose text (or hint) matches
lead => led @ \b(pipe|metal)\b
# replace the start of a word, keep its ending: colour, colours, coloured
colour~ => color~
# rules for some speakers only, then for everyone again
[speakers: Bob, Carol]
gonna => going to
[speakers: *]
# case-sensitive from here on ([options: nocase] switches back)
[options: case]
US => U S
```

Whole-word matching checks that the match doesn't touch another letter or
combining mark. This makes it correct for Devanagari, Kannada, Tamil, Thai
and other scripts, where a regex `\b` breaks words at vowel signs.

## voicecast.toml

`init` writes a commented example. The main sections are below; every setting has a default.

| section | what |
|---|---|
| `voices_file`, `glossary` | the voices file (default `voices.txt`) and the glossary files, in order |
| `[engine]` | `model`, `device`, `batch`, `attempts`, `seed`, `max_chars`, plausible seconds per character |
| `[text]` | `strip` (markup regexes), `cues` (stage-cue regexes), `unknown_cues` (speak / drop), `placeholders`, `numbers` (num2words language), `drop_latin`, `pause` |
| `[moods]` | `enabled`, `use_hint`, `rules` (`[{name, pattern, flags, template}]`, which replace the defaults) |
| `[output]` | `dir`, `format` (ogg / wav / flac), `name` (`{id}` `{hash}` `{speaker}` `{n}`), `gap`, `normalize` |
| `[order]` | `speakers_first` |
| `[[candidates.<Speaker>]]` | `key`, `label`, `description`: voices to compare with `samples` |
| `[[substitutions]]` | `pattern`, `replace`, optional `speakers`, `when`, `flags` |
| `[sounds]` | `"cue" = { say, tone, variants = [{ when, speakers, say, tone }] }`, or `"cue" = ""` for silence |

## Parallel workers and stopping

```bash
python -m voicecast run lines.jsonl --shard 0/2      # worker 1
python -m voicecast run lines.jsonl --shard 1/2      # worker 2 (other lines, never the same)
```

To stop, press Ctrl+C or create `voices/.stop`. Running again continues
where it left off. Lines that produced no usable audio are listed in
`voices/failed.jsonl`; run with `--redo` to retry them.

## Voice conversion (optional)

```bash
git clone https://github.com/Plachtaa/seed-vc     # then install its requirements
python -m voicecast convert --seed-vc seed-vc --reference target_voice.wav --speaker Alice
```

The originals are kept in `voices/original/`. `converted.json` records which
clips were converted, so later runs convert only new ones.

## Tests

```bash
pip install pytest
python -m pytest -q        # no model needed: a fake engine stands in for TTS
```

## License

The code is licensed under the Apache License 2.0 (see `LICENSE`). Models are not included and keep
their own licenses: Indic Parler-TTS is Apache-2.0, Seed-VC is GPL-3.0. A
reference voice used for conversion must be one you have the rights to use.
