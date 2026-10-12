"""Voicing a list of lines: plan, then generate.

Resumable: a line whose output file exists is skipped. Lines the model gives
no usable audio for are listed in <out>/failed.jsonl and skipped next time
(delete that file to retry them). Several workers can share one list:
--shard I/N takes the lines whose id hashes to I (mod N), so they never overlap.
"""
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from . import audio
from .text import chunks, prepare, spoken_len
from .voices import describe, with_mood


@dataclass
class Job:
    line: object
    description: str
    text: str
    pieces: list
    path: Path


def out_path(cfg, line):
    o = cfg["output"]
    h = hashlib.md5(line.id.encode("utf-8")).hexdigest()[:16]
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in line.id)
    name = o["name"].format(id=safe, hash=h, speaker="".join(c if c.isalnum() else "_" for c in line.speaker),
                            n=line.n)
    return cfg.out_dir / f"{name}.{o['format']}"


def plan(cfg, lines, *, only=None, shard=(0, 1), redo=False):
    """-> ([Job] to voice, {reason: count} skipped)."""
    failed = set()
    ff = cfg.out_dir / "failed.jsonl"
    if ff.exists() and not redo:
        for raw in ff.read_text(encoding="utf-8").splitlines():
            try:
                failed.add(json.loads(raw)["id"])
            except (ValueError, KeyError):
                pass
    first = [s.lower() for s in cfg["order"]["speakers_first"]]
    jobs, skipped = [], {}

    def skip(why):
        skipped[why] = skipped.get(why, 0) + 1

    for line in lines:
        if only and line.speaker.lower() not in only:
            continue
        if int(hashlib.md5(line.id.encode("utf-8")).hexdigest(), 16) % shard[1] != shard[0]:
            continue
        path = out_path(cfg, line)
        if path.exists() and not redo:
            skip("done")
            continue
        if line.id in failed:
            skip("failed before")
            continue
        text, tones = prepare(cfg, line)
        if not spoken_len(text):
            skip("nothing to say")
            continue
        desc = with_mood(cfg, describe(cfg, line.speaker), line, tones)
        jobs.append(Job(line, desc, text, chunks(text, cfg["engine"]["max_chars"]), path))

    def rank(j):
        s = j.line.speaker.lower()
        return (first.index(s) if s in first else len(first), -j.line.priority, j.line.n)
    jobs.sort(key=rank)
    return jobs, skipped


def voice(cfg, engine, jobs, *, log=print, stop_file=None):
    """Generate every job's audio. Returns the number of files written."""
    e, o = cfg["engine"], cfg["output"]
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    stop = Path(stop_file) if stop_file else cfg.out_dir / ".stop"
    if stop.exists():
        stop.unlink()
    # batches of similar length (less padding = fewer broken clips), sorted
    # inside small windows so the priority order holds overall
    pieces = [(j, k, t) for j in jobs for k, t in enumerate(j.pieces)]
    win = e["batch"] * 8
    batches = []
    for w in range(0, len(pieces), win):
        part = sorted(pieces[w:w + win], key=lambda x: spoken_len(x[2]))
        batches += [part[i:i + e["batch"]] for i in range(0, len(part), e["batch"])]
    got, bad, done, retried, t0 = {}, set(), 0, 0, time.time()
    index = open(cfg.out_dir / "index.tsv", "a", encoding="utf-8")
    try:
        for bi, batch in enumerate(batches):
            if stop.exists():
                log(f"stopped ({stop} exists); run again to continue")
                break
            waves = engine.generate([(j.description, t) for j, _, t in batch], seed=e["seed"] + bi)
            for (job, k, t), w in zip(batch, waves):
                if not engine.check.ok(len(w) / engine.rate, t):
                    retried += 1
                    w = engine.best_of(job.description, t, w, seed=e["seed"] + 7919 * (bi + 1))
                if w is None or len(w) < engine.rate // 10:
                    bad.add(job.line.id)
                got.setdefault(job.line.id, {})[k] = w
            for job, _, _ in batch:
                parts = got.get(job.line.id)
                if parts is None or len(parts) < len(job.pieces):
                    continue
                got.pop(job.line.id)
                if job.line.id in bad:
                    with open(cfg.out_dir / "failed.jsonl", "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"id": job.line.id, "text": job.text}, ensure_ascii=False) + "\n")
                    log(f"  no usable audio, skipped: {job.line.id}")
                    continue
                wave = audio.join([parts[k] for k in sorted(parts)], engine.rate, o["gap"])
                audio.write(job.path, audio.level(wave, o["normalize"]), engine.rate, o["format"])
                index.write(f"{job.line.id}\t{job.path.name}\t{job.line.speaker}\t{job.line.text}\n")
                done += 1
            index.flush()
            speed = done / max(time.time() - t0, 1e-6)
            left = (len(jobs) - done) / speed if speed else 0
            log(f"  {done}/{len(jobs)} lines, {retried} retried (~{left / 3600:.1f} h left)")
    except KeyboardInterrupt:
        log("interrupted; run again to continue")
    finally:
        index.close()
    return done
