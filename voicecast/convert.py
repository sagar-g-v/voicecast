"""Optional voice conversion: give finished clips another voice's timbre while
keeping the words and pronunciation (Seed-VC, zero-shot, from a reference clip).

Seed-VC is not bundled: clone https://github.com/Plachtaa/seed-vc, install its
requirements, and pass its folder with --seed-vc. The reference is any clean
10-25 s recording of the target voice (WAV/FLAC/OGG/MP3).

Converted clips are written over the originals after the originals are moved
to <out>/original/ (restore by moving them back); converted.json records
which files got which reference, so re-runs convert only new clips.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from . import audio


class SeedVC:
    def __init__(self, seed_dir, device="auto", steps=25, cfg_rate=0.7):
        import torch
        seed_dir = Path(seed_dir).resolve()
        sys.path.insert(0, str(seed_dir))
        here = os.getcwd()
        os.chdir(seed_dir)                       # Seed-VC reads its configs relative to itself
        try:
            import inference as sv
            dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
            sv.device = torch.device(dev)
            sv.fp16 = dev.startswith("cuda")
            args = argparse.Namespace(f0_condition=False, checkpoint=None, config=None, fp16=sv.fp16)
            (self.model, self.semantic_fn, _, self.vocoder, self.campplus,
             self.to_mel, mel_args) = sv.load_models(args)
        finally:
            os.chdir(here)
        self.sr = mel_args["sampling_rate"]
        self.sv, self.torch, self.device = sv, torch, sv.device
        self.steps, self.cfg_rate = steps, cfg_rate
        self.refs = {}

    def _ref(self, path):
        if path not in self.refs:
            import librosa
            import torchaudio as ta
            torch = self.torch
            wav = librosa.load(path, sr=self.sr)[0][: self.sr * 25]
            wav = torch.tensor(wav).unsqueeze(0).float().to(self.device)
            w16 = ta.functional.resample(wav, self.sr, 16000)
            mel2 = self.to_mel(wav)
            feat = ta.compliance.kaldi.fbank(w16, num_mel_bins=80, dither=0, sample_frequency=16000)
            style = self.campplus((feat - feat.mean(dim=0, keepdim=True)).unsqueeze(0))
            prompt = self.model.length_regulator(self.semantic_fn(w16), n_quantizers=3, f0=None,
                                                 ylens=torch.LongTensor([mel2.size(2)]).to(self.device))[0]
            self.refs[path] = (mel2, style, prompt)
        return self.refs[path]

    def convert(self, src, ref):
        """Any audio file -> numpy wave in the reference voice, at the source's loudness."""
        import librosa
        import torchaudio as ta
        torch = self.torch
        mel2, style, prompt = self._ref(str(ref))
        wav = torch.tensor(librosa.load(str(src), sr=self.sr)[0]).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            s_alt = self.semantic_fn(ta.functional.resample(wav, self.sr, 16000))
            mel = self.to_mel(wav)
            cond = self.model.length_regulator(s_alt, ylens=torch.LongTensor([mel.size(2)]).to(self.device),
                                               n_quantizers=3, f0=None)[0]
            cat = torch.cat([prompt, cond], dim=1)
            with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda",
                                dtype=torch.float16 if self.sv.fp16 else torch.float32):
                out = self.model.cfm.inference(cat, torch.LongTensor([cat.size(1)]).to(self.device),
                                               mel2, style, None, self.steps, inference_cfg_rate=self.cfg_rate)
            out = self.vocoder(out[:, :, mel2.size(-1):].float()).squeeze().cpu().numpy()
        src_rms = float(wav.pow(2).mean().sqrt())
        out_rms = float((out ** 2).mean() ** 0.5)
        return (out * (src_rms / out_rms) if out_rms > 1e-6 else out).clip(-1, 1)


def run(cfg, seed_dir, reference, speakers=None, device="auto", log=print):
    """Convert every voiced clip of `speakers` (all if None) not converted yet."""
    out = cfg.out_dir
    index = out / "index.tsv"
    if not index.exists():
        log("nothing voiced yet")
        return 0
    want = {s.lower() for s in speakers} if speakers else None
    done_file = out / "converted.json"
    done = json.loads(done_file.read_text(encoding="utf-8")) if done_file.exists() else {}
    files = []
    for raw in index.read_text(encoding="utf-8").splitlines():
        parts = raw.split("\t")
        if len(parts) >= 3 and (want is None or parts[2].lower() in want) and (out / parts[1]).exists() \
                and done.get(parts[1]) != str(reference):
            files.append(parts[1])
    files = list(dict.fromkeys(files))
    log(f"{len(files)} clip(s) to convert")
    if not files:
        return 0
    vc = SeedVC(seed_dir, device)
    keep = out / "original"
    keep.mkdir(exist_ok=True)
    for i, name in enumerate(files, 1):
        src = out / name
        orig = keep / name
        if not orig.exists():
            shutil.copy2(src, orig)
        wave = vc.convert(orig, reference)
        audio.write(src, wave, vc.sr, cfg["output"]["format"])
        done[name] = str(reference)
        if i % 20 == 0 or i == len(files):
            done_file.write_text(json.dumps(done, indent=0), encoding="utf-8")
            log(f"  {i}/{len(files)}")
    return len(files)
