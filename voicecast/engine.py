"""TTS engines. An engine turns (description, text) pairs into waves.

ParlerEngine: Parler-TTS family (a voice is described in words: speaker name,
pace, pitch, mood, recording quality). Default model: Indic Parler-TTS
(21 languages incl. English, many named speakers).
"""
from .audio import trim
from .text import spoken_len


class Plausibility:
    """Is a clip's length believable for its text? Catches clips the model cut
    off early (too short) and ones that ramble on with noise (too long)."""

    def __init__(self, e):
        self.e = e

    def ok(self, seconds, text):
        n, e = spoken_len(text), self.e
        low = e["min_per_char"] * n + e["min_extra"]
        high = e["max_per_char"] * n + (e["max_extra_short"] if n <= 6 else e["max_extra"])
        return low <= seconds <= high

    def ideal(self, text):
        return self.e["ideal_per_char"] * spoken_len(text)

    def usable(self, seconds, text):
        """A near miss is kept rather than nothing; a clearly broken clip isn't."""
        n = spoken_len(text)
        return 0.035 * n <= seconds <= 0.3 * n + 2.0


class ParlerEngine:
    def __init__(self, cfg):
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        e = cfg["engine"]
        self.torch, self.e = torch, e
        want = e["device"]
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if want == "auto" else want
        dtype = torch.float16 if (e["fp16"] and self.device != "cpu") else torch.float32
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(e["model"], torch_dtype=dtype).to(self.device)
        self.tok = AutoTokenizer.from_pretrained(e["model"])
        self.desc_tok = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        self.rate = self.model.config.sampling_rate
        self.frame_rate = getattr(self.model.config.audio_encoder, "frame_rate", None) or 86
        self.check = Plausibility(e)

    def generate(self, batch, seed=None):
        """[(description, text)] -> [trimmed wave]."""
        torch = self.torch
        if seed is not None:
            torch.manual_seed(seed)
        d = self.desc_tok([x[0] for x in batch], return_tensors="pt", padding=True).to(self.device)
        p = self.tok([x[1] for x in batch], return_tensors="pt", padding=True).to(self.device)
        longest = max(spoken_len(x[1]) for x in batch)
        cap = int(self.frame_rate * (0.24 * longest + 2.0)) + 10        # stops runaway noise
        with torch.no_grad():
            out = self.model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                      prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask,
                                      max_new_tokens=min(cap, 2580), return_dict_in_generate=True)
        waves = [trim(out.sequences[i, : out.audios_length[i]].float().cpu().numpy().reshape(-1), self.rate)
                 for i in range(len(batch))]
        if self.device != "cpu":
            torch.cuda.empty_cache()
        return waves

    def best_of(self, desc, text, first=None, seed=0):
        """Retry a piece whose length looks wrong; None if nothing usable."""
        tries = [first] if first is not None else []
        for k in range(self.e["attempts"]):
            w = self.generate([(desc, text)], seed=seed + k)[0]
            if self.check.ok(len(w) / self.rate, text):
                return w
            tries.append(w)
        w = min(tries, key=lambda w: abs(len(w) / self.rate - self.check.ideal(text)))
        return w if self.check.usable(len(w) / self.rate, text) else None


ENGINES = {"parler": ParlerEngine}
