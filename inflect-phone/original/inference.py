from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parent
VENDORED_FRONTEND = REPO_ROOT / "third_party" / "tiny_tts_frontend"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(VENDORED_FRONTEND))

from tiny_tts.nn import commons
from tiny_tts.text import phonemes_to_ids
from tiny_tts.text.english import grapheme_to_phoneme, normalize_text
from tiny_tts.utils import ADD_BLANK

from inflect_nano.text_cleaning import clean_tinytts_text
from inflect_nano.vocoder import HifiGanGenerator, make_config
from inflect_nano.acoustic import MicroFastSpeech, MicroFastSpeechConfig


DEFAULT_ACOUSTIC = REPO_ROOT / "weights" / "inflect_nano_v1_acoustic.pt"
DEFAULT_VOCODER = REPO_ROOT / "weights" / "inflect_nano_v1_vocoder.pt"


def text_to_tokens(text: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cleaned = clean_tinytts_text(text)
    normalized = normalize_text(cleaned)
    phones, tones, _ = grapheme_to_phoneme(normalized)
    phone_ids, tone_ids, lang_ids = phonemes_to_ids(phones, tones, "EN")
    if ADD_BLANK:
        phone_ids = commons.insert_blanks(phone_ids, 0)
        tone_ids = commons.insert_blanks(tone_ids, 0)
        lang_ids = commons.insert_blanks(lang_ids, 0)
    return torch.LongTensor(phone_ids), torch.LongTensor(tone_ids), torch.LongTensor(lang_ids)


def load_acoustic(path: Path, device: torch.device) -> tuple[MicroFastSpeech, dict[str, int], int]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = MicroFastSpeechConfig(**ckpt["config"])
    model = MicroFastSpeech(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    params = int(ckpt.get("params") or sum(p.numel() for p in model.parameters()))
    return model, ckpt.get("speakers") or {"mark": 0}, params


def load_vocoder(path: Path, device: torch.device) -> tuple[HifiGanGenerator, int]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = make_config((ckpt.get("config") or {}).get("variant", "snake_v2mid"))
    model = HifiGanGenerator(cfg).to(device)
    model.load_state_dict(ckpt["generator"])
    model.remove_weight_norm()
    model.eval()
    params = int(ckpt.get("generator_params") or sum(p.numel() for p in model.parameters()))
    return model, params


def rms_db(audio: np.ndarray) -> float:
    return 20.0 * math.log10(float(np.sqrt(np.mean(audio**2, dtype=np.float64))) + 1e-9)


def normalize_audio(audio: np.ndarray, target_rms_db: float = -20.0, peak_db: float = -1.0) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        audio = np.zeros(1, dtype=np.float32)
    audio = audio - float(audio.mean())
    audio *= 10 ** ((target_rms_db - rms_db(audio)) / 20.0)
    peak = float(np.max(np.abs(audio)) + 1e-9)
    peak_limit = 10 ** (peak_db / 20.0)
    if peak > peak_limit:
        audio *= peak_limit / peak
    return np.clip(audio, -1.0, 1.0)


@torch.inference_mode()
def synthesize(
    text: str,
    acoustic: MicroFastSpeech,
    vocoder: HifiGanGenerator,
    speakers: dict[str, int],
    device: torch.device,
    length_scale: float = 1.0,
    pitch_scale: float = 1.0,
    energy_scale: float = 1.0,
) -> np.ndarray:
    phone, tone, lang = text_to_tokens(text)
    phone = phone.unsqueeze(0).to(device)
    tone = tone.unsqueeze(0).to(device)
    lang = lang.unsqueeze(0).to(device)
    speaker = torch.LongTensor([int(speakers.get("mark", next(iter(speakers.values()), 0)))]).to(device)
    mel = acoustic.infer(
        phone,
        tone,
        lang,
        speaker,
        length_scale=float(length_scale),
        pitch_scale=float(pitch_scale),
        energy_scale=float(energy_scale),
    )
    wav = vocoder(mel).squeeze().detach().cpu().numpy()
    return normalize_audio(wav)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Inflect-Nano-v1 text-to-speech.")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", type=Path, default=Path("inflect_nano_v1_output.wav"))
    ap.add_argument("--acoustic", type=Path, default=DEFAULT_ACOUSTIC)
    ap.add_argument("--vocoder", type=Path, default=DEFAULT_VOCODER)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--length-scale", type=float, default=1.0)
    ap.add_argument("--pitch-scale", type=float, default=1.0)
    ap.add_argument("--energy-scale", type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device(args.device)
    acoustic, speakers, acoustic_params = load_acoustic(args.acoustic, device)
    vocoder, vocoder_params = load_vocoder(args.vocoder, device)
    audio = synthesize(
        args.text,
        acoustic,
        vocoder,
        speakers,
        device,
        length_scale=args.length_scale,
        pitch_scale=args.pitch_scale,
        energy_scale=args.energy_scale,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.out), audio, 24000, subtype="PCM_16")
    print(f"Wrote {args.out}")
    print(f"Params: acoustic={acoustic_params:,} vocoder={vocoder_params:,} total={acoustic_params + vocoder_params:,}")


if __name__ == "__main__":
    main()
