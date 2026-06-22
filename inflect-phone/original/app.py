from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import soundfile as sf
import torch

from inference import DEFAULT_ACOUSTIC, DEFAULT_VOCODER, load_acoustic, load_vocoder, synthesize


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACOUSTIC, SPEAKERS, ACOUSTIC_PARAMS = load_acoustic(DEFAULT_ACOUSTIC, DEVICE)
VOCODER, VOCODER_PARAMS = load_vocoder(DEFAULT_VOCODER, DEVICE)


def generate(text: str, length_scale: float, pitch_scale: float, energy_scale: float) -> str:
    text = (text or "").strip()
    if not text:
        raise gr.Error("Enter text first.")
    if len(text) > 350:
        raise gr.Error("Keep text under 350 characters for this tiny demo model.")
    audio = synthesize(
        text,
        ACOUSTIC,
        VOCODER,
        SPEAKERS,
        DEVICE,
        length_scale=length_scale,
        pitch_scale=pitch_scale,
        energy_scale=energy_scale,
    )
    path = Path(tempfile.mkdtemp()) / "inflect_nano_v1.wav"
    sf.write(str(path), audio, 24000, subtype="PCM_16")
    return str(path)


DESCRIPTION = f"""
Experimental ultra-small English TTS stack.

Inference params: {(ACOUSTIC_PARAMS + VOCODER_PARAMS) / 1_000_000:.3f}M total
({ACOUSTIC_PARAMS / 1_000_000:.3f}M acoustic + {VOCODER_PARAMS / 1_000_000:.3f}M vocoder).

This is a research/demo model, not a polished production-quality TTS system.
"""


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(
            label="Text",
            value="Wait, are you actually being for real now? I can't believe it!",
            lines=3,
        ),
        gr.Slider(0.85, 1.20, value=1.00, step=0.01, label="Length scale"),
        gr.Slider(0.85, 1.15, value=1.00, step=0.01, label="Pitch scale"),
        gr.Slider(0.85, 1.15, value=1.00, step=0.01, label="Energy scale"),
    ],
    outputs=gr.Audio(label="Generated audio", type="filepath"),
    title="Inflect-Nano-v1",
    description=DESCRIPTION,
    examples=[
        ["Please say chrysanthemum, thoroughly, proprietary, and rural without rushing through the middle syllables.", 1.0, 1.0, 1.0],
        ["No, seriously, did Jordan leave the receipt in Albuquerque, or did Priya move it to Worcester?", 1.0, 1.0, 1.0],
        ["The Wi-Fi password is Q7-Delta-9921, but please do not say the dash like a minus sign.", 1.0, 1.0, 1.0],
    ],
)


if __name__ == "__main__":
    demo.launch()
