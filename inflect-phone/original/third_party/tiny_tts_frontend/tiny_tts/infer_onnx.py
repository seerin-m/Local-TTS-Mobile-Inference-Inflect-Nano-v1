"""
ONNX Runtime inference engine for TinyTTS.

Replaces the PyTorch VoiceSynthesizer.infer() with equivalent
ONNX Runtime sessions + NumPy ops for the non-exported parts
(alignment path computation).
"""
import os
import numpy as np
import soundfile as sf

from tiny_tts.text.english import normalize_text, grapheme_to_phoneme
from tiny_tts.text import phonemes_to_ids
from tiny_tts.nn import commons
from tiny_tts.utils.config import (
    SAMPLING_RATE, ADD_BLANK, SPK2ID,
)

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("onnxruntime is required. Run: pip install onnxruntime")


def _build_session(path: str, use_gpu: bool = False):
    """Create an ORT InferenceSession with optional GPU support."""
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_gpu else
        ["CPUExecutionProvider"]
    )
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = os.cpu_count() or 4
    return ort.InferenceSession(path, sess_options=opts, providers=providers)


def _create_length_mask_np(lengths, max_len=None):
    """NumPy equivalent of commons.create_length_mask."""
    if max_len is None:
        max_len = int(lengths.max())
    ids = np.arange(max_len, dtype=np.float32)     # [T]
    mask = (ids[None, :] < lengths[:, None]).astype(np.float32)  # [B, T]
    return mask


def _compute_alignment_path_np(w_ceil, attn_mask):
    """
    Monotonic alignment path - vectorized via cumsum (much faster than Python loops).
    w_ceil:    [B, 1, T_x] — integer duration per phone
    attn_mask: [B, 1, T_y, T_x] — joint mask
    Returns attn: [B, 1, T_y, T_x]
    """
    B, _, T_x = w_ceil.shape
    T_y = attn_mask.shape[2]

    # Build duration matrix: for each phone column expand the duration
    # cumulative sum of durations gives us the end frame index for each phone
    dur = w_ceil[:, 0, :]                            # [B, T_x]
    cum_dur = np.cumsum(dur, axis=1)                 # [B, T_x]  — end frame (1-indexed)
    cum_dur_prev = np.pad(cum_dur[:, :-1], ((0,0),(1,0)))  # [B, T_x] — start frame

    # Frame indices: [1, T_y, 1]
    frame_idx = np.arange(T_y, dtype=np.float32)[None, :, None]   # [1, T_y, 1]
    # For each phone, mark frames [start, end)
    # cum_dur_prev: [B,1,T_x], cum_dur: [B,1,T_x]
    start = cum_dur_prev[:, None, :]                 # [B, 1, T_x]
    end   = cum_dur[:, None, :]                      # [B, 1, T_x]
    attn  = ((frame_idx >= start) & (frame_idx < end)).astype(np.float32)  # [B, T_y, T_x]
    attn  = attn[:, None, :, :]                      # [B, 1, T_y, T_x]
    return attn * attn_mask


class OnnxTinyTTS:
    """
    Inference using ONNX Runtime.

    Args:
        onnx_dir: directory containing the 4 .onnx files
        use_gpu:  if True, try CUDAExecutionProvider
    """

    def __init__(self, onnx_dir: str = "onnx", use_gpu: bool = False):
        onnx_dir = os.path.abspath(onnx_dir)
        print(f"Loading ONNX sessions from: {onnx_dir}")

        self._enc  = _build_session(os.path.join(onnx_dir, "text_encoder.onnx"), use_gpu)
        self._dp   = _build_session(os.path.join(onnx_dir, "duration_predictor.onnx"), use_gpu)
        self._flow = _build_session(os.path.join(onnx_dir, "flow.onnx"), use_gpu)
        self._dec  = _build_session(os.path.join(onnx_dir, "decoder.onnx"), use_gpu)

        print("ONNX sessions ready ✅")

    def _text_to_ids(self, text: str):
        normalized = normalize_text(text)
        phones, tones, _ = grapheme_to_phoneme(normalized)
        phone_ids, tone_ids, lang_ids = phonemes_to_ids(phones, tones, "EN")

        if ADD_BLANK:
            phone_ids = commons.insert_blanks(phone_ids, 0)
            tone_ids  = commons.insert_blanks(tone_ids, 0)
            lang_ids  = commons.insert_blanks(lang_ids, 0)

        return phone_ids, tone_ids, lang_ids

    def speak(
        self,
        text: str,
        output_path: str = "onnx_output.wav",
        speaker: str = "female",
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
        length_scale: float = 1.0,
        output_sr: int = None,
    ) -> np.ndarray:
        """Synthesize speech and save to output_path.
        
        Args:
            output_sr: If set (e.g. 22050), resample the output from 44100 Hz.
                       Useful to reduce file size while keeping quality.
        """
        print(f"[ONNX] Synthesizing: {text}")

        phone_ids, tone_ids, lang_ids = self._text_to_ids(text)
        T = len(phone_ids)

        # Prepare inputs as float32 / int64 arrays
        x       = np.array(phone_ids, dtype=np.int64)[None, :]        # [1, T]
        x_len   = np.array([T], dtype=np.int64)                        # [1]
        tone    = np.array(tone_ids, dtype=np.int64)[None, :]          # [1, T]
        lang    = np.array(lang_ids, dtype=np.int64)[None, :]          # [1, T]
        bert    = np.zeros((1, 1024, T), dtype=np.float32)
        ja_bert = np.zeros((1, 768,  T), dtype=np.float32)
        sid_val = SPK2ID.get(speaker, 0)
        sid     = np.array([sid_val], dtype=np.int64)                  # [1]

        # ── 1. Text Encoder ──────────────────────────────────────────────
        x_enc, m_p, logs_p, x_mask, g = self._enc.run(
            None,
            {
                "phone_ids":    x,
                "phone_lengths":x_len,
                "tone_ids":     tone,
                "language_ids": lang,
                "bert":         bert,
                "ja_bert":      ja_bert,
                "speaker_id":   sid,
            },
        )

        # ── 2. Duration Predictor ─────────────────────────────────────────
        logw = self._dp.run(None, {"x": x_enc, "x_mask": x_mask, "g": g})[0]

        # ── 3. Alignment Path (NumPy) ─────────────────────────────────────
        w        = np.exp(logw) * x_mask * length_scale          # [1, 1, T]
        w_ceil   = np.ceil(w)                                    # [1, 1, T]
        y_len    = max(1, int(w_ceil.sum()))
        y_lens   = np.array([y_len], dtype=np.int64)

        y_mask   = _create_length_mask_np(y_lens, y_len)         # [1, T_y]
        y_mask   = y_mask[:, None, :]                            # [1, 1, T_y]
        # attn_mask: [1, 1, T_y, T_x]  (outer product of frame mask and phone mask)
        attn_mask = y_mask[:, :, :, None] * x_mask[:, :, None, :]   # [1,1,T_y,T_x]
        attn     = _compute_alignment_path_np(w_ceil, attn_mask) # [1, 1, T_y, T_x]

        # Expand prior stats via alignment
        m_p_exp    = np.matmul(attn[:, 0], m_p.transpose(0, 2, 1)).transpose(0, 2, 1)
        logs_p_exp = np.matmul(attn[:, 0], logs_p.transpose(0, 2, 1)).transpose(0, 2, 1)

        # ── 4. Sample z_p ─────────────────────────────────────────────────
        z_p = m_p_exp + np.random.randn(*m_p_exp.shape).astype(np.float32) * \
              np.exp(logs_p_exp) * noise_scale

        # ── 5. Flow (reverse) ─────────────────────────────────────────────
        z = self._flow.run(
            None,
            {"z_p": z_p, "y_mask": y_mask.astype(np.float32), "g": g},
        )[0]

        # ── 6. Decoder ────────────────────────────────────────────────────
        z_masked = (z * y_mask).astype(np.float32)
        audio = self._dec.run(None, {"z": z_masked, "g": g})[0]  # [1, 1, samples]

        audio_np = audio[0, 0]
        save_sr = SAMPLING_RATE
        if output_sr is not None and output_sr != SAMPLING_RATE:
            try:
                import torchaudio
                import torch
                wav_t = torch.from_numpy(audio_np).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(SAMPLING_RATE, output_sr)
                audio_np = resampler(wav_t).squeeze(0).numpy()
                save_sr = output_sr
            except Exception as e:
                print(f"[ONNX] Resampling failed ({e}), saving at {SAMPLING_RATE}Hz")

        sf.write(output_path, audio_np, save_sr)
        print(f"[ONNX] Saved: {output_path} ({save_sr}Hz)")
        return audio_np
