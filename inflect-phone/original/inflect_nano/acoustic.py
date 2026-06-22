from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio

SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[0]
FRONTEND_ROOT = PROJECT_ROOT / "third_party" / "tiny_tts_frontend"
sys.path = [str(FRONTEND_ROOT), str(SCRIPT_ROOT)] + [p for p in sys.path if p]

from inflect_nano.vocoder import HifiGanConfig, HifiGanGenerator, MelFrontend


@dataclass
class MicroFastSpeechConfig:
    vocab_size: int = 256
    tone_size: int = 16
    lang_size: int = 4
    n_mels: int = 80
    hidden: int = 168
    encoder_layers: int = 5
    decoder_layers: int = 6
    decoder_ff_mult: int = 3
    kernel_size: int = 7
    speaker_count: int = 2
    speaker_dim: int = 64
    dropout: float = 0.08
    sample_rate: int = 24000
    max_frames: int = 1400
    postnet_scale: float = 0.10
    use_frame_pitch: bool = True
    abs_frame_bins: int = 512
    use_contextual_predictors: bool = False
    use_group_duration_planner: bool = False


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_rows(path: Path, max_rows: int = 0) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if Path(str(row.get("target_audio") or "")).is_file():
                    rows.append(row)
                    if max_rows and len(rows) >= max_rows:
                        break
    if not rows:
        raise RuntimeError(f"No usable rows in {path}")
    return rows


def load_audio(path: str, sample_rate: int, max_seconds: float) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav[:, : int(sample_rate * max_seconds)].squeeze(0).clamp(-1.0, 1.0)


def fit_durations(durations: list[int], target_frames: int) -> list[int]:
    if sum(durations) == target_frames:
        return list(durations)
    total = max(1, sum(durations))
    raw = [max(0.0, d * target_frames / total) for d in durations]
    out = [int(math.floor(x)) for x in raw]
    order = sorted(((raw[i] - out[i], i) for i in range(len(out))), reverse=True)
    for _, idx in order[: max(0, target_frames - sum(out))]:
        out[idx] += 1
    while sum(out) > target_frames:
        idx = max(range(len(out)), key=lambda i: out[i])
        out[idx] -= 1
    return out


def pad_1d(items: list[torch.Tensor], value: float = 0.0) -> torch.Tensor:
    max_len = max(x.numel() for x in items)
    out = torch.full((len(items), max_len), value, dtype=items[0].dtype)
    for i, item in enumerate(items):
        out[i, : item.numel()] = item
    return out


def pad_2d(items: list[torch.Tensor], value: float = 0.0) -> torch.Tensor:
    max_len = max(x.shape[0] for x in items)
    dim = items[0].shape[1]
    out = torch.full((len(items), max_len, dim), value, dtype=items[0].dtype)
    for i, item in enumerate(items):
        out[i, : item.shape[0]] = item
    return out


def pad_mels(items: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(x.shape[-1] for x in items)
    n_mels = items[0].shape[0]
    out = torch.zeros(len(items), n_mels, max_len, dtype=items[0].dtype)
    mask = torch.zeros(len(items), max_len, dtype=torch.bool)
    for i, mel in enumerate(items):
        frames = mel.shape[-1]
        out[i, :, :frames] = mel
        mask[i, :frames] = True
    return out, mask


def pad_wavs(items: list[torch.Tensor], frames: list[int], hop_size: int) -> torch.Tensor:
    max_len = max(max(1, int(frame_count)) * hop_size for frame_count in frames)
    out = torch.zeros(len(items), max_len, dtype=items[0].dtype)
    for i, (wav, frame_count) in enumerate(zip(items, frames)):
        length = max(1, int(frame_count)) * hop_size
        cropped = wav[:length]
        out[i, : cropped.numel()] = cropped
    return out


def aggregate_token_features(mel: torch.Tensor, durations: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    # mel: [80, frames], log-mel from the exact V2+ frontend.
    frames = mel.shape[-1]
    amp = torch.exp(mel).clamp_min(1e-5)
    energy_frame = mel.mean(dim=0)
    bins = torch.linspace(0.0, 1.0, mel.shape[0], device=mel.device).view(-1, 1)
    bright_frame = (amp * bins).sum(dim=0) / amp.sum(dim=0).clamp_min(1e-5)
    energies = []
    brights = []
    pos = 0
    for dur in durations:
        end = min(frames, pos + max(0, int(dur)))
        if end > pos:
            energies.append(energy_frame[pos:end].mean())
            brights.append(bright_frame[pos:end].mean())
        else:
            energies.append(torch.zeros((), device=mel.device, dtype=mel.dtype))
            brights.append(torch.zeros((), device=mel.device, dtype=mel.dtype))
        pos = end
    return torch.stack(energies), torch.stack(brights)


def aggregate_token_pitch(pitch_frame: torch.Tensor, durations: list[int]) -> torch.Tensor:
    # pitch_frame: [2, frames] with normalized log-f0 and voiced flag.
    frames = pitch_frame.shape[-1]
    out = []
    pos = 0
    for dur in durations:
        end = min(frames, pos + max(0, int(dur)))
        if end > pos:
            span = pitch_frame[:, pos:end]
            voiced = span[1].mean()
            voiced_mask = span[1] > 0.5
            if bool(voiced_mask.any()):
                log_f0 = span[0, voiced_mask].mean()
            else:
                log_f0 = torch.zeros((), dtype=pitch_frame.dtype)
            out.append(torch.stack([log_f0, voiced]))
        else:
            out.append(torch.zeros(2, dtype=pitch_frame.dtype))
        pos = end
    return torch.stack(out, dim=0)


def extract_pitch_features(wav: torch.Tensor, sample_rate: int, frames: int) -> torch.Tensor:
    # Returns [2, frames]: normalized log-f0 and voiced flag. The detector can
    # produce octave spikes, so clip to speech range and median-smooth lightly.
    pitch = torchaudio.functional.detect_pitch_frequency(
        wav.unsqueeze(0).cpu(),
        sample_rate,
        frame_time=256 / sample_rate,
    ).squeeze(0)
    if pitch.numel() < frames:
        pitch = F.pad(pitch, (0, frames - pitch.numel()), value=0.0)
    pitch = pitch[:frames]
    voiced = ((pitch >= 55.0) & (pitch <= 420.0)).float()
    pitch = pitch.clamp(55.0, 420.0)
    # Median filter over 5 frames to reduce spurious jumps.
    if pitch.numel() >= 5:
        padded = F.pad(pitch.view(1, 1, -1), (2, 2), mode="replicate")
        windows = padded.unfold(-1, 5, 1).squeeze(0).squeeze(0)
        pitch = windows.median(dim=-1).values
    log_f0 = (torch.log(pitch) - math.log(140.0)) / 0.45
    log_f0 = log_f0.clamp(-3.0, 3.0) * voiced
    return torch.stack([log_f0, voiced], dim=0)


class ConvFFNBlock(nn.Module):
    def __init__(self, hidden: int, kernel_size: int, dropout: float, ff_mult: int = 4) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.norm1 = nn.LayerNorm(hidden)
        self.depth = nn.Conv1d(hidden, hidden * 2, kernel_size, padding=pad, groups=hidden)
        self.point = nn.Conv1d(hidden, hidden, 1)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * ff_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * ff_mult, hidden),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        y = self.norm1(x).transpose(1, 2)
        a, b = self.depth(y).chunk(2, dim=1)
        y = self.point(a * torch.sigmoid(b)).transpose(1, 2)
        x = x + self.drop(y)
        x = x + self.drop(self.ff(self.norm2(x)))
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return x


class MicroFastSpeech(nn.Module):
    def __init__(self, cfg: MicroFastSpeechConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Phone id 0 is a real inserted blank/silence token from TinyTTS, not
        # padding. Padding is tracked by duration masks instead.
        self.phone = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.tone = nn.Embedding(cfg.tone_size, cfg.hidden)
        self.lang = nn.Embedding(cfg.lang_size, cfg.hidden)
        self.speaker = nn.Embedding(cfg.speaker_count, cfg.speaker_dim)
        self.speaker_proj = nn.Linear(cfg.speaker_dim, cfg.hidden)
        self.encoder = nn.ModuleList([ConvFFNBlock(cfg.hidden, cfg.kernel_size, cfg.dropout) for _ in range(cfg.encoder_layers)])
        self.duration_head = nn.Sequential(nn.LayerNorm(cfg.hidden), nn.Linear(cfg.hidden, cfg.hidden), nn.SiLU(), nn.Linear(cfg.hidden, 1))
        self.energy_head = nn.Sequential(nn.LayerNorm(cfg.hidden), nn.Linear(cfg.hidden, cfg.hidden // 2), nn.SiLU(), nn.Linear(cfg.hidden // 2, 1))
        self.bright_head = nn.Sequential(nn.LayerNorm(cfg.hidden), nn.Linear(cfg.hidden, cfg.hidden // 2), nn.SiLU(), nn.Linear(cfg.hidden // 2, 1))
        self.pitch_head = nn.Sequential(nn.LayerNorm(cfg.hidden), nn.Linear(cfg.hidden, cfg.hidden), nn.SiLU(), nn.Linear(cfg.hidden, 2))
        self.group_duration_delta = nn.Linear(cfg.hidden, 1) if cfg.use_group_duration_planner else None
        if self.group_duration_delta is not None:
            nn.init.zeros_(self.group_duration_delta.weight)
            nn.init.zeros_(self.group_duration_delta.bias)
        self.predictor_context = (
            ConvFFNBlock(cfg.hidden, 5, cfg.dropout, 2) if cfg.use_contextual_predictors else nn.Identity()
        )
        self.duration_delta = nn.Linear(cfg.hidden, 1) if cfg.use_contextual_predictors else None
        self.energy_delta = nn.Linear(cfg.hidden, 1) if cfg.use_contextual_predictors else None
        self.bright_delta = nn.Linear(cfg.hidden, 1) if cfg.use_contextual_predictors else None
        self.pitch_delta = nn.Linear(cfg.hidden, 2) if cfg.use_contextual_predictors else None
        if cfg.use_contextual_predictors:
            for layer in (self.duration_delta, self.energy_delta, self.bright_delta, self.pitch_delta):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.energy_proj = nn.Linear(1, cfg.hidden)
        self.bright_proj = nn.Linear(1, cfg.hidden)
        self.pitch_proj = nn.Sequential(nn.Linear(2, cfg.hidden), nn.SiLU(), nn.Linear(cfg.hidden, cfg.hidden))
        self.abs_frame = nn.Embedding(cfg.abs_frame_bins, cfg.hidden)
        self.frame_proj = nn.Sequential(nn.Linear(8, cfg.hidden), nn.SiLU(), nn.Linear(cfg.hidden, cfg.hidden))
        self.local_ctx = nn.Sequential(
            nn.Linear(cfg.hidden * 3, cfg.hidden * 2),
            nn.SiLU(),
            nn.Linear(cfg.hidden * 2, cfg.hidden),
        )
        self.decoder = nn.ModuleList([ConvFFNBlock(cfg.hidden, cfg.kernel_size, cfg.dropout, cfg.decoder_ff_mult) for _ in range(cfg.decoder_layers)])
        self.frame_gru = nn.GRU(cfg.hidden, cfg.hidden // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.mel_head = nn.Sequential(nn.LayerNorm(cfg.hidden), nn.Linear(cfg.hidden, cfg.hidden), nn.SiLU(), nn.Linear(cfg.hidden, cfg.n_mels))
        self.postnet = nn.Sequential(
            nn.Conv1d(cfg.n_mels, cfg.hidden, 5, padding=2),
            nn.Tanh(),
            nn.Conv1d(cfg.hidden, cfg.hidden, 5, padding=2),
            nn.Tanh(),
            nn.Conv1d(cfg.hidden, cfg.n_mels, 5, padding=2),
        )

    def encode(self, phone: torch.Tensor, tone: torch.Tensor, lang: torch.Tensor, speaker: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        x = self.phone(phone) + self.tone(tone.clamp_max(self.cfg.tone_size - 1)) + self.lang(lang.clamp_max(self.cfg.lang_size - 1))
        x = x + self.speaker_proj(self.speaker(speaker)).unsqueeze(1)
        x = x * token_mask.unsqueeze(-1)
        for block in self.encoder:
            x = block(x, token_mask)
        return x

    def regulate(self, encoded: torch.Tensor, durations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_frames = []
        batch_meta = []
        lengths = []
        device = encoded.device
        for b in range(encoded.shape[0]):
            reps = []
            meta = []
            durs = durations[b].long().clamp_min(0)
            token_count = max(1, int((durs > 0).sum().item()))
            for i, dur_t in enumerate(durs.tolist()):
                dur = int(dur_t)
                if dur <= 0:
                    continue
                reps.append(encoded[b, i].view(1, -1).expand(dur, -1))
                rel = torch.linspace(0.0, 1.0, dur, device=device)
                token_pos = torch.full((dur,), i / max(1, token_count - 1), device=device)
                log_dur = torch.full((dur,), math.log1p(dur) / 6.0, device=device)
                inv_rel = 1.0 - rel
                center = 1.0 - torch.abs(rel * 2.0 - 1.0)
                meta.append(
                    torch.stack(
                        [
                            rel,
                            inv_rel,
                            center,
                            torch.sin(rel * math.pi),
                            torch.cos(rel * math.pi),
                            token_pos,
                            log_dur,
                            torch.full_like(rel, dur / 40.0),
                        ],
                        dim=-1,
                    )
                )
            if reps:
                frames = torch.cat(reps, dim=0)
                frame_meta = torch.cat(meta, dim=0)
            else:
                frames = encoded[b, :1]
                frame_meta = torch.zeros(1, 8, device=device)
            batch_frames.append(frames[: self.cfg.max_frames])
            batch_meta.append(frame_meta[: self.cfg.max_frames])
            lengths.append(min(frames.shape[0], self.cfg.max_frames))
        max_len = max(lengths)
        out = torch.zeros(encoded.shape[0], max_len, encoded.shape[-1], device=device)
        meta_out = torch.zeros(encoded.shape[0], max_len, 8, device=device)
        mask = torch.zeros(encoded.shape[0], max_len, dtype=torch.bool, device=device)
        for b, frames in enumerate(batch_frames):
            n = min(frames.shape[0], max_len)
            out[b, :n] = frames[:n]
            meta_out[b, :n] = batch_meta[b][:n]
            mask[b, :n] = True
        return out, meta_out, mask

    def add_local_context(self, encoded: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        device = encoded.device
        batch_frames = []
        for b in range(encoded.shape[0]):
            reps = []
            durs = durations[b].long().clamp_min(0)
            for i, dur_t in enumerate(durs.tolist()):
                dur = int(dur_t)
                if dur <= 0:
                    continue
                prev_i = max(0, i - 1)
                next_i = min(encoded.shape[1] - 1, i + 1)
                ctx = torch.cat([encoded[b, prev_i], encoded[b, i], encoded[b, next_i]], dim=-1)
                reps.append(ctx.view(1, -1).expand(dur, -1))
            if reps:
                frames = torch.cat(reps, dim=0)
            else:
                frames = torch.zeros(1, encoded.shape[-1] * 3, device=device)
            batch_frames.append(frames[: self.cfg.max_frames])
        max_len = max(x.shape[0] for x in batch_frames)
        ctx_out = torch.zeros(encoded.shape[0], max_len, encoded.shape[-1] * 3, device=device)
        for b, frames in enumerate(batch_frames):
            ctx_out[b, : frames.shape[0]] = frames
        return self.local_ctx(ctx_out)

    def expand_token_feature(self, feature: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        device = feature.device
        batch_frames = []
        for b in range(feature.shape[0]):
            reps = []
            durs = durations[b].long().clamp_min(0)
            for i, dur_t in enumerate(durs.tolist()):
                dur = int(dur_t)
                if dur <= 0:
                    continue
                reps.append(feature[b, i].view(1, -1).expand(dur, -1))
            if reps:
                frames = torch.cat(reps, dim=0)
            else:
                frames = torch.zeros(1, feature.shape[-1], device=device)
            batch_frames.append(frames[: self.cfg.max_frames])
        max_len = max(x.shape[0] for x in batch_frames)
        out = torch.zeros(feature.shape[0], max_len, feature.shape[-1], device=device)
        for b, frames in enumerate(batch_frames):
            out[b, : frames.shape[0]] = frames
        return out

    def forward(
        self,
        phone: torch.Tensor,
        tone: torch.Tensor,
        lang: torch.Tensor,
        speaker: torch.Tensor,
        durations: torch.Tensor,
        energy_target: torch.Tensor | None = None,
        bright_target: torch.Tensor | None = None,
        pitch_frame: torch.Tensor | None = None,
        predicted_prosody_mix: float = 0.0,
        detach_mixed_predictions: bool = True,
    ) -> dict[str, torch.Tensor]:
        token_mask = durations.gt(0)
        encoded = self.encode(phone, tone, lang, speaker, token_mask)
        log_dur, energy_pred, bright_pred, pitch_pred = self.predict_prosody(encoded, token_mask)
        mixed_energy_pred = energy_pred.detach() if detach_mixed_predictions else energy_pred
        mixed_bright_pred = bright_pred.detach() if detach_mixed_predictions else bright_pred
        if energy_target is not None:
            energy = torch.lerp(energy_target, mixed_energy_pred, predicted_prosody_mix)
        else:
            energy = energy_pred
        if bright_target is not None:
            bright = torch.lerp(bright_target, mixed_bright_pred, predicted_prosody_mix)
        else:
            bright = bright_pred
        conditioned = encoded + self.energy_proj(energy.unsqueeze(-1)) + self.bright_proj(bright.unsqueeze(-1))
        frames, frame_meta, frame_mask = self.regulate(conditioned, durations)
        x = frames + self.frame_proj(frame_meta) + self.add_local_context(conditioned, durations)
        pos = torch.arange(x.shape[1], device=x.device)
        pos = torch.div(pos * self.cfg.abs_frame_bins, max(1, self.cfg.max_frames), rounding_mode="floor").clamp_max(
            self.cfg.abs_frame_bins - 1
        )
        x = x + self.abs_frame(pos).unsqueeze(0)
        if self.cfg.use_frame_pitch:
            if pitch_frame is not None:
                pitch_frame = pitch_frame[:, :, : x.shape[1]].transpose(1, 2)
                if pitch_frame.shape[1] < x.shape[1]:
                    pitch_frame = F.pad(pitch_frame, (0, 0, 0, x.shape[1] - pitch_frame.shape[1]))
                if predicted_prosody_mix > 0.0:
                    mixed_pitch_pred = pitch_pred.detach() if detach_mixed_predictions else pitch_pred
                    predicted_pitch_frame = self.expand_token_feature(mixed_pitch_pred, durations)[:, : x.shape[1]]
                    pitch_frame = torch.lerp(pitch_frame, predicted_pitch_frame, predicted_prosody_mix)
            else:
                pitch_frame = self.expand_token_feature(pitch_pred, durations)[:, : x.shape[1]]
            x = x + self.pitch_proj(pitch_frame)
        for block in self.decoder:
            x = block(x, frame_mask)
        x = x + self.frame_gru(x)[0]
        mel = self.mel_head(x).transpose(1, 2)
        mel = mel + self.cfg.postnet_scale * self.postnet(mel)
        group_log_dur, group_mask = self.group_log_durations(phone, log_dur, encoded)
        return {
            "mel": mel,
            "frame_mask": frame_mask,
            "log_dur": log_dur,
            "group_log_dur": group_log_dur,
            "group_mask": group_mask,
            "energy": energy_pred,
            "bright": bright_pred,
            "pitch": pitch_pred,
            "token_mask": token_mask,
        }

    def predict_prosody(
        self, encoded: torch.Tensor, token_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        log_dur = self.duration_head(encoded).squeeze(-1)
        energy = self.energy_head(encoded).squeeze(-1)
        bright = self.bright_head(encoded).squeeze(-1)
        pitch = self.pitch_head(encoded)
        if self.cfg.use_contextual_predictors:
            context = self.predictor_context(encoded, token_mask)
            log_dur = log_dur + self.duration_delta(context).squeeze(-1)
            energy = energy + self.energy_delta(context).squeeze(-1)
            bright = bright + self.bright_delta(context).squeeze(-1)
            pitch = pitch + self.pitch_delta(context)
        return log_dur, energy, bright, pitch

    def group_log_durations(
        self, phone: torch.Tensor, log_dur: torch.Tensor, encoded: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict stable blank-plus-phone region durations at visible phones."""
        base_dur = torch.expm1(log_dur).clamp_min(0.05)
        grouped = torch.zeros_like(log_dur)
        group_mask = torch.zeros_like(phone, dtype=torch.bool)
        delta = self.group_duration_delta(encoded).squeeze(-1) if self.group_duration_delta is not None else None
        for batch_index in range(phone.shape[0]):
            pending: list[torch.Tensor] = []
            last_visible: int | None = None
            for token_index in range(phone.shape[1]):
                pending.append(base_dur[batch_index, token_index])
                if int(phone[batch_index, token_index].item()) != 0:
                    value = torch.stack(pending).sum()
                    if delta is not None:
                        value = value * torch.exp(delta[batch_index, token_index].clamp(-1.5, 1.5))
                    grouped[batch_index, token_index] = torch.log1p(value)
                    group_mask[batch_index, token_index] = True
                    pending = []
                    last_visible = token_index
            if pending and last_visible is not None:
                value = torch.expm1(grouped[batch_index, last_visible]) + torch.stack(pending).sum()
                grouped[batch_index, last_visible] = torch.log1p(value)
        return grouped, group_mask

    def apply_group_duration_plan(
        self, phone: torch.Tensor, log_dur: torch.Tensor, encoded: torch.Tensor, length_scale: float, max_duration: int
    ) -> torch.Tensor:
        base = torch.expm1(log_dur).clamp_min(0.05)
        group_log, _ = self.group_log_durations(phone, log_dur, encoded)
        planned = torch.zeros_like(base, dtype=torch.long)
        for batch_index in range(phone.shape[0]):
            pending: list[int] = []
            last_visible: int | None = None
            for token_index in range(phone.shape[1]):
                pending.append(token_index)
                if int(phone[batch_index, token_index].item()) != 0:
                    target = max(len(pending), int(round(float(torch.expm1(group_log[batch_index, token_index]) * length_scale))))
                    weights = base[batch_index, pending]
                    remaining = target - len(pending)
                    raw = weights / weights.sum().clamp_min(1e-6) * remaining
                    allocated = torch.ones_like(raw, dtype=torch.long) + torch.floor(raw).long()
                    remainder = target - int(allocated.sum().item())
                    if remainder > 0:
                        order = torch.argsort(raw - torch.floor(raw), descending=True)
                        allocated[order[:remainder]] += 1
                    planned[batch_index, pending] = allocated
                    pending = []
                    last_visible = token_index
            if pending and last_visible is not None:
                planned[batch_index, last_visible] += max(1, int(round(float(base[batch_index, pending].sum() * length_scale))))
        return planned.clamp(0, max_duration)

    @torch.no_grad()
    def infer(
        self,
        phone: torch.Tensor,
        tone: torch.Tensor,
        lang: torch.Tensor,
        speaker: torch.Tensor,
        length_scale: float = 1.0,
        min_duration: int = 1,
        max_duration: int = 80,
        pitch_scale: float = 1.0,
        energy_scale: float = 1.0,
        smooth_predictors: bool = False,
    ) -> torch.Tensor:
        # In single-sample inference there is no padded tail; id 0 remains the
        # explicit blank/pause token and must keep duration.
        token_mask = torch.ones_like(phone, dtype=torch.bool)
        encoded = self.encode(phone, tone, lang, speaker, token_mask)
        log_dur, energy, bright, pitch = self.predict_prosody(encoded, token_mask)
        if self.group_duration_delta is not None:
            durations = self.apply_group_duration_plan(phone, log_dur, encoded, length_scale, max_duration)
            durations = durations.masked_fill(~token_mask, 0)
        else:
            pred_dur = torch.expm1(log_dur).clamp(0, max_duration) * length_scale
            durations = torch.round(pred_dur).long().clamp_min(min_duration).masked_fill(~token_mask, 0)
        energy = energy * energy_scale
        pitch = torch.stack([pitch[..., 0] * pitch_scale, pitch[..., 1].clamp(0.0, 1.0)], dim=-1)
        if smooth_predictors and phone.shape[1] >= 3:
            energy = F.avg_pool1d(energy.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
            bright = F.avg_pool1d(bright.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
            pitch_t = pitch.transpose(1, 2)
            pitch = F.avg_pool1d(pitch_t, 3, stride=1, padding=1).transpose(1, 2)
        conditioned = encoded + self.energy_proj(energy.unsqueeze(-1)) + self.bright_proj(bright.unsqueeze(-1))
        frames, frame_meta, frame_mask = self.regulate(conditioned, durations)
        x = frames + self.frame_proj(frame_meta) + self.add_local_context(conditioned, durations)
        pos = torch.arange(x.shape[1], device=x.device)
        pos = torch.div(pos * self.cfg.abs_frame_bins, max(1, self.cfg.max_frames), rounding_mode="floor").clamp_max(
            self.cfg.abs_frame_bins - 1
        )
        x = x + self.abs_frame(pos).unsqueeze(0)
        if self.cfg.use_frame_pitch:
            pitch_frame = self.expand_token_feature(pitch, durations)[:, : x.shape[1]]
            x = x + self.pitch_proj(pitch_frame)
        for block in self.decoder:
            x = block(x, frame_mask)
        x = x + self.frame_gru(x)[0]
        mel = self.mel_head(x).transpose(1, 2)
        mel = mel + self.cfg.postnet_scale * self.postnet(mel)
        return mel


def collate(batch: list[dict], cfg: MicroFastSpeechConfig, mel_frontend: MelFrontend, device: torch.device, max_seconds: float, hop_size: int):
    phones = [torch.LongTensor(x["phone_ids"]) for x in batch]
    tones = [torch.LongTensor(x["tone_ids"]) for x in batch]
    langs = [torch.LongTensor(x["lang_ids"]) for x in batch]
    durations_raw = [list(map(int, x["hifigan_durations"])) for x in batch]
    speakers = torch.LongTensor([int(x["speaker_id"]) for x in batch])
    phone = pad_1d(phones, 0).long()
    tone = pad_1d(tones, 0).long()
    lang = pad_1d(langs, 0).long()
    mels = []
    durations = []
    energies = []
    brights = []
    pitches = []
    token_pitches = []
    wavs = []
    frame_counts = []
    with torch.no_grad():
        for row, dur in zip(batch, durations_raw):
            wav_1d = load_audio(str(row["target_audio"]), cfg.sample_rate, max_seconds)
            wav = wav_1d.unsqueeze(0).to(device)
            mel = mel_frontend(wav).squeeze(0).detach().cpu()
            dur = fit_durations(dur[: len(row["phone_ids"])], min(mel.shape[-1], cfg.max_frames))
            mel = mel[:, : sum(dur)]
            energy, bright = aggregate_token_features(mel, dur)
            pitch = extract_pitch_features(wav_1d, cfg.sample_rate, mel.shape[-1])
            token_pitch = aggregate_token_pitch(pitch, dur)
            mels.append(mel)
            durations.append(torch.LongTensor(dur))
            energies.append(energy)
            brights.append(bright)
            pitches.append(pitch)
            token_pitches.append(token_pitch)
            wavs.append(wav_1d)
            frame_counts.append(mel.shape[-1])
    duration = pad_1d(durations, 0).long()
    energy = pad_1d(energies, 0.0).float()
    bright = pad_1d(brights, 0.0).float()
    token_pitch = pad_2d(token_pitches, 0.0).float()
    target_mel, frame_mask = pad_mels(mels)
    pitch_frame, _ = pad_mels(pitches)
    target_wav = pad_wavs(wavs, frame_counts, hop_size)
    return (
        phone.to(device),
        tone.to(device),
        lang.to(device),
        speakers.to(device),
        duration.to(device),
        energy.to(device),
        bright.to(device),
        token_pitch.to(device),
        target_mel.to(device),
        frame_mask.to(device),
        pitch_frame.to(device),
        target_wav.to(device),
    )


def prepare_row_features(
    row: dict,
    cfg: MicroFastSpeechConfig,
    mel_frontend: MelFrontend,
    device: torch.device,
    max_seconds: float,
) -> dict:
    dur = list(map(int, row["hifigan_durations"]))
    wav_1d = load_audio(str(row["target_audio"]), cfg.sample_rate, max_seconds)
    with torch.no_grad():
        wav = wav_1d.unsqueeze(0).to(device)
        mel = mel_frontend(wav).squeeze(0).detach().cpu()
    dur = fit_durations(dur[: len(row["phone_ids"])], min(mel.shape[-1], cfg.max_frames))
    mel = mel[:, : sum(dur)]
    energy, bright = aggregate_token_features(mel, dur)
    pitch = extract_pitch_features(wav_1d, cfg.sample_rate, mel.shape[-1])
    token_pitch = aggregate_token_pitch(pitch, dur)
    return {
        "phone": torch.LongTensor(row["phone_ids"]),
        "tone": torch.LongTensor(row["tone_ids"]),
        "lang": torch.LongTensor(row["lang_ids"]),
        "speaker": int(row["speaker_id"]),
        "duration": torch.LongTensor(dur),
        "energy": energy.float(),
        "bright": bright.float(),
        "token_pitch": token_pitch.float(),
        "target_mel": mel.float(),
        "pitch_frame": pitch.float(),
        "target_wav": wav_1d.float(),
        "frame_count": int(mel.shape[-1]),
    }


def collate_prepared(batch: list[dict], device: torch.device, hop_size: int):
    phone = pad_1d([x["phone"] for x in batch], 0).long()
    tone = pad_1d([x["tone"] for x in batch], 0).long()
    lang = pad_1d([x["lang"] for x in batch], 0).long()
    speakers = torch.LongTensor([int(x["speaker"]) for x in batch])
    duration = pad_1d([x["duration"] for x in batch], 0).long()
    energy = pad_1d([x["energy"] for x in batch], 0.0).float()
    bright = pad_1d([x["bright"] for x in batch], 0.0).float()
    token_pitch = pad_2d([x["token_pitch"] for x in batch], 0.0).float()
    target_mel, frame_mask = pad_mels([x["target_mel"] for x in batch])
    pitch_frame, _ = pad_mels([x["pitch_frame"] for x in batch])
    target_wav = pad_wavs([x["target_wav"] for x in batch], [int(x["frame_count"]) for x in batch], hop_size)
    return (
        phone.to(device),
        tone.to(device),
        lang.to(device),
        speakers.to(device),
        duration.to(device),
        energy.to(device),
        bright.to(device),
        token_pitch.to(device),
        target_mel.to(device),
        frame_mask.to(device),
        pitch_frame.to(device),
        target_wav.to(device),
    )


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    common = min(pred.shape[-1], target.shape[-1], mask.shape[-1])
    pred = pred[..., :common]
    target = target[..., :common]
    mask = mask[:, :common].unsqueeze(1)
    return (torch.abs(pred - target) * mask).sum() / (mask.sum() * pred.shape[1]).clamp_min(1.0)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    common = min(pred.shape[-1], target.shape[-1], mask.shape[-1])
    pred = pred[..., :common]
    target = target[..., :common]
    mask = mask[:, :common].unsqueeze(1)
    return (((pred - target) ** 2) * mask).sum() / (mask.sum() * pred.shape[1]).clamp_min(1.0)


def masked_delta_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    common = min(pred.shape[-1], target.shape[-1], mask.shape[-1])
    if common < 2:
        return torch.zeros((), device=pred.device)
    dp = pred[..., 1:common] - pred[..., : common - 1]
    dt = target[..., 1:common] - target[..., : common - 1]
    dm = (mask[:, 1:common] & mask[:, : common - 1]).unsqueeze(1)
    return (torch.abs(dp - dt) * dm).sum() / (dm.sum() * pred.shape[1]).clamp_min(1.0)


def masked_accel_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    common = min(pred.shape[-1], target.shape[-1], mask.shape[-1])
    if common < 3:
        return torch.zeros((), device=pred.device)
    dp = pred[..., 2:common] - 2.0 * pred[..., 1 : common - 1] + pred[..., : common - 2]
    dt = target[..., 2:common] - 2.0 * target[..., 1 : common - 1] + target[..., : common - 2]
    dm = (mask[:, 2:common] & mask[:, 1 : common - 1] & mask[:, : common - 2]).unsqueeze(1)
    return (torch.abs(dp - dt) * dm).sum() / (dm.sum() * pred.shape[1]).clamp_min(1.0)


def token_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    common = min(pred.shape[1], target.shape[1], mask.shape[1])
    pred = pred[:, :common]
    target = target[:, :common]
    mask = mask[:, :common]
    return (((pred - target) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)


def token_mse_nd(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    common = min(pred.shape[1], target.shape[1], mask.shape[1])
    pred = pred[:, :common]
    target = target[:, :common]
    mask = mask[:, :common].unsqueeze(-1)
    return (((pred - target) ** 2) * mask).sum() / (mask.sum() * pred.shape[-1]).clamp_min(1.0)


def group_duration_targets(phone: torch.Tensor, durations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grouped = torch.zeros_like(durations, dtype=torch.float32)
    mask = torch.zeros_like(phone, dtype=torch.bool)
    for batch_index in range(phone.shape[0]):
        pending: list[torch.Tensor] = []
        last_visible: int | None = None
        for token_index in range(phone.shape[1]):
            pending.append(durations[batch_index, token_index].float())
            if int(phone[batch_index, token_index].item()) != 0:
                grouped[batch_index, token_index] = torch.log1p(torch.stack(pending).sum())
                mask[batch_index, token_index] = True
                pending = []
                last_visible = token_index
        if pending and last_visible is not None:
            value = torch.expm1(grouped[batch_index, last_visible]) + torch.stack(pending).sum()
            grouped[batch_index, last_visible] = torch.log1p(value)
    return grouped, mask


def masked_wav_l1(pred: torch.Tensor, target: torch.Tensor, frame_mask: torch.Tensor, hop_size: int) -> torch.Tensor:
    if pred.dim() == 3:
        pred = pred.squeeze(1)
    common = min(pred.shape[-1], target.shape[-1], frame_mask.shape[-1] * hop_size)
    pred = pred[:, :common]
    target = target[:, :common]
    sample_mask = frame_mask.repeat_interleave(hop_size, dim=1)[:, :common].to(pred.dtype)
    return (torch.abs(pred - target) * sample_mask).sum() / sample_mask.sum().clamp_min(1.0)


def load_frozen_vocoder(path: Path, device: torch.device) -> tuple[HifiGanGenerator, HifiGanConfig]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_payload = ckpt.get("config") or {"variant": "v2plus"}
    cfg = HifiGanConfig(**cfg_payload) if isinstance(cfg_payload, dict) else cfg_payload
    vocoder = HifiGanGenerator(cfg).to(device)
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    for param in vocoder.parameters():
        param.requires_grad_(False)
    return vocoder, cfg


def load_model_state_flexible(model: nn.Module, state: dict[str, torch.Tensor]) -> tuple[int, int]:
    current = model.state_dict()
    compatible = {key: value for key, value in state.items() if key in current and current[key].shape == value.shape}
    model.load_state_dict(compatible, strict=False)
    return len(compatible), len(state) - len(compatible)


def set_trainable_by_mode(model: MicroFastSpeech, mode: str) -> None:
    if mode == "all":
        for param in model.parameters():
            param.requires_grad_(True)
        return
    for param in model.parameters():
        param.requires_grad_(False)
    prefixes: tuple[str, ...]
    if mode == "duration":
        prefixes = ("phone.", "tone.", "lang.", "speaker.", "speaker_proj.", "encoder.", "duration_head.")
    elif mode == "predictors":
        prefixes = (
            "phone.",
            "tone.",
            "lang.",
            "speaker.",
            "speaker_proj.",
            "encoder.",
            "duration_head.",
            "energy_head.",
            "bright_head.",
            "pitch_head.",
        )
    elif mode == "heads":
        prefixes = (
            "duration_head.",
            "energy_head.",
            "bright_head.",
            "pitch_head.",
            "predictor_context.",
            "duration_delta.",
            "energy_delta.",
            "bright_delta.",
            "pitch_delta.",
        )
    elif mode == "contextual":
        prefixes = (
            "predictor_context.",
            "duration_delta.",
            "energy_delta.",
            "bright_delta.",
            "pitch_delta.",
        )
    elif mode == "group_duration":
        prefixes = ("group_duration_delta.",)
    elif mode == "decoder_adapt":
        prefixes = (
            "energy_proj.",
            "bright_proj.",
            "pitch_proj.",
            "abs_frame.",
            "frame_proj.",
            "local_ctx.",
            "decoder.",
            "frame_gru.",
            "mel_head.",
            "postnet.",
        )
    else:
        raise ValueError(f"Unknown trainable mode: {mode}")
    for name, param in model.named_parameters():
        if name.startswith(prefixes):
            param.requires_grad_(True)


def latest_checkpoint(out_dir: Path) -> Path | None:
    found = []
    for path in out_dir.glob("inflect-micro-fastspeech-*.pt"):
        tail = path.stem.rsplit("-", 1)[-1]
        if tail.isdigit():
            found.append((int(tail), path))
    return max(found)[1] if found else None


def save_checkpoint(path: Path, model: nn.Module, optim, cfg: MicroFastSpeechConfig, step: int, args, speakers: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "config": asdict(cfg),
            "step": step,
            "speakers": speakers,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "params": count_parameters(model),
        },
        tmp,
    )
    tmp.replace(path)


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    rows = load_rows(args.durations_jsonl, args.max_rows)
    speakers = {voice: idx for idx, voice in enumerate(sorted({str(r.get("voice_id") or "mark") for r in rows}))}
    max_phone_id = max(max(map(int, r["phone_ids"])) for r in rows)
    max_tone_id = max(max(map(int, r["tone_ids"])) for r in rows)
    max_lang_id = max(max(map(int, r["lang_ids"])) for r in rows)
    cfg = MicroFastSpeechConfig(
        vocab_size=max(256, max_phone_id + 1),
        tone_size=max(16, max_tone_id + 1),
        lang_size=max(4, max_lang_id + 1),
        speaker_count=max(2, len(speakers)),
        hidden=args.hidden,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        decoder_ff_mult=args.decoder_ff_mult,
        max_frames=args.max_frames,
        postnet_scale=args.postnet_scale,
        abs_frame_bins=args.abs_frame_bins,
        use_contextual_predictors=args.contextual_predictors,
        use_group_duration_planner=args.group_duration_planner,
    )
    for row in rows:
        row["speaker_id"] = speakers[str(row.get("voice_id") or "mark")]
    random.Random(args.seed).shuffle(rows)

    model = MicroFastSpeech(cfg).to(device)
    start_step = 0
    if args.init_checkpoint and not args.resume:
        ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        copied, skipped = load_model_state_flexible(model, ckpt["model"])
        print(f"Initialized model from {args.init_checkpoint} ({copied} tensors copied, {skipped} skipped)")
    set_trainable_by_mode(model, args.trainable)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
    if args.resume:
        ckpt_path = latest_checkpoint(args.out_dir)
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optim.load_state_dict(ckpt["optim"])
            start_step = int(ckpt.get("step") or 0)
            print(f"Resumed {ckpt_path} at step {start_step}")

    hifi_cfg = HifiGanConfig(variant="v2plus")
    mel_frontend = MelFrontend(hifi_cfg).to(device)
    prepared_rows = None
    if args.preload_features:
        print("Preloading audio/mel/pitch features...", flush=True)
        prepared_rows = [prepare_row_features(row, cfg, mel_frontend, device, args.max_seconds) for row in rows]
        total_frames = sum(int(row["frame_count"]) for row in prepared_rows)
        print(f"Preloaded {len(prepared_rows)} rows ({total_frames:,} frames)", flush=True)
    consistency_vocoder = None
    if args.vocoder_checkpoint:
        consistency_vocoder, consistency_cfg = load_frozen_vocoder(args.vocoder_checkpoint, device)
        if consistency_cfg.hop_size != hifi_cfg.hop_size:
            raise RuntimeError(f"Vocoder hop mismatch: {consistency_cfg.hop_size} != {hifi_cfg.hop_size}")
        print(f"Loaded frozen vocoder consistency checkpoint: {args.vocoder_checkpoint}")
    if (args.vocoder_wav_weight > 0.0 or args.vocoder_mel_weight > 0.0) and consistency_vocoder is None:
        raise RuntimeError("--vocoder-checkpoint is required when vocoder consistency losses are enabled")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(
        json.dumps({"config": asdict(cfg), "speakers": speakers, "rows": len(rows), "params": count_parameters(model)}, indent=2),
        encoding="utf-8",
    )

    print(f"Rows: {len(rows)}")
    print(f"Speakers: {speakers}")
    print(f"Acoustic params: {count_parameters(model):,} ({count_parameters(model)/1_000_000:.3f}M)")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} mode={args.trainable}")
    print(f"Total with V2+ vocoder: {(count_parameters(model)+1_426_842):,} ({(count_parameters(model)+1_426_842)/1_000_000:.3f}M)")

    rng = random.Random(args.seed + start_step)
    step = start_step
    started = time.time()
    while step < args.steps:
        source_rows = prepared_rows if prepared_rows is not None else rows
        batch = [source_rows[rng.randrange(len(source_rows))] for _ in range(args.batch_size)]
        if prepared_rows is not None:
            phone, tone, lang, speaker, durations, energy_t, bright_t, pitch_token_t, target_mel, frame_mask, pitch_frame, target_wav = collate_prepared(
                batch, device, hifi_cfg.hop_size
            )
        else:
            phone, tone, lang, speaker, durations, energy_t, bright_t, pitch_token_t, target_mel, frame_mask, pitch_frame, target_wav = collate(
                batch, cfg, mel_frontend, device, args.max_seconds, hifi_cfg.hop_size
            )
        out = model(phone, tone, lang, speaker, durations, energy_t, bright_t, pitch_frame)
        token_mask = out["token_mask"]
        log_dur_t = torch.log1p(durations.float())
        group_log_dur_t, group_mask = group_duration_targets(phone, durations)
        mel_l1 = masked_l1(out["mel"], target_mel, frame_mask)
        mel_mse = masked_mse(out["mel"], target_mel, frame_mask)
        delta = masked_delta_loss(out["mel"], target_mel, frame_mask)
        accel = masked_accel_loss(out["mel"], target_mel, frame_mask)
        dur_loss = token_mse(out["log_dur"], log_dur_t, token_mask)
        group_dur_loss = token_mse(out["group_log_dur"], group_log_dur_t, group_mask)
        energy_loss = token_mse(out["energy"], energy_t, token_mask)
        bright_loss = token_mse(out["bright"], bright_t, token_mask)
        pitch_loss = token_mse_nd(out["pitch"], pitch_token_t, token_mask)
        predicted_prosody_mel_loss = torch.zeros((), device=device)
        predicted_prosody_delta_loss = torch.zeros((), device=device)
        if args.predicted_prosody_mel_weight > 0.0 or args.predicted_prosody_delta_weight > 0.0:
            # Train the predictor heads against the acoustic result they produce at
            # inference, while retaining reference durations so this path remains
            # differentiable and isolates prosody exposure bias.
            predicted_conditioning = model(phone, tone, lang, speaker, durations)
            if args.predicted_prosody_mel_weight > 0.0:
                predicted_prosody_mel_loss = masked_l1(predicted_conditioning["mel"], target_mel, frame_mask)
            if args.predicted_prosody_delta_weight > 0.0:
                predicted_prosody_delta_loss = masked_delta_loss(predicted_conditioning["mel"], target_mel, frame_mask)
        robust_prosody_mel_loss = torch.zeros((), device=device)
        robust_prosody_delta_loss = torch.zeros((), device=device)
        if args.robust_prosody_mel_weight > 0.0 or args.robust_prosody_delta_weight > 0.0:
            robust_conditioning = model(
                phone,
                tone,
                lang,
                speaker,
                durations,
                energy_t,
                bright_t,
                pitch_frame,
                predicted_prosody_mix=args.robust_prosody_mix,
                detach_mixed_predictions=True,
            )
            if args.robust_prosody_mel_weight > 0.0:
                robust_prosody_mel_loss = masked_l1(robust_conditioning["mel"], target_mel, frame_mask)
            if args.robust_prosody_delta_weight > 0.0:
                robust_prosody_delta_loss = masked_delta_loss(robust_conditioning["mel"], target_mel, frame_mask)
        voc_wav_loss = torch.zeros((), device=device)
        voc_mel_loss = torch.zeros((), device=device)
        if consistency_vocoder is not None and (args.vocoder_wav_weight > 0.0 or args.vocoder_mel_weight > 0.0):
            pred_wav = consistency_vocoder(out["mel"].clamp(-12.0, 2.0))
            if args.vocoder_wav_weight > 0.0:
                voc_wav_loss = masked_wav_l1(pred_wav, target_wav, frame_mask, hifi_cfg.hop_size)
            if args.vocoder_mel_weight > 0.0:
                pred_recon_mel = mel_frontend(pred_wav.squeeze(1))
                voc_mel_loss = masked_l1(pred_recon_mel, target_mel, frame_mask)
        loss = (
            mel_l1
            + args.mse_weight * mel_mse
            + args.delta_weight * delta
            + args.accel_weight * accel
            + args.duration_weight * dur_loss
            + args.group_duration_weight * group_dur_loss
            + args.energy_weight * energy_loss
            + args.bright_weight * bright_loss
            + args.pitch_weight * pitch_loss
            + args.predicted_prosody_mel_weight * predicted_prosody_mel_loss
            + args.predicted_prosody_delta_weight * predicted_prosody_delta_loss
            + args.robust_prosody_mel_weight * robust_prosody_mel_loss
            + args.robust_prosody_delta_weight * robust_prosody_delta_loss
            + args.vocoder_wav_weight * voc_wav_loss
            + args.vocoder_mel_weight * voc_mel_loss
        )
        optim.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        optim.step()
        step += 1

        if step == 1 or step % args.log_interval == 0:
            elapsed = max(1e-6, time.time() - started)
            speed = (step - start_step) / elapsed
            eta = (args.steps - step) / max(1e-6, speed)
            print(
                f"step={step}/{args.steps} loss={loss.item():.4f} mel={mel_l1.item():.4f} "
                f"mse={mel_mse.item():.4f} delta={delta.item():.4f} accel={accel.item():.4f} dur={dur_loss.item():.4f} "
                f"gdur={group_dur_loss.item():.4f} "
                f"energy={energy_loss.item():.4f} bright={bright_loss.item():.4f} "
                f"pitch={pitch_loss.item():.4f} pmel={predicted_prosody_mel_loss.item():.4f} "
                f"pdelta={predicted_prosody_delta_loss.item():.4f} rmel={robust_prosody_mel_loss.item():.4f} "
                f"rdelta={robust_prosody_delta_loss.item():.4f} vwav={voc_wav_loss.item():.4f} "
                f"vmel={voc_mel_loss.item():.4f} grad={float(grad):.2f} "
                f"speed={speed:.3f} step/s eta={eta/60:.1f}m",
                flush=True,
            )
        if step % args.save_interval == 0 or step >= args.steps:
            save_checkpoint(args.out_dir / f"inflect-micro-fastspeech-{step}.pt", model, optim, cfg, step, args, speakers)
            save_checkpoint(args.out_dir / "inflect-micro-fastspeech-latest.pt", model, optim, cfg, step, args, speakers)

    print(f"Done. {args.out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Inflect Micro duration-conditioned acoustic model.")
    ap.add_argument("--durations-jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--weight-decay", type=float, default=1.0e-4)
    ap.add_argument("--hidden", type=int, default=168)
    ap.add_argument("--encoder-layers", type=int, default=5)
    ap.add_argument("--decoder-layers", type=int, default=6)
    ap.add_argument("--decoder-ff-mult", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=12.0)
    ap.add_argument("--max-frames", type=int, default=1400)
    ap.add_argument("--mse-weight", type=float, default=0.25)
    ap.add_argument("--delta-weight", type=float, default=0.18)
    ap.add_argument("--accel-weight", type=float, default=0.0)
    ap.add_argument("--duration-weight", type=float, default=0.08)
    ap.add_argument("--group-duration-weight", type=float, default=0.0)
    ap.add_argument("--energy-weight", type=float, default=0.04)
    ap.add_argument("--bright-weight", type=float, default=0.04)
    ap.add_argument("--pitch-weight", type=float, default=0.04)
    ap.add_argument("--predicted-prosody-mel-weight", type=float, default=0.0)
    ap.add_argument("--predicted-prosody-delta-weight", type=float, default=0.0)
    ap.add_argument("--robust-prosody-mix", type=float, default=0.0)
    ap.add_argument("--robust-prosody-mel-weight", type=float, default=0.0)
    ap.add_argument("--robust-prosody-delta-weight", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--postnet-scale", type=float, default=0.10)
    ap.add_argument("--abs-frame-bins", type=int, default=512)
    ap.add_argument("--init-checkpoint", type=Path)
    ap.add_argument("--vocoder-checkpoint", type=Path)
    ap.add_argument("--vocoder-wav-weight", type=float, default=0.0)
    ap.add_argument("--vocoder-mel-weight", type=float, default=0.0)
    ap.add_argument("--save-interval", type=int, default=2000)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--preload-features", action="store_true", help="Cache decoded audio, mels, pitch, and token features in RAM before training.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--trainable",
        choices=["all", "duration", "predictors", "heads", "contextual", "group_duration", "decoder_adapt"],
        default="all",
    )
    ap.add_argument("--contextual-predictors", action="store_true")
    ap.add_argument("--group-duration-planner", action="store_true")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
