from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class HifiGanConfig:
    variant: str
    sample_rate: int = 24000
    n_fft: int = 1024
    hop_size: int = 256
    win_size: int = 1024
    num_mels: int = 80
    fmin: float = 0.0
    fmax: float = 12000.0
    resblock: str = "1"
    upsample_rates: tuple[int, ...] = (8, 8, 2, 2)
    upsample_kernel_sizes: tuple[int, ...] = (16, 16, 4, 4)
    upsample_initial_channel: int = 128
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    activation: str = "lrelu"
    conditioning_channels: int = 0


def make_config(variant: str) -> HifiGanConfig:
    if variant == "v2":
        return HifiGanConfig(variant="v2")
    if variant == "v2plus":
        return HifiGanConfig(variant="v2plus", upsample_initial_channel=160)
    if variant == "v2wide":
        return HifiGanConfig(variant="v2wide", upsample_initial_channel=176)
    if variant == "snake_v2mid":
        return HifiGanConfig(variant="snake_v2mid", upsample_initial_channel=144, activation="snake")
    if variant == "snake_v2balanced":
        return HifiGanConfig(variant="snake_v2balanced", upsample_initial_channel=160, activation="snake")
    if variant == "source_snake_v2balanced":
        return HifiGanConfig(
            variant="source_snake_v2balanced",
            upsample_initial_channel=160,
            activation="snake",
            conditioning_channels=5,
        )
    if variant == "v3":
        return HifiGanConfig(
            variant="v3",
            resblock="2",
            upsample_rates=(8, 8, 4),
            upsample_kernel_sizes=(16, 16, 8),
            upsample_initial_channel=256,
            resblock_kernel_sizes=(3, 5, 7),
            resblock_dilation_sizes=((1, 2), (2, 6), (3, 12)),
        )
    raise ValueError(f"Unknown variant: {variant}")


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class SnakeActivation(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.log_alpha.exp().clamp(1e-4, 100.0)
        return x + torch.sin(alpha * x).pow(2) / alpha


def make_activation(channels: int, activation: str) -> nn.Module:
    if activation == "snake":
        return SnakeActivation(channels)
    return nn.LeakyReLU(0.1)


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: tuple[int, ...], activation: str = "lrelu"):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilations
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))
                )
                for _ in dilations
            ]
        )
        self.acts1 = nn.ModuleList([make_activation(channels, activation) for _ in dilations])
        self.acts2 = nn.ModuleList([make_activation(channels, activation) for _ in dilations])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.acts1, self.acts2):
            y = a1(x)
            y = c1(y)
            y = a2(y)
            y = c2(y)
            x = x + y
        return x

    def remove_weight_norm(self) -> None:
        for layer in list(self.convs1) + list(self.convs2):
            remove_weight_norm(layer)


class ResBlock2(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: tuple[int, ...], activation: str = "lrelu"):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilations
            ]
        )
        self.acts = nn.ModuleList([make_activation(channels, activation) for _ in dilations])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, act in zip(self.convs, self.acts):
            y = act(x)
            y = conv(y)
            x = x + y
        return x

    def remove_weight_norm(self) -> None:
        for layer in self.convs:
            remove_weight_norm(layer)


class HifiGanGenerator(nn.Module):
    def __init__(self, cfg: HifiGanConfig):
        super().__init__()
        self.cfg = cfg
        self.num_kernels = len(cfg.resblock_kernel_sizes)
        self.num_upsamples = len(cfg.upsample_rates)
        self.conv_pre = weight_norm(
            nn.Conv1d(cfg.num_mels + cfg.conditioning_channels, cfg.upsample_initial_channel, 7, 1, padding=3)
        )
        self.ups = nn.ModuleList()
        self.up_acts = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        resblock_cls = ResBlock1 if cfg.resblock == "1" else ResBlock2
        for i, (rate, kernel) in enumerate(zip(cfg.upsample_rates, cfg.upsample_kernel_sizes)):
            in_ch = cfg.upsample_initial_channel // (2**i)
            out_ch = cfg.upsample_initial_channel // (2 ** (i + 1))
            self.up_acts.append(make_activation(in_ch, cfg.activation))
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        in_ch,
                        out_ch,
                        kernel,
                        rate,
                        padding=(kernel - rate) // 2,
                    )
                )
            )
            for k, d in zip(cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes):
                self.resblocks.append(resblock_cls(out_ch, k, d, cfg.activation))
        final_ch = cfg.upsample_initial_channel // (2 ** len(cfg.upsample_rates))
        self.post_act = make_activation(final_ch, cfg.activation)
        self.conv_post = weight_norm(nn.Conv1d(final_ch, 1, 7, 1, padding=3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for i, up in enumerate(self.ups):
            x = self.up_acts[i](x)
            x = up(x)
            xs = 0.0
            for j in range(self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = self.post_act(x)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self) -> None:
        remove_weight_norm(self.conv_pre)
        for up in self.ups:
            remove_weight_norm(up)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_post)


def extract_source_features(
    wav: torch.Tensor,
    cfg: HifiGanConfig,
    frames: int,
    dropout: float = 0.0,
    noise: float = 0.0,
) -> torch.Tensor:
    """Return low-rate F0/voicing features for source-conditioned generators."""
    pitch = torchaudio.functional.detect_pitch_frequency(
        wav.detach().cpu(),
        sample_rate=cfg.sample_rate,
        frame_time=cfg.hop_size / cfg.sample_rate,
        win_length=30,
    ).to(wav.device)
    if pitch.ndim == 1:
        pitch = pitch.unsqueeze(0)
    if pitch.shape[-1] < frames:
        pitch = F.pad(pitch, (0, frames - pitch.shape[-1]), value=0.0)
    pitch = pitch[..., :frames]
    voiced = ((pitch >= 55.0) & (pitch <= 420.0)).float()
    pitch = pitch.clamp(55.0, 420.0)
    log_f0 = ((torch.log(pitch) - math.log(140.0)) / 0.45).clamp(-3.0, 3.0) * voiced
    if noise > 0.0:
        log_f0 = (log_f0 + torch.randn_like(log_f0) * noise * voiced).clamp(-3.0, 3.0)
    jump = F.pad((log_f0[..., 1:] - log_f0[..., :-1]).abs(), (1, 0))
    confidence = torch.exp(-1.5 * jump) * voiced
    reconstructed_f0 = torch.exp(log_f0 * 0.45 + math.log(140.0))
    phase = torch.cumsum(2.0 * math.pi * reconstructed_f0 * (cfg.hop_size / cfg.sample_rate), dim=-1)
    source = torch.stack(
        [log_f0, voiced, confidence, torch.sin(phase) * confidence, torch.cos(phase) * confidence],
        dim=1,
    )
    if dropout > 0.0:
        # Drop the complete source sketch for some examples so inference remains
        # stable when predicted F0 confidence is poor.
        keep = (torch.rand(source.shape[0], 1, 1, device=source.device) >= dropout).to(source.dtype)
        source = source * keep
    return source


class DiscriminatorP(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
                weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
                weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
                weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
                weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            x = F.pad(x, (0, self.period - (t % self.period)), mode="reflect")
            t = x.shape[-1]
        x = x.view(b, c, t // self.period, self.period)
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in (2, 3, 5, 7, 11)])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList(
            [
                norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
                norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorS(True), DiscriminatorS(), DiscriminatorS()])
        self.meanpools = nn.ModuleList([nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for i, d in enumerate(self.discriminators):
            if i:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SpectrogramDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (32, 64, 128, 128)
        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch, stride in zip(channels, ((1, 2), (2, 2), (2, 2), (2, 1))):
            layers.append(weight_norm(nn.Conv2d(in_ch, out_ch, (5, 5), stride=stride, padding=(2, 2))))
            in_ch = out_ch
        self.convs = nn.ModuleList(layers)
        self.conv_post = weight_norm(nn.Conv2d(in_ch, 1, (3, 3), padding=(1, 1)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiResolutionSpectrogramDiscriminator(nn.Module):
    def __init__(self, fft_sizes: tuple[int, ...] = (256, 512, 1024), hop_sizes: tuple[int, ...] = (64, 128, 256), win_lengths: tuple[int, ...] = (256, 512, 1024)):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.discriminators = nn.ModuleList([SpectrogramDiscriminator() for _ in fft_sizes])

    def _features(self, wav: torch.Tensor, fft: int, hop: int, win_len: int) -> torch.Tensor:
        wav = wav.squeeze(1)
        window = torch.hann_window(win_len, device=wav.device)
        spec = torch.stft(wav, n_fft=fft, hop_length=hop, win_length=win_len, window=window, return_complex=True)
        mag = torch.log(spec.abs().clamp_min(1e-5))
        mean = mag.mean(dim=(1, 2), keepdim=True)
        std = mag.std(dim=(1, 2), keepdim=True).clamp_min(1e-4)
        return ((mag - mean) / std).unsqueeze(1)

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for disc, fft, hop, win_len in zip(self.discriminators, self.fft_sizes, self.hop_sizes, self.win_lengths):
            y_feat = self._features(y, fft, hop, win_len)
            y_hat_feat = self._features(y_hat, fft, hop, win_len)
            y_d_r, fmap_r = disc(y_feat)
            y_d_g, fmap_g = disc(y_hat_feat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class MelFrontend(nn.Module):
    def __init__(self, cfg: HifiGanConfig):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_size,
            hop_length=cfg.hop_size,
            f_min=cfg.fmin,
            f_max=cfg.fmax,
            n_mels=cfg.num_mels,
            power=1.0,
            center=True,
            norm="slaney",
            mel_scale="slaney",
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(self.mel(wav), min=1e-5))


def load_rows(path: Path, max_rows: int, min_seconds: float, max_seconds: float) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            audio = Path(str(row.get("target_audio") or ""))
            text = str(row.get("target_text") or "").strip()
            dur = float(row.get("target_duration_s") or 0.0)
            if audio.is_file() and text and min_seconds <= (dur or 4.0) <= max_seconds:
                rows.append({"audio": str(audio), "text": text, "duration": dur})
                if max_rows > 0 and len(rows) >= max_rows:
                    break
    if not rows:
        raise RuntimeError(f"No rows loaded from {path}")
    return rows


def load_audio(path: str, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    wav = wav.squeeze(0)
    return wav.clamp(-1, 1)


class AudioDataset(Dataset):
    def __init__(self, rows: list[dict], cfg: HifiGanConfig, segment_size: int, seed: int):
        self.rows = rows
        self.cfg = cfg
        self.segment_size = segment_size
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        wav = load_audio(self.rows[idx]["audio"], self.cfg.sample_rate)
        if wav.numel() >= self.segment_size:
            start = self.rng.randint(0, wav.numel() - self.segment_size)
            return wav[start : start + self.segment_size]
        return F.pad(wav, (0, self.segment_size - wav.numel()))


def feature_loss(fmap_r, fmap_g) -> torch.Tensor:
    loss = 0.0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss = loss + F.l1_loss(rl.detach(), gl)
    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs) -> torch.Tensor:
    loss = 0.0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        loss = loss + torch.mean((1 - dr) ** 2) + torch.mean(dg**2)
    return loss


def generator_loss(disc_outputs) -> torch.Tensor:
    loss = 0.0
    for dg in disc_outputs:
        loss = loss + torch.mean((1 - dg) ** 2)
    return loss


def stft_mag_loss(y_hat: torch.Tensor, y: torch.Tensor, fft_sizes: tuple[int, ...], hop_sizes: tuple[int, ...], win_lengths: tuple[int, ...]) -> torch.Tensor:
    # Multi-resolution spectral loss catches buzz/shimmer that can hide behind
    # mel loss, especially for a small generator near convergence.
    y_hat = y_hat.squeeze(1)
    y = y.squeeze(1)
    total = torch.zeros((), device=y.device)
    for fft, hop, win_len in zip(fft_sizes, hop_sizes, win_lengths):
        window = torch.hann_window(win_len, device=y.device)
        pred = torch.stft(y_hat, n_fft=fft, hop_length=hop, win_length=win_len, window=window, return_complex=True)
        target = torch.stft(y, n_fft=fft, hop_length=hop, win_length=win_len, window=window, return_complex=True)
        pred_mag = pred.abs().clamp_min(1e-7)
        target_mag = target.abs().clamp_min(1e-7)
        sc = torch.linalg.vector_norm(target_mag - pred_mag) / torch.linalg.vector_norm(target_mag).clamp_min(1e-7)
        log_mag = F.l1_loss(torch.log(pred_mag), torch.log(target_mag))
        total = total + sc + log_mag
    return total / max(1, len(fft_sizes))


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def jsonable_args(args: argparse.Namespace) -> dict:
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def save_checkpoint(
    path: Path,
    generator: nn.Module,
    mpd: nn.Module,
    msd: nn.Module,
    optim_g,
    optim_d,
    cfg: HifiGanConfig,
    step: int,
    args,
    mrsd: nn.Module | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
            "generator": generator.state_dict(),
            "mpd": mpd.state_dict(),
            "msd": msd.state_dict(),
            "optim_g": optim_g.state_dict(),
            "optim_d": optim_d.state_dict(),
            "config": asdict(cfg),
            "step": step,
            "args": jsonable_args(args),
            "generator_params": count_parameters(generator),
        }
    if mrsd is not None:
        payload["mrsd"] = mrsd.state_dict()
    torch.save(payload, tmp)
    tmp.replace(path)


def checkpoint_step(path: Path) -> int:
    stem = path.stem
    tail = stem.rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else -1


def prune_checkpoints(out_dir: Path, variant: str, keep: int) -> None:
    if keep <= 0:
        return
    numbered = [p for p in out_dir.glob(f"hifigan-{variant}-*.pt") if checkpoint_step(p) >= 0]
    numbered.sort(key=checkpoint_step, reverse=True)
    for old in numbered[keep:]:
        old.unlink(missing_ok=True)


def latest_checkpoint(out_dir: Path) -> Path | None:
    numbered = [p for p in out_dir.glob("hifigan-*-*.pt") if checkpoint_step(p) >= 0]
    if numbered:
        return max(numbered, key=checkpoint_step)
    ckpts = sorted(out_dir.glob("hifigan-*-latest.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0] if ckpts else None


def partial_load_state(module: nn.Module, state: dict[str, torch.Tensor]) -> tuple[int, int]:
    current = module.state_dict()
    patched: dict[str, torch.Tensor] = {}
    copied = 0
    skipped = 0
    for name, target in current.items():
        source = state.get(name)
        if source is None:
            skipped += 1
            continue
        if source.shape == target.shape:
            patched[name] = source
            copied += 1
            continue
        if source.ndim != target.ndim:
            skipped += 1
            continue
        value = target.clone()
        slices = tuple(slice(0, min(a, b)) for a, b in zip(target.shape, source.shape))
        value[slices] = source[slices].to(value.device, value.dtype)
        patched[name] = value
        copied += 1
    module.load_state_dict(patched, strict=False)
    return copied, skipped


def train(args: argparse.Namespace) -> None:
    torch.backends.cudnn.benchmark = True
    cfg = make_config(args.variant)
    device = torch.device(args.device)
    rows = load_rows(args.train_jsonl, args.max_rows, args.min_seconds, args.max_seconds)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    dataset = AudioDataset(rows, cfg, args.segment_size, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    mel_frontend = MelFrontend(cfg).to(device)
    generator = HifiGanGenerator(cfg).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    mrsd = MultiResolutionSpectrogramDiscriminator().to(device) if args.spec_disc_weight > 0.0 else None
    optim_g = torch.optim.AdamW(generator.parameters(), lr=args.lr, betas=(0.8, 0.99))
    disc_params = list(mpd.parameters()) + list(msd.parameters())
    if mrsd is not None:
        disc_params += list(mrsd.parameters())
    optim_d = torch.optim.AdamW(disc_params, lr=args.lr, betas=(0.8, 0.99))
    start_step = 0
    if args.init_checkpoint and not args.resume:
        ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        if args.partial_init:
            copied, skipped = partial_load_state(generator, ckpt["generator"])
            print(f"Partially initialized generator from {args.init_checkpoint}: copied={copied} skipped={skipped}")
        else:
            generator.load_state_dict(ckpt["generator"])
        if "mpd" in ckpt and "msd" in ckpt:
            mpd.load_state_dict(ckpt["mpd"])
            msd.load_state_dict(ckpt["msd"])
        if mrsd is not None and "mrsd" in ckpt:
            mrsd.load_state_dict(ckpt["mrsd"])
        can_load_disc_optim = mrsd is None or "mrsd" in ckpt
        if not args.partial_init and not args.reset_optim and "optim_g" in ckpt:
            optim_g.load_state_dict(ckpt["optim_g"])
        if not args.partial_init and not args.reset_optim and can_load_disc_optim and "optim_d" in ckpt:
            optim_d.load_state_dict(ckpt["optim_d"])
        for group in optim_g.param_groups:
            group["lr"] = args.lr
        for group in optim_d.param_groups:
            group["lr"] = args.lr
        start_step = int(ckpt.get("step") or 0)
        print(f"Initialized {args.init_checkpoint} at step {start_step}; lr={args.lr:g}")
    if args.resume:
        ckpt_path = latest_checkpoint(args.out_dir)
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            generator.load_state_dict(ckpt["generator"])
            mpd.load_state_dict(ckpt["mpd"])
            msd.load_state_dict(ckpt["msd"])
            if mrsd is not None and "mrsd" in ckpt:
                mrsd.load_state_dict(ckpt["mrsd"])
            optim_g.load_state_dict(ckpt["optim_g"])
            optim_d.load_state_dict(ckpt["optim_d"])
            for group in optim_g.param_groups:
                group["lr"] = args.lr
            for group in optim_d.param_groups:
                group["lr"] = args.lr
            start_step = int(ckpt.get("step") or 0)
            print(f"Resumed {ckpt_path} at step {start_step}; lr={args.lr:g}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prune_checkpoints(args.out_dir, args.variant, args.keep_checkpoints)
    (args.out_dir / "config.json").write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "args": jsonable_args(args),
                "rows": len(rows),
                "generator_params": count_parameters(generator),
                "mpd_params": count_parameters(mpd),
                "msd_params": count_parameters(msd),
                "mrsd_params": count_parameters(mrsd) if mrsd is not None else 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Variant: {args.variant}")
    print(f"Rows: {len(rows)}")
    print(f"Generator params: {count_parameters(generator):,} ({count_parameters(generator)/1_000_000:.3f}M)")
    print(f"MPD params: {count_parameters(mpd):,} MSD params: {count_parameters(msd):,} (training only)")
    if mrsd is not None:
        print(f"MRSD params: {count_parameters(mrsd):,} (training only)")
    if args.steps == 0:
        return

    step = start_step
    started = time.time()
    try:
        while step < args.steps:
            for wav in loader:
                step += 1
                y = wav.unsqueeze(1).to(device)
                with torch.no_grad():
                    mel = mel_frontend(wav.to(device))
                    if cfg.conditioning_channels:
                        source = extract_source_features(
                            wav.to(device),
                            cfg,
                            mel.shape[-1],
                            dropout=args.source_dropout,
                            noise=args.source_noise,
                        )
                        generator_input = torch.cat([mel, source], dim=1)
                    else:
                        generator_input = mel
                y_hat = generator(generator_input)
                common = min(y.shape[-1], y_hat.shape[-1])
                y = y[..., :common]
                y_hat = y_hat[..., :common]
                y_mel = mel_frontend(y.squeeze(1))
                y_hat_mel = mel_frontend(y_hat.squeeze(1))

                optim_d.zero_grad(set_to_none=True)
                y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_hat.detach())
                y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_hat.detach())
                loss_disc = discriminator_loss(y_df_hat_r, y_df_hat_g) + discriminator_loss(y_ds_hat_r, y_ds_hat_g)
                loss_spec_disc = torch.zeros((), device=device)
                if mrsd is not None:
                    y_dm_hat_r, y_dm_hat_g, _, _ = mrsd(y, y_hat.detach())
                    loss_spec_disc = discriminator_loss(y_dm_hat_r, y_dm_hat_g)
                    loss_disc = loss_disc + args.spec_disc_weight * loss_spec_disc
                loss_disc.backward()
                torch.nn.utils.clip_grad_norm_(disc_params, args.grad_clip)
                optim_d.step()

                optim_g.zero_grad(set_to_none=True)
                mel_loss = F.l1_loss(y_mel, y_hat_mel) * args.mel_weight
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_hat)
                loss_fm = feature_loss(fmap_f_r, fmap_f_g) + feature_loss(fmap_s_r, fmap_s_g)
                loss_gen = generator_loss(y_df_hat_g) + generator_loss(y_ds_hat_g)
                loss_spec_gen = torch.zeros((), device=device)
                loss_spec_fm = torch.zeros((), device=device)
                if mrsd is not None:
                    y_dm_hat_r, y_dm_hat_g, fmap_m_r, fmap_m_g = mrsd(y, y_hat)
                    loss_spec_gen = generator_loss(y_dm_hat_g)
                    loss_spec_fm = feature_loss(fmap_m_r, fmap_m_g)
                wav_l1 = F.l1_loss(y_hat, y) * args.wav_weight
                stft_loss = torch.zeros((), device=device)
                if args.stft_weight > 0.0:
                    stft_loss = stft_mag_loss(y_hat, y, (512, 1024, 2048), (128, 256, 512), (512, 1024, 2048)) * args.stft_weight
                loss_g = (
                    mel_loss
                    + args.fm_weight * loss_fm
                    + args.adv_weight * loss_gen
                    + wav_l1
                    + stft_loss
                    + args.spec_disc_weight * loss_spec_gen
                    + args.spec_fm_weight * loss_spec_fm
                )
                loss_g.backward()
                grad_g = torch.nn.utils.clip_grad_norm_(generator.parameters(), args.grad_clip)
                optim_g.step()

                if step == 1 or step % args.log_interval == 0:
                    elapsed = max(time.time() - started, 1e-6)
                    speed = (step - start_step) / elapsed
                    eta = (args.steps - step) / max(speed, 1e-6)
                    print(
                        f"step={step}/{args.steps} g={loss_g.item():.4f} d={loss_disc.item():.4f} "
                        f"mel={mel_loss.item():.4f} fm={loss_fm.item():.4f} adv={loss_gen.item():.4f} "
                        f"wav={wav_l1.item():.4f} stft={stft_loss.item():.4f} "
                        f"sd={loss_spec_disc.item():.4f} sfm={loss_spec_fm.item():.4f} sadv={loss_spec_gen.item():.4f} "
                        f"grad={float(grad_g):.3f} speed={speed:.3f} step/s eta={eta/60:.1f}m",
                        flush=True,
                    )
                if step % args.save_interval == 0 or step >= args.steps:
                    prune_checkpoints(args.out_dir, args.variant, max(args.keep_checkpoints - 1, 0))
                    save_checkpoint(args.out_dir / f"hifigan-{args.variant}-{step}.pt", generator, mpd, msd, optim_g, optim_d, cfg, step, args, mrsd)
                    save_checkpoint(args.out_dir / f"hifigan-{args.variant}-latest.pt", generator, mpd, msd, optim_g, optim_d, cfg, step, args, mrsd)
                if step >= args.steps:
                    break
    except KeyboardInterrupt:
        if step > start_step:
            save_checkpoint(args.out_dir / f"hifigan-{args.variant}-interrupt-{step}.pt", generator, mpd, msd, optim_g, optim_d, cfg, step, args, mrsd)
            save_checkpoint(args.out_dir / f"hifigan-{args.variant}-latest.pt", generator, mpd, msd, optim_g, optim_d, cfg, step, args, mrsd)
            print(f"Interrupted. Saved checkpoint at step {step}.", flush=True)
        raise
    save_checkpoint(args.out_dir / f"hifigan-{args.variant}-final.pt", generator, mpd, msd, optim_g, optim_d, cfg, step, args, mrsd)
    print(f"Done. {args.out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train exact-ish HiFi-GAN V2/V3 oracle vocoders on corrected Mark audio.")
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--variant",
        choices=["v2", "v2plus", "v2wide", "snake_v2mid", "snake_v2balanced", "source_snake_v2balanced", "v3"],
        required=True,
    )
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--min-seconds", type=float, default=1.0)
    ap.add_argument("--max-seconds", type=float, default=12.0)
    ap.add_argument("--segment-size", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--mel-weight", type=float, default=45.0)
    ap.add_argument("--wav-weight", type=float, default=1.0)
    ap.add_argument("--fm-weight", type=float, default=1.0)
    ap.add_argument("--adv-weight", type=float, default=1.0)
    ap.add_argument("--stft-weight", type=float, default=0.0)
    ap.add_argument("--spec-disc-weight", type=float, default=0.0, help="Training-only multi-resolution spectrogram adversarial weight.")
    ap.add_argument("--spec-fm-weight", type=float, default=0.0, help="Training-only spectrogram discriminator feature-matching weight.")
    ap.add_argument("--source-dropout", type=float, default=0.0, help="Probability of dropping source conditioning per training example.")
    ap.add_argument("--source-noise", type=float, default=0.0, help="Stddev of normalized log-F0 corruption for source conditioning.")
    ap.add_argument("--grad-clip", type=float, default=1000.0)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--save-interval", type=int, default=1000)
    ap.add_argument("--keep-checkpoints", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-checkpoint", type=Path)
    ap.add_argument("--partial-init", action="store_true", help="Slice-copy compatible generator weights from init-checkpoint into a resized generator.")
    ap.add_argument("--reset-optim", action="store_true", help="When initializing from a checkpoint, load model/discriminators but start fresh optimizers.")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
