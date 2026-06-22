import os
import sys
import re
import torch
import soundfile as sf
import argparse
from tiny_tts.text.english import normalize_text, grapheme_to_phoneme
from tiny_tts.text import phonemes_to_ids
from tiny_tts.nn import commons
from tiny_tts.models import VoiceSynthesizer
from tiny_tts.text.symbols import symbols
from tiny_tts.utils import (
    SAMPLING_RATE, SEGMENT_FRAMES, ADD_BLANK, SPEC_CHANNELS,
    N_SPEAKERS, SPK2ID, MODEL_PARAMS,
)


def load_engine(checkpoint_path, device='cuda'):
    print(f"Loading model from {checkpoint_path}")
    net_g = VoiceSynthesizer(
        len(symbols),
        SPEC_CHANNELS,
        SEGMENT_FRAMES,
        n_speakers=N_SPEAKERS,
        **MODEL_PARAMS
    ).to(device)

    # Count model parameters
    total_params = sum(p.numel() for p in net_g.parameters())
    trainable_params = sum(p.numel() for p in net_g.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model']

    # Remove module. prefix and filter shape mismatches
    model_state = net_g.state_dict()
    new_state_dict = {}
    skipped = []
    for k, v in state_dict.items():
        key = k[7:] if k.startswith('module.') else k
        if key in model_state:
            if v.shape == model_state[key].shape:
                new_state_dict[key] = v
            else:
                skipped.append(f"{key}: ckpt{v.shape} vs model{model_state[key].shape}")
        else:
            new_state_dict[key] = v

    if skipped:
        print(f"Skipped {len(skipped)} mismatched keys:")
        for s in skipped[:5]:
            print(f"  {s}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped)-5} more")

    net_g.load_state_dict(new_state_dict, strict=False)
    net_g.eval()

    # Fold weight_norm into weight tensors for faster inference (~18% speedup)
    net_g.dec.remove_weight_norm()

    return net_g


def synthesize(text, output_path, model, speaker="MALE", device='cuda', speed=1.0):
    print(f"Synthesizing: {text}")

    # Normalize text
    normalized = normalize_text(text)

    # Phonemize
    phones, tones, word2ph = grapheme_to_phoneme(normalized)

    # Convert to sequence
    phone_ids, tone_ids, lang_ids = phonemes_to_ids(phones, tones, "EN")

    # Add blanks
    if ADD_BLANK:
        phone_ids = commons.insert_blanks(phone_ids, 0)
        tone_ids = commons.insert_blanks(tone_ids, 0)
        lang_ids = commons.insert_blanks(lang_ids, 0)

    x = torch.LongTensor(phone_ids).unsqueeze(0).to(device)
    x_lengths = torch.LongTensor([len(phone_ids)]).to(device)
    tone = torch.LongTensor(tone_ids).unsqueeze(0).to(device)
    language = torch.LongTensor(lang_ids).unsqueeze(0).to(device)

    # Speaker ID
    if speaker not in SPK2ID:
        print(f"Warning: Speaker {speaker} not found, using ID 0")
        sid = torch.LongTensor([0]).to(device)
    else:
        sid = torch.LongTensor([SPK2ID[speaker]]).to(device)

    # BERT features (disabled - using zero tensors)
    bert = torch.zeros(1024, len(phone_ids)).to(device).unsqueeze(0)
    ja_bert = torch.zeros(768, len(phone_ids)).to(device).unsqueeze(0)

    # speed > 1.0 = faster speech, < 1.0 = slower speech
    length_scale = 1.0 / speed

    with torch.no_grad():
        audio, *_ = model.infer(
            x, x_lengths, sid, tone, language, bert, ja_bert,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=length_scale
        )

    audio = audio[0, 0].cpu().numpy()
    sf.write(output_path, audio, SAMPLING_RATE)
    print(f"Saved audio to {output_path}")


def get_latest_checkpoint(checkpoint_dir):
    """Finds the latest G_*.pth checkpoint in the given directory."""
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('G_') and f.endswith('.pth')]
    if not checkpoints:
        return None

    def get_step(filename):
        match = re.search(r'_(\d+)\.pth', filename)
        return int(match.group(1)) if match else -1

    latest_ckpt = max(checkpoints, key=get_step)
    return os.path.join(checkpoint_dir, latest_ckpt)


def main():
    parser = argparse.ArgumentParser(description="TinyTTS — English Text-to-Speech Inference")
    parser.add_argument("--text", "-t", type=str, default="The weather is nice today, and I feel very relaxed.", help="Text to synthesize")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Path to checkpoint. Auto-downloads if not provided.")
    parser.add_argument("--output", "-o", type=str, default="output.wav", help="Output audio file path")
    parser.add_argument("--speaker", "-s", type=str, default="MALE", help="Speaker ID")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (1.0=normal, 1.5=faster, 0.7=slower)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    if args.checkpoint is None:
        try:
            from huggingface_hub import hf_hub_download
            print("Downloading/Loading checkpoint from Hugging Face Hub (backtracking/tiny-tts)...")
            args.checkpoint = hf_hub_download(repo_id="backtracking/tiny-tts", filename="G.pth")
        except ImportError:
            print("Error: huggingface_hub is required for auto-download. Run: pip install huggingface_hub")
            sys.exit(1)
        except Exception as e:
            print(f"Error downloading checkpoint: {e}")
            sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint or directory not found at {args.checkpoint}")
        sys.exit(1)

    if os.path.isdir(args.checkpoint):
        latest_ckpt = get_latest_checkpoint(args.checkpoint)
        if not latest_ckpt:
            print(f"Error: No G_*.pth checkpoints found in directory {args.checkpoint}")
            sys.exit(1)
        args.checkpoint = latest_ckpt
        print(f"Auto-detected latest checkpoint: {args.checkpoint}")

    # Extract step from checkpoint filename
    ckpt_basename = os.path.basename(args.checkpoint)
    match = re.search(r'_(\d+)\.pth', ckpt_basename)
    step_str = match.group(1) if match else "unknown"

    # Save to output folder
    out_dir = "infer_outputs"
    os.makedirs(out_dir, exist_ok=True)

    out_name = os.path.basename(args.output)
    name, ext = os.path.splitext(out_name)
    model = load_engine(args.checkpoint, args.device)

    if args.speaker.lower() == "all":
        if not SPK2ID:
            print("Error: No speakers found")
            sys.exit(1)
        print(f"Synthesizing for all {len(SPK2ID)} speakers...")
        for spk in SPK2ID.keys():
            final_output = os.path.join(out_dir, f"{name}_step{step_str}_spk{spk}{ext}")
            synthesize(args.text, final_output, model, speaker=spk, device=args.device, speed=args.speed)
    else:
        final_output = os.path.join(out_dir, f"{name}_step{step_str}_spk{args.speaker}{ext}")
        synthesize(args.text, final_output, model, speaker=args.speaker, device=args.device, speed=args.speed)

if __name__ == "__main__":
    main()
