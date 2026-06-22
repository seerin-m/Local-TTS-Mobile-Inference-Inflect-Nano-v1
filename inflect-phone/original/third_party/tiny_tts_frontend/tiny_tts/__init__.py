import os
import torch
import soundfile as sf
from tiny_tts.text.english import normalize_text, grapheme_to_phoneme
from tiny_tts.text import phonemes_to_ids
from tiny_tts.nn import commons
from tiny_tts.models.synthesizer import VoiceSynthesizer
from tiny_tts.text.symbols import symbols
from tiny_tts.utils.config import (
    SAMPLING_RATE, SEGMENT_FRAMES, ADD_BLANK, SPEC_CHANNELS,
    N_SPEAKERS, SPK2ID, MODEL_PARAMS,
)
from tiny_tts.infer import load_engine

class TinyTTS:
    def __init__(self, checkpoint_path=None, device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        if checkpoint_path is None:
            # Look for default checkpoint in pacakage
            pkg_dir = os.path.dirname(os.path.abspath(__file__))
            default_ckpt = os.path.join(os.path.dirname(pkg_dir), "checkpoints", "G.pth")
            # 2. Check HuggingFace Cache / Download
            if not os.path.exists(default_ckpt):
                try:
                    from huggingface_hub import hf_hub_download
                    print("Downloading/Loading checkpoint from Hugging Face Hub (backtracking/tiny-tts)...")
                    default_ckpt = hf_hub_download(repo_id="backtracking/tiny-tts", filename="G.pth")
                except ImportError:
                    raise ImportError("huggingface_hub is required to auto-download the model. Run: pip install huggingface_hub")
                except Exception as e:
                    raise ValueError(f"Failed to download checkpoint from Hugging Face: {e}")

            checkpoint_path = default_ckpt
                
        self.model = load_engine(checkpoint_path, self.device)

    def speak(self, text, output_path="output.wav", speaker="MALE", speed=1.0):
        """Synthesize text to speech and save to output_path."""
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

        x = torch.LongTensor(phone_ids).unsqueeze(0).to(self.device)
        x_lengths = torch.LongTensor([len(phone_ids)]).to(self.device)
        tone = torch.LongTensor(tone_ids).unsqueeze(0).to(self.device)
        language = torch.LongTensor(lang_ids).unsqueeze(0).to(self.device)

        # Speaker ID
        if speaker not in SPK2ID:
            print(f"Warning: Speaker '{speaker}' not found, using ID 0. Available: {list(SPK2ID.keys())}")
            sid = torch.LongTensor([0]).to(self.device)
        else:
            sid = torch.LongTensor([SPK2ID[speaker]]).to(self.device)

        # BERT features (disabled - using zero tensors)
        bert = torch.zeros(1024, len(phone_ids)).to(self.device).unsqueeze(0)
        ja_bert = torch.zeros(768, len(phone_ids)).to(self.device).unsqueeze(0)

        # speed > 1.0 = faster speech, < 1.0 = slower speech
        length_scale = 1.0 / speed

        with torch.no_grad():
            audio, *_ = self.model.infer(
                x, x_lengths, sid, tone, language, bert, ja_bert,
                noise_scale=0.667,
                noise_scale_w=0.8,
                length_scale=length_scale
            )

        audio_np = audio[0, 0].cpu().numpy()
        sf.write(output_path, audio_np, SAMPLING_RATE)
        print(f"Saved audio to {output_path}")
        return audio_np
