from __future__ import annotations
import sys
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent / "original"
VENDORED_FRONTEND = REPO_ROOT / "third_party" / "tiny_tts_frontend"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(VENDORED_FRONTEND))

from tiny_tts.nn import commons
from tiny_tts.text import phonemes_to_ids
from tiny_tts.text.english import grapheme_to_phoneme, normalize_text
from tiny_tts.utils import ADD_BLANK
from inflect_nano.text_cleaning import clean_tinytts_text

app = FastAPI()

class TextIn(BaseModel):
    text: str

@app.post("/phonemize")
def phonemize(body: TextIn):
    cleaned = clean_tinytts_text(body.text)
    normalized = normalize_text(cleaned)
    phones, tones, _ = grapheme_to_phoneme(normalized)
    phone_ids, tone_ids, lang_ids = phonemes_to_ids(phones, tones, "EN")
    if ADD_BLANK:
        phone_ids = commons.insert_blanks(phone_ids, 0)
        tone_ids = commons.insert_blanks(tone_ids, 0)
        lang_ids = commons.insert_blanks(lang_ids, 0)
    speaker_id = 0  # "mark" speaker
    return {
        "phone":   list(phone_ids),
        "tone":    list(tone_ids),
        "lang":    list(lang_ids),
        "speaker": [speaker_id]
    }
