from __future__ import annotations

import re


_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": "",
        "\u201d": "",
        "\u2014": ",",
        "\u2013": ",",
        ";": ",",
        ":": ",",
        "\n": ".",
    }
)


def clean_tinytts_text(text: str) -> str:
    """Normalize text into punctuation TinyTTS actually has symbols for."""
    text = str(text).translate(_QUOTE_TRANSLATION)
    text = text.replace("...", "…")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?…])", r"\1", text)
    text = re.sub(r"([,.!?…]){2,}", r"\1", text)
    return text
