# coding=utf-8
"""Text normalization shared by data preparation, training and inference.

Use the same function everywhere the model sees text. Yoruba, Igbo and Hausa letters with diacritics (ọ, ẹ, ṣ, ị, ụ,
tone marks, ɓ, ɗ, ƙ) can be stored as one composed character or as a base letter plus combining marks, and look the
same on screen. The tokenizer sees different tokens for them, so the same word is spelled several ways to the model.

    from text_utils import normalize_text
    wavs, sr = tts.generate_custom_voice(text=normalize_text(sentence), ...)
"""
import re
import unicodedata

# zero-width space, soft hyphen, word joiner and byte-order mark: invisible, never part of a word. (Zero-width joiners
# are kept: they matter in some scripts.)
_INVISIBLE = dict.fromkeys(map(ord, "​­⁠﻿"), None)
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text):
    """Unicode NFC (composed form, marks in canonical order), invisible characters dropped, whitespace collapsed.

    Punctuation, case and apostrophes are left alone, so it never changes how a word is spelled.
    """
    text = unicodedata.normalize("NFC", text).translate(_INVISIBLE)
    return _WHITESPACE.sub(" ", text).strip()
