"""Helpers for the text shown in the approval dialog."""

from __future__ import annotations

import unicodedata

# Control, format (bidi overrides, zero-width characters) and line/paragraph separators.
_HIDDEN_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})
_ALLOWED = frozenset({"\n", "\t"})


def reveal_hidden(text: str) -> str:
    """Make invisible or reordering characters visible, e.g. U+202E -> ``<U+202E>``.

    Without this, a right-to-left override could make ``evil\\u202etxt.exe`` read as
    ``evilexe.txt`` in a confirmation dialog, or a zero-width character could hide
    part of a command from the user.
    """
    if text.isascii() and text.isprintable():
        return text
    return "".join(
        f"<U+{ord(ch):04X}>" if ch not in _ALLOWED and unicodedata.category(ch) in _HIDDEN_CATEGORIES else ch
        for ch in text
    )
