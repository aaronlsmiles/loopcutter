"""Key per track: the file's own tag first (Mixed In Key writes it), keyfinder-cli second."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable

import mutagen

from .keys import Key, KeyParseError, parse_key

_CAMELOT_IN_TEXT = re.compile(r"\b(1[0-2]|[1-9])\s*([AB])\b", re.IGNORECASE)


@dataclass(frozen=True)
class KeyResult:
    key: Key | None
    source: str
    alternative: Key | None = None

    @property
    def disagrees(self) -> bool:
        return self.alternative is not None


def key_from_tags(path) -> tuple[Key | None, str]:
    tags = getattr(mutagen.File(str(path)), "tags", None)
    if not tags:
        return None, ""
    candidates: list[tuple[str, str]] = []
    if hasattr(tags, "getall"):
        candidates += [("tag", str(f.text[0])) for f in tags.getall("TKEY") if f.text]
        candidates += [("comment", str(f.text[0])) for f in tags.getall("COMM") if f.text]
    else:
        candidates += [("tag", str(tags[n][0])) for n in ("initialkey", "key") if n in tags]
    for source, text in candidates:
        match = _CAMELOT_IN_TEXT.search(text) if source == "comment" else None
        if source == "comment" and not match:
            continue
        try:
            return parse_key(match.group(0) if match else text), source
        except KeyParseError:
            continue
    return None, ""


def key_from_keyfinder(path) -> Key | None:
    binary = shutil.which("keyfinder-cli")
    if not binary:
        return None
    run = subprocess.run([binary, "-n", "camelot", str(path)], capture_output=True, text=True)
    if run.returncode != 0:
        return None
    try:
        return parse_key(run.stdout.strip())
    except KeyParseError:
        return None


def detect_key(path, keyfinder: Callable = key_from_keyfinder) -> KeyResult:
    tagged, source = key_from_tags(path)
    found = keyfinder(path)
    if tagged is None:
        return KeyResult(found, "keyfinder" if found else "")
    return KeyResult(tagged, source, found if found is not None and found != tagged else None)
