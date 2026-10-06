"""Lightweight reader for OpenFOAM dictionary files.

Used only for fast, read-only summaries. Edits always go through OpenFOAM's own
`foamDictionary` so that writes are exactly as robust as OpenFOAM itself.

Values are returned as raw strings (tokens joined with single spaces); sub-dictionaries
become nested dicts. Directives such as `#include "x"` are kept as keys with value None.
"""

from __future__ import annotations

import mmap
import re
from pathlib import Path

_PUNCT = set("{}();[]")
_LIST_HEAD = re.compile(r"nonuniform\s+List<\w+>\s+(\d+)")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    i, n = 0, len(text)
    at_line_start = True
    while i < n:
        c = text[i]
        if c == "\n":
            at_line_start = True
            i += 1
            continue
        if c.isspace():
            i += 1
            continue
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if text.startswith("#{", i):
            j = text.find("#}", i + 2)
            j = n if j < 0 else j + 2
            tokens.append(text[i:j])
            i = j
            at_line_start = False
            continue
        if c == "#" and at_line_start:
            # A directive (#include, #includeFunc, #remove, ...) occupies the rest of the line.
            j = text.find("\n", i)
            j = n if j < 0 else j
            tokens.append("\0" + text[i:j].strip())
            i = j
            continue
        at_line_start = False
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            tokens.append(text[i : j + 1])
            i = j + 1
            continue
        if c in _PUNCT:
            tokens.append(c)
            i += 1
            continue
        j = i
        while j < n and not text[j].isspace() and text[j] not in _PUNCT and not text.startswith("//", j):
            if text[j] == "\n":
                break
            j += 1
        tokens.append(text[i:j])
        i = j
    return tokens


def _join(tokens: list[str]) -> str:
    out = ""
    for t in tokens:
        if not out or out.endswith(("(", "[")) or t in (")", "]"):
            out += t
        else:
            out += " " + t
    return out


def _parse_dict(tokens: list[str], pos: int, top: bool) -> tuple[dict, int]:
    result: dict = {}
    n = len(tokens)
    while pos < n:
        tok = tokens[pos]
        if tok == "}":
            if top:
                pos += 1
                continue
            return result, pos + 1
        if tok == ";":
            pos += 1
            continue
        if tok.startswith("\0"):
            key = tok[1:]
            pos += 1
            # `#includeFunc name` may continue with a parenthesised argument list on following lines.
            if pos < n and tokens[pos] == "(" and "(" not in key:
                depth, start = 0, pos
                while pos < n:
                    depth += tokens[pos] == "("
                    depth -= tokens[pos] == ")"
                    pos += 1
                    if depth == 0:
                        break
                key = f"{key}{_join(tokens[start:pos])}"
            result[key] = None
            continue
        key = tok
        pos += 1
        if pos < n and tokens[pos] == "{":
            sub, pos = _parse_dict(tokens, pos + 1, top=False)
            result[key] = sub
            continue
        depth = 0
        start = pos
        while pos < n:
            t = tokens[pos]
            if t in ("(", "[", "{"):
                depth += 1
            elif t in (")", "]", "}"):
                if depth == 0:
                    break
                depth -= 1
            elif t == ";" and depth == 0:
                break
            pos += 1
        result[key] = _join(tokens[start:pos])
        if pos < n and tokens[pos] == ";":
            pos += 1
    return result, pos


def parse(text: str, elide: bool = True) -> dict:
    """Parse dictionary text into nested dicts of raw string values.

    With `elide`, large `nonuniform List<T> N (...)` payloads are replaced by a short placeholder.
    """
    if elide:
        text = _elide_lists(text)
    result, _ = _parse_dict(tokenize(text), 0, top=True)
    return result


def _elide_lists(text: str, threshold: int = 50) -> str:
    """Replace large `nonuniform List<T> N (...)` payloads with a placeholder to keep parsing fast."""
    out = []
    pos = 0
    for m in _LIST_HEAD.finditer(text):
        if m.start() < pos:
            continue
        count = int(m.group(1))
        if count < threshold:
            continue
        open_at = text.find("(", m.end())
        if open_at < 0:
            break
        depth, j = 0, open_at
        while j < len(text):
            ch = text[j]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append(text[pos : m.start()])
        out.append(f"nonuniform<{count}_values>")
        pos = j + 1
    out.append(text[pos:])
    return "".join(out)


def read(path: Path, max_bytes: int = 8_000_000) -> dict:
    """Parse a dictionary file. Very large field files are parsed from `boundaryField` on."""
    size = path.stat().st_size
    if size <= max_bytes:
        return parse(path.read_text(errors="replace"))
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        at = mm.find(b"\nboundaryField")
        head = mm[: min(size, 4096)].decode(errors="replace")
        tail = mm[at:].decode(errors="replace") if at >= 0 and size - at <= max_bytes else ""
    data = parse(head.split("internalField")[0] + tail)
    data.setdefault("internalField", "<large field, not parsed>")
    return data


def get(data: dict, path: str, default=None):
    """Fetch `a/b/c` from nested dicts."""
    cur = data
    for part in path.split("/"):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def first_word(value) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value.split()[0].strip('"')
