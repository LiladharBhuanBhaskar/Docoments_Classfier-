from __future__ import annotations

import re

_DL_NUMBER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b[A-Z]{2}\s?\d{2}\s?\d{4}\s?\d{5,7}\b", re.IGNORECASE),  # MH01 2014 0017867
    re.compile(r"\b[A-Z]{2}-?\d{2}\d{11,13}\b", re.IGNORECASE),
    re.compile(r"\b[A-Z]{2}\d{13,15}\b", re.IGNORECASE),
]


def _normalize_lines(text: str) -> list[str]:
    text = (text or "").replace("\r", "\n")
    return [l.strip() for l in text.splitlines() if l.strip()]


def extract_dl_number(text: str) -> str | None:
    upper = (text or "").upper()
    for pattern in _DL_NUMBER_PATTERNS:
        m = pattern.search(upper)
        if m:
            return m.group(0).replace(" ", "")
    return None


def _extract_date_after_label(text: str, label: str) -> str | None:
    pattern = re.compile(rf"{re.escape(label)}\s*:?\s*.*?(\d{{2}}[/-]\d{{2}}[/-]\d{{4}})", re.IGNORECASE)
    m = pattern.search(text or "")
    return m.group(1) if m else None


def extract_dob(text: str) -> str | None:
    for label in ["DOB", "DATE OF BIRTH", "D.O.B"]:
        value = _extract_date_after_label(text, label)
        if value:
            return value

    m = re.search(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b", text or "")
    return m.group(0) if m else None


def extract_issue_date(text: str) -> str | None:
    for label in ["DOI", "ISSUE", "ISSUE DATE", "DATE OF ISSUE", "D.O.I"]:
        value = _extract_date_after_label(text, label)
        if value:
            return value
    return None


def extract_valid_till(text: str) -> str | None:
    for label in ["VALID TILL", "VALIDITY", "VALID UPTO", "EXPIRY", "EXPIRY DATE"]:
        value = _extract_date_after_label(text, label)
        if value:
            return value
    return None


def extract_name(text: str) -> str | None:
    lines = _normalize_lines(text)
    for i, line in enumerate(lines):
        low = line.lower()
        if low in {"name", "name :", "name-"} or low.startswith("name"):
            if i + 1 < len(lines):
                candidate = lines[i + 1].strip()
                if candidate and not re.search(r"\d", candidate):
                    return candidate
    return None


def parse_driving_license(text: str) -> dict:
    return {
        "dl_number": extract_dl_number(text),
        "name": extract_name(text),
        "dob": extract_dob(text),
        "issue_date": extract_issue_date(text),
        "valid_till": extract_valid_till(text),
    }

