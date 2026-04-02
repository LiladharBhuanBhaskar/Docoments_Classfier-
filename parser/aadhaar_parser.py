import re

from utils.verhoeff import verhoeff_validate

_LATIN_RE = re.compile(r"[A-Za-z]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def normalize(text: str) -> list[str]:
    text = (text or "").replace("\r", "\n")
    return [l.strip() for l in text.splitlines() if l.strip()]


def _clean_text(value: str) -> str:
    value = re.sub(r"[^A-Za-z\u0900-\u097F0-9,/\- ]", " ", value or "")
    return re.sub(r"\s+", " ", value).strip()


def _clean_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z\u0900-\u097F ]", " ", value or "")
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(
        r"^(?:name|naam|नाम|father name|father s name|पिता|पिता का नाम)\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return value.strip()


def _has_script(value: str, script: str) -> bool:
    if script == "hindi":
        return bool(_DEVANAGARI_RE.search(value or ""))
    return bool(_LATIN_RE.search(value or ""))


def _is_valid_name(value: str) -> bool:
    if not value or len(value.split()) < 2:
        return False
    if re.search(r"\d", value):
        return False

    blacklist = {
        "government", "india", "aadhaar", "authority", "enrollment", "number", "uidai",
        "address", "dob", "male", "female", "year", "birth", "download", "issue",
        "सरकार", "भारत", "आधार", "पता", "जन्म", "पुरुष", "महिला",
    }
    low = value.lower()
    if any(token in low for token in blacklist):
        return False
    return True


def _identity_anchors(lines: list[str]) -> list[int]:
    anchors: list[int] = []
    for i, line in enumerate(lines):
        upper = line.upper()
        if any(
            token in upper or token in line
            for token in ("DOB", "DATE OF BIRTH", "MALE", "FEMALE", "जन्म", "पुरुष", "महिला")
        ):
            anchors.append(i)
    return anchors


def _extract_name_by_script(lines: list[str], script: str) -> str | None:
    anchors = _identity_anchors(lines)
    if not anchors:
        return None

    for anchor in anchors:
        for offset in range(1, 7):
            idx = anchor - offset
            if idx < 0:
                continue
            candidate = _clean_name(lines[idx])
            if _is_valid_name(candidate) and _has_script(candidate, script):
                return candidate
    return None


def extract_name_english(lines: list[str]) -> str | None:
    return _extract_name_by_script(lines, "english")


def extract_name_hindi(lines: list[str]) -> str | None:
    return _extract_name_by_script(lines, "hindi")


def extract_name(lines: list[str]) -> str | None:
    return extract_name_english(lines) or extract_name_hindi(lines)


def extract_aadhaar(lines: list[str]) -> str | None:
    candidates: list[str] = []
    for line in lines:
        if "VID" in line.upper():
            continue
        candidates.extend(re.findall(r"\b\d{4}\s\d{4}\s\d{4}\b", line))

    if not candidates:
        return None

    valid = [candidate for candidate in candidates if verhoeff_validate(candidate)]
    if not valid:
        return "Aadhar Card Fake"
    return max(set(valid), key=valid.count)


def extract_vid(lines: list[str]) -> str | None:
    for line in lines:
        m = re.search(r"\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b", line)
        if m and "VID" in line.upper():
            return m.group()
    return None


def extract_dob(lines: list[str]) -> str | None:
    for line in lines:
        m = re.search(r"\b\d{2}/\d{2}/\d{4}\b", line)
        if not m:
            continue
        upper = line.upper()
        if "DOB" in upper or "DATE OF BIRTH" in upper or "जन्म" in line:
            return m.group()
    return None


def extract_gender(lines: list[str]) -> str | None:
    for line in lines:
        upper = line.upper()
        if "FEMALE" in upper or "महिला" in line:
            return "FEMALE"
        if "MALE" in upper or "पुरुष" in line:
            return "MALE"
    return None


def extract_mobile(lines: list[str]) -> str | None:
    for line in lines:
        m = re.search(r"\b[6-9]\d{9}\b", line)
        if m:
            return m.group()
    return None


def _address_lines(lines: list[str]) -> list[str]:
    address_lines: list[str] = []
    capture = False

    for line in lines:
        upper = line.upper()
        if "ADDRESS" in upper or "पता" in line:
            capture = True
            continue

        if not capture:
            continue

        if any(token in upper for token in ("UIDAI", "HELP", "WWW", "DOWNLOAD", "VID", "AADHAAR", "ISSUE DATE")):
            break
        address_lines.append(line)

    return address_lines


def _clean_address_value(value: str, script: str) -> str | None:
    cleaned = _clean_text(value)
    cleaned = re.sub(r"\b\d{4}\s\d{4}\s\d{4}\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
    if not cleaned:
        return None
    if script == "hindi" and not _DEVANAGARI_RE.search(cleaned):
        return None
    if script == "english" and not _LATIN_RE.search(cleaned):
        return None
    return cleaned if len(cleaned) > 8 else None


def extract_address_english(lines: list[str]) -> str | None:
    filtered = [line for line in _address_lines(lines) if _LATIN_RE.search(line)]
    return _clean_address_value(" ".join(filtered), "english")


def extract_address_hindi(lines: list[str]) -> str | None:
    filtered = [line for line in _address_lines(lines) if _DEVANAGARI_RE.search(line)]
    return _clean_address_value(" ".join(filtered), "hindi")


def extract_address(lines: list[str]) -> str | None:
    return extract_address_english(lines) or extract_address_hindi(lines)


def extract_father_name(lines: list[str]) -> str | None:
    text = "\n".join(lines)
    match = re.search(r"\b(?:S/O|D/O|C/O)\s*:?\s*([A-Za-z\u0900-\u097F ]+)", text, re.IGNORECASE)
    if match:
        candidate = _clean_name(match.group(1))
        return candidate if _is_valid_name(candidate) else None
    return None


def extract_pincode(lines: list[str]) -> str | None:
    for line in lines:
        m = re.search(r"\b\d{6}\b", line)
        if m:
            return m.group()
    return None


def parse_aadhaar(ocr_text: str) -> dict:
    lines = normalize(ocr_text)

    name_english = extract_name_english(lines)
    name_hindi = extract_name_hindi(lines)
    father_name = extract_father_name(lines)
    address_english = extract_address_english(lines)
    address_hindi = extract_address_hindi(lines)

    father_name_english = father_name if father_name and _LATIN_RE.search(father_name) else None
    father_name_hindi = father_name if father_name and _DEVANAGARI_RE.search(father_name) else None

    return {
        "aadhaar_number": extract_aadhaar(lines),
        "vid": extract_vid(lines),
        "name": name_english or name_hindi,
        "name_english": name_english,
        "name_hindi": name_hindi,
        "dob": extract_dob(lines),
        "gender": extract_gender(lines),
        "mobile": extract_mobile(lines),
        "father_name": father_name,
        "father_name_english": father_name_english,
        "father_name_hindi": father_name_hindi,
        "address": address_english or address_hindi,
        "address_english": address_english,
        "address_hindi": address_hindi,
        "pincode": extract_pincode(lines),
    }
