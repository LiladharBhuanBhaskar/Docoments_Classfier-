import re

# ---------------- NORMALIZE ----------------
def normalize(text: str) -> list[str]:
    text = text.replace("\r", "\n")
    return [l.strip() for l in text.splitlines() if l.strip()]


# ---------------- PAN NUMBER ----------------
def extract_pan(lines):
    for l in lines:
        m = re.search(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', l)
        if m:
            return m.group()
    return None


# ---------------- DOB ----------------
def extract_dob(lines):
    for l in lines:
        m = re.search(r'\b\d{2}/\d{2}/\d{4}\b', l)
        if m:
            return m.group()
    return None


# ---------------- NAME & FATHER NAME ----------------
def extract_names(lines):

    def is_name_like(text):

        if len(text.split()) < 1:
            return False

        if re.search(r'\d', text):
            return False

        reject_words = [
            "date", "birth", "dob", "signature",
            "permanent", "account", "number",
            "gov", "india"
        ]

        if any(r in text.lower() for r in reject_words):
            return False

        return True

    name = None
    father_name = None

    # ---------------- LABEL BASED ----------------
    for i, line in enumerate(lines):

        low = line.lower()

        if "father" in low:
            for j in range(1, 4):
                if i + j < len(lines):
                    candidate = lines[i + j].strip()
                    if is_name_like(candidate):
                        father_name = candidate
                        break

        if "name" in low and "father" not in low:
            for j in range(1, 4):
                if i + j < len(lines):
                    candidate = lines[i + j].strip()
                    if is_name_like(candidate):
                        name = candidate
                        break

    # ---------------- POSITIONAL FALLBACK ----------------
    if not name or not father_name:

        dob_index = None

        for i, line in enumerate(lines):
            if re.search(r'\d{2}/\d{2}/\d{4}', line):
                dob_index = i
                break

        if dob_index is not None:

            # Father usually just above DOB
            if not father_name and dob_index - 1 >= 0:
                candidate = lines[dob_index - 1].strip()
                if is_name_like(candidate):
                    father_name = candidate

            # Name usually above father
            if not name and dob_index - 2 >= 0:
                candidate = lines[dob_index - 2].strip()
                if is_name_like(candidate):
                    name = candidate

    return name, father_name




# ---------------- MASTER ----------------
def parse_pan(ocr_text: str) -> dict:
    lines = normalize(ocr_text)

    name, father_name = extract_names(lines)

    return {
        "pan_number": extract_pan(lines),
        "name": name,
        "father_name": father_name,
        "dob": extract_dob(lines)
    }
