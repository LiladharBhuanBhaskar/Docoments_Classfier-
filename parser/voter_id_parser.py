import re


# -------- EPIC NUMBER --------
def extract_epic(text):

    candidates = re.findall(r'\b[A-Z]{3}[0-9]{7}\b', text.upper())

    if candidates:
        return candidates[0]

    return None


# -------- DOB --------
def extract_dob(text):

    match = re.search(r'\b\d{2}[/-]\d{2}[/-]\d{4}\b', text)

    return match.group() if match else None


# -------- GENDER --------
def extract_gender(text):

    if re.search(r'\bmale\b', text, re.I):
        return "MALE"

    if re.search(r'\bfemale\b', text, re.I):
        return "FEMALE"

    return None


# -------- NAME --------
def extract_name_voter(lines):

    for i, line in enumerate(lines):

        if "name" in line.lower():
            if i + 1 < len(lines):
                return lines[i+1].strip()

    return None


# -------- MASTER --------
def parse_voter_id(text):

    lines = text.split("\n")

    return {
        "epic_number": extract_epic(text),
        "name": extract_name_voter(lines),
        "dob": extract_dob(text),
        "gender": extract_gender(text)
     }
