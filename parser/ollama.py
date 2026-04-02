from __future__ import annotations

import json
from typing import Any
import ollama
import re

DEFAULT_OLLAMA_MODEL = "mistral:latest"

DEFAULT_CHAT_PROMPT_TEMPLATE = """You are a helpful assistant.

User message:
{message}
"""


def build_prompt(prompt_template: str, *, message: str) -> str:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")
    try:
        return prompt_template.format(message=message)
    except Exception as e:
        raise ValueError("Invalid prompt_template; expected a '{message}' placeholder") from e


def chat_with_ollama(
    message: str,
    *,
    model: str = DEFAULT_OLLAMA_MODEL,
    prompt_template: str = DEFAULT_CHAT_PROMPT_TEMPLATE,
    system_prompt: str | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    Sends a prompt-template-rendered message to an Ollama chat model.

    Returns a JSON-serializable dict containing:
      - final_prompt: the final prompt string sent as the user message
      - model_response: the model's response text
    """

    final_prompt = build_prompt(prompt_template, message=message)

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": final_prompt})

    final_options: dict[str, Any] = {"temperature": 0}
    if options:
        final_options.update(options)

    try:
        raw = ollama.chat(model=model, messages=messages, options=final_options)
    except Exception as e:
        raise RuntimeError(f"Ollama chat failed: {e}") from e

    def _get(value: Any, key: str) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            return value.get(key)
        if hasattr(value, key):
            return getattr(value, key)
        getter = getattr(value, "get", None)
        if callable(getter):
            try:
                return getter(key)
            except Exception:
                return None
        return None

    msg = _get(raw, "message")
    content = _get(msg, "content")
    if content is None:
        content = _get(raw, "response")

    if not isinstance(content, str):
        raw_type = type(raw).__name__
        msg_type = type(msg).__name__ if msg is not None else "None"
        raise RuntimeError(f"Unexpected response format from ollama.chat() (raw={raw_type}, message={msg_type})")

    return {"final_prompt": final_prompt, "model_response": content}


def build_document_assistant_prompt(*, message: str, document_json: dict[str, Any]) -> str:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")
    if not isinstance(document_json, dict):
        raise ValueError("document_json must be a dict")

    document_json_str = json.dumps(document_json, ensure_ascii=False, indent=2)

    return (
        "You are a document assistant.\n\n"
        "Below is the extracted document data in JSON format:\n\n"
        f"{document_json_str}\n\n"
        "Answer the user's question strictly using the above JSON data.\n"
        "Do not guess.\n"
        "Do not use external knowledge.\n"
        "If the requested information is not present, respond with:\n"
        '"Information not available in the document."\n\n'
        "User question:\n"
        f"{message}\n"
    )


def answer_question_from_document(
    message: str,
    document_json: dict[str, Any],
    *,
    model: str = DEFAULT_OLLAMA_MODEL,
    options: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    Answers a user question strictly using the provided document JSON.

    Returns:
      - final_prompt: the full prompt sent to the model
      - model_response: the model's response text
    """

    final_prompt = build_document_assistant_prompt(message=message, document_json=document_json)

    final_options: dict[str, Any] = {"temperature": 0}
    if options:
        final_options.update(options)

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": final_prompt}],
            options=final_options,
        )
    except Exception as e:
        raise RuntimeError(f"Ollama chat failed: {e}") from e

    try:
        model_response: Any = response["message"]["content"]
    except Exception as e:
        raise RuntimeError("Unexpected response format from ollama.chat()") from e

    if not isinstance(model_response, str):
        raise RuntimeError("Unexpected response format from ollama.chat()")

    return {"final_prompt": final_prompt, "model_response": model_response}


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _loads_first_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_code_fences(text)

    # Fix missing commas between JSON fields
    cleaned = re.sub(
        r'("\s*\n\s*")', '",\n  "', cleaned
    )

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        candidate = cleaned[start:end + 1]

        # Try fixing common comma issue again
        candidate = re.sub(r'("\s*\n\s*")', '",\n  "', candidate)

        parsed = json.loads(candidate)

    if not isinstance(parsed, dict):
        raise ValueError("Model returned JSON but it wasn't an object")

    return parsed


_LATIN_RE = re.compile(r"[A-Za-z]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def _clean_optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _ensure_bilingual_fields(data: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    for field in fields:
        english_key = f"{field}_english"
        hindi_key = f"{field}_hindi"

        base_value = _clean_optional_text(data.get(field))
        english_value = _clean_optional_text(data.get(english_key))
        hindi_value = _clean_optional_text(data.get(hindi_key))

        if base_value:
            if _LATIN_RE.search(base_value) and not english_value:
                english_value = base_value
            if _DEVANAGARI_RE.search(base_value) and not hindi_value:
                hindi_value = base_value

        data[field] = base_value or english_value or hindi_value
        data[english_key] = english_value
        data[hindi_key] = hindi_value

    return data

def parse_with_ollama(normalized_text: str, doc_type: str, *, model: str = DEFAULT_OLLAMA_MODEL) -> dict[str, Any]:
    """
    Parses OCR text into a structured JSON object using Ollama.

    Supported doc_type values: Aadhaar, PAN, Voter ID, Driving License.
    """

    schema_map: dict[str, dict[str, Any]] = {
        "AADHAAR": {
            "name": None,
            "name_english": None,
            "name_hindi": None,
            "dob": None,
            "gender": None,
            "aadhaar_number": None,
            "vid_number": None,
            "address": None,
            "address_english": None,
            "address_hindi": None,
            "father_name": None,
            "father_name_english": None,
            "father_name_hindi": None,
        },
        "PAN": {
            "name": None,
            "name_english": None,
            "name_hindi": None,
            "father_name": None,
            "father_name_english": None,
            "father_name_hindi": None,
            "dob": None,
            "pan_number": None,
        },
        "VOTER_ID": {
            "epic_number": None,
            "name": None,
            "name_english": None,
            "name_hindi": None,
            "dob": None,
            "gender": None,
        },
        "DRIVING_LICENSE": {
            "dl_number": None,
            "name": None,
            "name_english": None,
            "name_hindi": None,
            "dob": None,
            "issue_date": None,
            "valid_till": None,
        },
    }

    doc_type_key = (doc_type or "").strip().lower()
    if doc_type_key in {"aadhaar", "aadhar"}:
        doc_type_standard = "AADHAAR"
    elif doc_type_key == "pan":
        doc_type_standard = "PAN"
    elif doc_type_key in {"voter_id", "voterid", "voter", "voters", "epic"}:
        doc_type_standard = "VOTER_ID"
    elif doc_type_key in {"driving_license", "driving_licence", "license", "licence", "dl", "driving"}:
        doc_type_standard = "DRIVING_LICENSE"
    else:
        raise ValueError(f"Unsupported doc_type for Ollama parsing: {doc_type!r}")

    schema = schema_map[doc_type_standard]
    schema_json = json.dumps(schema, indent=2)
    schema_json = schema_json.replace("{", "{{").replace("}", "}}")

    if doc_type_standard == "PAN":
        prompt_template = f"""You are a strict PAN card parsing engine.

    Extract data from the OCR text.

    PAN Card Layout Rules:
    - First full uppercase name line = Person Name.
    - Second full uppercase name line = Father's Name.
    - Date of Birth is in DD/MM/YYYY format.
    - PAN number format: 5 letters + 4 digits + 1 letter (e.g., ABCDE1234F).
    - Extract bilingual values where present:
      - name_english / name_hindi
      - father_name_english / father_name_hindi
    - Keep name and father_name as English when available, otherwise Hindi.
    - Do NOT guess missing fields.
    - Return ONLY valid JSON.
    - If field not found return null.

    Schema:
    {schema_json}

    OCR TEXT:
    {{message}}
    """
    elif doc_type_standard == "AADHAAR":
        prompt_template = f"""You are a strict document parsing engine.

Extract data from the OCR text.

Important Rules:
- Aadhaar number = exactly 12 digits (XXXX XXXX XXXX).
- VID = exactly 16 digits (XXXX XXXX XXXX XXXX), only if VID present.
- Address is the block of text under the word "Address" or after the name on the back side.
- Extract full address exactly as written (multi-line allowed) and don't merge two addresses if multiple addresses present keep the 2nd one.
- Father's name is the name mentioned after "S/O" or "D/O" or "C/O".
- Extract bilingual values where present:
  - name_english / name_hindi
  - father_name_english / father_name_hindi
  - address_english / address_hindi
- Keep name, father_name, and address as English when available, otherwise Hindi.
- Date of Birth is in DD/MM/YYYY format only if (DOD, Date of Birth, Year of Birth) present otherwise set to null.
- Gender is either Male, Female, or Other.
- Do NOT treat father's name (after S/O, D/O, C/O) as the person's name but in pan.
- If the person's name is not explicitly present above DOB or Aadhaar number, set "name" to null.
- Do NOT guess missing fields.
- Ignore any date (Issue Date, Download Date, etc.) other than DOB.
- Never merge Aadhaar and VID.
- If a field is not found, return null.
- Return ONLY valid JSON.

Schema:
{schema_json}

 OCR TEXT:
 {{message}}
 """
    elif doc_type_standard == "VOTER_ID":
        prompt_template = f"""You are a strict Voter ID card parsing engine.

Extract data from the OCR text.

Important Rules:
- EPIC number format: 3 letters + 7 digits (e.g., ABC1234567).
- Date of Birth format: DD/MM/YYYY or DD-MM-YYYY (keep as found).
- Gender: MALE/FEMALE/OTHER (uppercase) if present.
- Extract name_english and name_hindi when available.
- Keep name as English when available, otherwise Hindi.
- Do NOT guess missing fields.
- If a field is not found, return null.
- Return ONLY valid JSON.

Schema:
{schema_json}

OCR TEXT:
{{message}}
"""
    elif doc_type_standard == "DRIVING_LICENSE":
        prompt_template = f"""You are a strict Driving License parsing engine.

Extract data from the OCR text.

Important Rules:
- DL number is an alphanumeric id (keep letters+digits, remove spaces). Do NOT invent it.
- Dates may be in DD/MM/YYYY or DD-MM-YYYY (keep as found).
- Extract name_english and name_hindi when available.
- Keep name as English when available, otherwise Hindi.
- Do NOT guess missing fields.
- If a field is not found, return null.
- Return ONLY valid JSON.

Schema:
{schema_json}

OCR TEXT:
{{message}}
"""
    else:
        raise ValueError(f"Unsupported doc_type for Ollama parsing: {doc_type!r}")


    result = chat_with_ollama(
        normalized_text,
        model=model,
        prompt_template=prompt_template,
        options={"temperature": 0},
    )

    raw_output = result["model_response"]
    # print("Raw output:\n", raw_output)

    try:
        data = _loads_first_json_object(raw_output)
        if doc_type_standard == "AADHAAR":
            # ===============================
            # STRICT AADHAAR VALIDATION LAYER
            # ===============================

            # --- 1. Strict DOB (Only valid if explicitly DOB present) ---
            dob_match = re.search(r"\b\d{2}/\d{2}/\d{4}\b", normalized_text)

            if dob_match and re.search(r"\bDOB\b|जन्म", normalized_text, re.IGNORECASE):
                data["dob"] = dob_match.group()
            else:
                data["dob"] = None


            # --- 2. Aadhaar Number (Strict 12 digit only) ---
            # Normalize whitespace
            clean_text = re.sub(r"\s+", " ", normalized_text)

            # --- Extract VID strictly ---
            vid_match = re.search(r"VID\s*:\s*(\d{4}\s\d{4}\s\d{4}\s\d{4})", clean_text)
            data["vid_number"] = vid_match.group(1) if vid_match else None

            # --- Extract Aadhaar using line-based logic ---
            aadhaar_number = None

            lines = [l.strip() for l in normalized_text.splitlines() if l.strip()]

            for line in lines:
                # Must be exactly 12-digit format
                if re.fullmatch(r"\d{4}\s\d{4}\s\d{4}", line):
                    aadhaar_number = line
                    break

            data["aadhaar_number"] = aadhaar_number
            # ----------------------- father name extraction (strict S/O, D/O, C/O) -----------------------
            father_match = re.search(
                r"\b(?:S/O|D/O|C/O)\s*:?\s*([A-Za-z\u0900-\u097F\s]+)",
                normalized_text,
                re.IGNORECASE
            )

            if father_match:
                data["father_name"] = father_match.group(1).strip()
            else:
                data["father_name"] = None



            # --- 5. Backside Detection (No DOB marker = no primary name) ---
            if not re.search(r"\bDOB\b|जन्म", normalized_text, re.IGNORECASE):
                data["name"] = None


            # --- 6. Address Cleanup ---
            if isinstance(data.get("address"), str):

                address = data["address"]

                # Remove Download Date contamination
                address = re.sub(r"Download Date:.*", "", address, flags=re.IGNORECASE)

                # Remove father line
                if data.get("father_name"):
                    address = re.sub(
                        rf"S/O\s+{re.escape(data['father_name'])}",
                        "",
                        address,
                        flags=re.IGNORECASE,
                    )

                # Clean whitespace
                lines = [l.strip() for l in address.splitlines() if l.strip()]
                data["address"] = ", ".join(dict.fromkeys(lines))
                # print("Ollama Parsed Successfully")
            return _ensure_bilingual_fields(data, ("name", "father_name", "address"))
        if doc_type_standard == "PAN":
            return _ensure_bilingual_fields(data, ("name", "father_name"))
        if doc_type_standard == "VOTER_ID":
            epic = data.get("epic_number")
            if isinstance(epic, str):
                epic_clean = epic.strip().upper()
                if not re.fullmatch(r"[A-Z]{3}[0-9]{7}", epic_clean):
                    epic_clean = None
                data["epic_number"] = epic_clean
            gender = data.get("gender")
            if isinstance(gender, str):
                data["gender"] = gender.strip().upper() or None
            return _ensure_bilingual_fields(data, ("name",))
        if doc_type_standard == "DRIVING_LICENSE":
            dl = data.get("dl_number")
            if isinstance(dl, str):
                dl_clean = re.sub(r"\s+", "", dl.strip().upper())
                data["dl_number"] = dl_clean or None
            return _ensure_bilingual_fields(data, ("name",))
        raise ValueError(f"Unsupported doc_type for Ollama parsing: {doc_type!r}")

    except Exception as e:
        raise RuntimeError(f"Failed to parse Ollama JSON: {e}")

