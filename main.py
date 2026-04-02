from __future__ import annotations

import argparse
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import pytesseract

_PAN_NUMBER_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_AADHAAR_NUMBER_RE = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
_EPIC_NUMBER_RE = re.compile(r"\b[A-Z]{3}[0-9]{7}\b")
_DL_NUMBER_RE = re.compile(
    r"\b[A-Z]{2}\s?\d{2}\s?\d{4}\s?\d{5,7}\b|\b[A-Z]{2}-?\d{2}\d{11,13}\b|\b[A-Z]{2}\d{13,15}\b",
    re.IGNORECASE,
)
_LATIN_RE = re.compile(r"[A-Za-z]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_BILINGUAL_FIELDS: tuple[str, ...] = ("name", "father_name", "address")

_RESUME_SKILLS = {
    "python",
    "java",
    "javascript",
    "typescript",
    "react",
    "node",
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
    "tensorflow",
    "pytorch",
    "keras",
    "fastapi",
    "flask",
    "django",
    "machine learning",
    "deep learning",
    "nlp",
    "computer vision",
    "git",
    "excel",
    "power bi",
    "tableau",
}


@dataclass(frozen=True)
class OcrTextBundle:
    english: str
    hindi: str
    combined: str


def _clean_optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _dedupe_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        deduped.append(line)
    return deduped


def _split_text_by_script(text: str) -> dict[str, str]:
    lines = [line.strip() for line in str(text or "").splitlines() if line and line.strip()]

    english_lines: list[str] = []
    hindi_lines: list[str] = []
    for line in lines:
        if _LATIN_RE.search(line):
            english_lines.append(line)
        if _DEVANAGARI_RE.search(line):
            hindi_lines.append(line)

    return {
        "english": "\n".join(_dedupe_lines(english_lines)).strip(),
        "hindi": "\n".join(_dedupe_lines(hindi_lines)).strip(),
        "combined": "\n".join(_dedupe_lines(lines)).strip(),
    }


def _normalize_ocr_payload(english: str = "", hindi: str = "", combined: str = "") -> dict[str, str]:
    merged = _split_text_by_script(combined)
    english_text = _clean_optional_text(english) or merged["english"]
    hindi_text = _clean_optional_text(hindi) or merged["hindi"]
    combined_text = merged["combined"] or _clean_optional_text(combined) or ""
    return {
        "english": english_text or "",
        "hindi": hindi_text or "",
        "combined": combined_text,
    }


def _attach_bilingual_output(data: dict[str, Any] | None) -> dict[str, Any]:
    enriched = dict(data or {})
    for field in _BILINGUAL_FIELDS:
        english_key = f"{field}_english"
        hindi_key = f"{field}_hindi"

        base_value = _clean_optional_text(enriched.get(field))
        english_value = _clean_optional_text(enriched.get(english_key))
        hindi_value = _clean_optional_text(enriched.get(hindi_key))

        if base_value:
            if _LATIN_RE.search(base_value) and not english_value:
                english_value = base_value
            if _DEVANAGARI_RE.search(base_value) and not hindi_value:
                hindi_value = base_value

        enriched[field] = base_value or english_value or hindi_value
        enriched[english_key] = english_value
        enriched[hindi_key] = hindi_value

    return enriched


def _normalize_doc_type(value: Any) -> str | None:
    v = str(value or "").strip().lower()
    if not v:
        return None
    if v in {"aadhaar", "aadhar"}:
        return "aadhaar"
    if v == "pan":
        return "pan"
    if v in {"voter", "voters", "voterid", "voter_id", "epic"}:
        return "voter_id"
    if v in {"licence", "dl", "driving_license", "license", "driving_licence"}:
        return "driving_licence"
    if v in {"resume", "cv"}:
        return "resume"
    return None


def _doc_type_label(doc_type_hint: str | None) -> str:
    if doc_type_hint == "pan":
        return "Pan"
    if doc_type_hint == "aadhaar":
        return "Aadhar"
    if doc_type_hint == "voter_id":
        return "Voters"
    if doc_type_hint == "driving_licence":
        return "Driving License"
    if doc_type_hint == "resume":
        return "Resume"
    return "Others"


def _predict_label_from_text(text: str, *, doc_type_hint: str | None = None) -> tuple[str, float]:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    upper = normalized.upper()

    if (
        _PAN_NUMBER_RE.search(upper)
        or "PERMANENT ACCOUNT NUMBER" in upper
        or "INCOME TAX DEPARTMENT" in upper
    ):
        return "Pan", 0.98

    if _AADHAAR_NUMBER_RE.search(normalized) or "UIDAI" in upper or "AADHAAR" in upper:
        return "Aadhar", 0.92 if _AADHAAR_NUMBER_RE.search(normalized) else 0.75

    if _EPIC_NUMBER_RE.search(upper) or "ELECTION COMMISSION" in upper or "ELECTOR" in upper:
        return "Voters", 0.88 if _EPIC_NUMBER_RE.search(upper) else 0.72

    if _DL_NUMBER_RE.search(upper) or "DRIVING LICENCE" in upper or "DRIVING LICENSE" in upper:
        return "Driving License", 0.84 if _DL_NUMBER_RE.search(upper) else 0.68

    if (
        "RESUME" in upper
        or "CURRICULUM VITAE" in upper
        or "EXPERIENCE" in upper
        or "EDUCATION" in upper
    ):
        return "Resume", 0.7

    hint = _normalize_doc_type(doc_type_hint)
    if hint:
        return _doc_type_label(hint), 0.55

    return "Others", 0.5


def _try_predict_document(image_path: str, doc_type_hint: str | None) -> tuple[str, float] | None:
    try:
        from doc_classifier.router import predict_document_routed
    except Exception:
        return None

    try:
        return predict_document_routed(image_path, doc_type_hint=doc_type_hint)
    except Exception:
        return None


def _extract_pdf_text(pdf_path: str, max_pages: int | None = None) -> tuple[list[str], int]:
    try:
        import fitz
    except Exception:
        fitz = None

    if fitz is not None:
        doc = fitz.open(pdf_path)
        try:
            total_pages = doc.page_count
            pages_to_read = total_pages if max_pages is None else min(total_pages, max_pages)
            page_texts = []
            for i in range(pages_to_read):
                page = doc.load_page(i)
                page_texts.append(page.get_text("text") or "")
            return page_texts, total_pages
        finally:
            doc.close()

    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError("PDF text extraction is unavailable. Install PyMuPDF or pypdf.") from exc

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    pages_to_read = total_pages if max_pages is None else min(total_pages, max_pages)
    page_texts = []
    for i in range(pages_to_read):
        page_texts.append(reader.pages[i].extract_text() or "")
    return page_texts, total_pages


def _render_pdf_pages_to_images(
    pdf_path: str,
    output_dir: str,
    dpi: int = 220,
    max_pages: int | None = None,
) -> tuple[list[str], int]:
    try:
        import fitz
    except Exception as exc:
        raise RuntimeError("PDF rendering is unavailable. Install PyMuPDF to process scanned PDFs.") from exc

    zoom = float(dpi) / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    doc = fitz.open(pdf_path)
    try:
        total_pages = doc.page_count
        pages_to_render = total_pages if max_pages is None else min(total_pages, max_pages)

        image_paths: list[str] = []
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in range(pages_to_render):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out_path = out_dir / f"page-{i + 1}.png"
            pix.save(str(out_path))
            image_paths.append(str(out_path))

        return image_paths, total_pages
    finally:
        doc.close()


def _parse_by_class_id(class_id_normalized: str, ocr_text: str) -> dict[str, Any]:
    if class_id_normalized in {"aadhaar", "aadhar"}:
        from parser.aadhaar_parser import parse_aadhaar

        return parse_aadhaar(ocr_text)
    if class_id_normalized == "pan":
        from parser.pan_parser import parse_pan

        return parse_pan(ocr_text)
    if class_id_normalized in {"voter_id", "voterid", "voter", "voters"}:
        from parser.voter_id_parser import parse_voter_id

        return parse_voter_id(ocr_text)
    if class_id_normalized in {"driving_licence", "driving_license", "license", "licence", "dl"}:
        from parser.driving_license_parser import parse_driving_license

        return parse_driving_license(ocr_text)
    if class_id_normalized == "resume":
        return _parse_resume_locally(ocr_text)
    return {"raw_text": ocr_text}


def _parse_text_with_fallback(class_id_normalized: str, text: str) -> dict[str, Any]:
    try:
        from parser.ollama import parse_with_ollama

        return parse_with_ollama(text, class_id_normalized)
    except Exception:
        return _parse_by_class_id(class_id_normalized, text)


def _extract_class_id(label: Any) -> str:
    label_str = str(label or "").strip()
    if label_str:
        return label_str.split()[-1]
    return "Others"


def _build_mismatch_payload(
    *,
    expected_doc_type: str,
    detected_class_id: str,
    confidence: float,
    source: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "error": "Uploaded document does not match the selected portal.",
        "mismatch": True,
        "stopped_early": True,
        "expected_doc_type": expected_doc_type,
        "detected_doc_type": _normalize_doc_type(detected_class_id),
        "class_id": detected_class_id,
        "confidence": float(confidence or 0.0),
        "source": source,
    }
    if extra:
        payload.update(extra)
    return payload


def _detect_document_source(image_path: str) -> str:
    try:
        from utils.source_detector import detect_document_source

        return detect_document_source(image_path)
    except Exception:
        suffix = Path(image_path).suffix.lower()
        if suffix == ".pdf":
            return "Digital (PDF Source)"
        return "Image Upload"


def _enhance_image_for_ocr(image_path: str) -> Any:
    try:
        from preprocess.image_enhancer import enhance_for_ocr

        return enhance_for_ocr(image_path)
    except Exception:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found")
        return image


def _tesseract_text(
    image_path: str,
    *,
    lang: str | None,
    allow_default_fallback: bool = True,
) -> str:
    kwargs: dict[str, Any] = {}
    if lang:
        kwargs["lang"] = lang
    try:
        return pytesseract.image_to_string(image_path, **kwargs)
    except Exception:
        if lang and allow_default_fallback:
            return pytesseract.image_to_string(image_path)
        if lang:
            return ""
        raise


def _ocr_with_fallback(image_path: str) -> OcrTextBundle:
    try:
        from ocr.azure_reader import read_text_multilingual

        bundle = read_text_multilingual(image_path)
        return OcrTextBundle(
            english=bundle.english,
            hindi=bundle.hindi,
            combined=bundle.combined,
        )
    except Exception:
        english = _clean_optional_text(_tesseract_text(image_path, lang="eng")) or ""
        hindi = _clean_optional_text(
            _tesseract_text(image_path, lang="hin", allow_default_fallback=False)
        ) or ""
        combined = _clean_optional_text(_tesseract_text(image_path, lang="eng+hin")) or ""
        if not combined:
            combined = _clean_optional_text(_tesseract_text(image_path, lang=None)) or ""
        merged = _normalize_ocr_payload(english=english, hindi=hindi, combined=combined)
        return OcrTextBundle(
            english=merged["english"],
            hindi=merged["hindi"],
            combined=merged["combined"],
        )


def _parse_resume_locally(text: str) -> dict[str, Any]:
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text or "")
    phone_match = re.search(r"(\+?\d{1,3}[\s-]?)?\d{10}", text or "")
    years_match = re.search(r"(\d+)\+?\s+years?", str(text or ""), re.IGNORECASE)

    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    name = None
    for line in lines[:10]:
        if "@" in line or re.search(r"\d", line):
            continue
        word_count = len(line.split())
        if 2 <= word_count <= 4:
            name = line
            break

    lowered = str(text or "").lower()
    skills = sorted(skill for skill in _RESUME_SKILLS if skill in lowered)

    return {
        "name": name,
        "email": email_match.group(0) if email_match else None,
        "phone": phone_match.group(0) if phone_match else None,
        "skills": skills,
        "experience_years": int(years_match.group(1)) if years_match else None,
    }


def process_document(image_path: str, doc_type: str | None = None) -> dict[str, Any]:
    doc_type_hint = _normalize_doc_type(doc_type)

    prediction = _try_predict_document(image_path, doc_type_hint)
    if prediction is not None:
        label, confidence = prediction
    else:
        label = "Others"
        confidence = 0.0

    class_id = _extract_class_id(label)
    class_id_normalized = class_id.strip().lower()
    detected_doc_type = _normalize_doc_type(class_id_normalized)

    if doc_type_hint and detected_doc_type and detected_doc_type != doc_type_hint:
        return _build_mismatch_payload(
            expected_doc_type=doc_type_hint,
            detected_class_id=class_id,
            confidence=confidence,
            source="Classification (Early Stop)",
        )

    source = _detect_document_source(image_path)
    enhanced = _enhance_image_for_ocr(image_path)

    tmp_path: str | None = None
    ocr_payload = {"english": "", "hindi": "", "combined": ""}

    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        if not cv2.imwrite(tmp_path, enhanced):
            raise ValueError("Failed to write temporary image for OCR")

        ocr_bundle = _ocr_with_fallback(tmp_path)
        ocr_payload = _normalize_ocr_payload(
            english=ocr_bundle.english,
            hindi=ocr_bundle.hindi,
            combined=ocr_bundle.combined,
        )
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    if prediction is None:
        label, confidence = _predict_label_from_text(ocr_payload["combined"], doc_type_hint=doc_type_hint)
        class_id = _extract_class_id(label)
        class_id_normalized = class_id.strip().lower()
        detected_doc_type = _normalize_doc_type(class_id_normalized)

    if doc_type_hint and detected_doc_type and detected_doc_type != doc_type_hint:
        return _build_mismatch_payload(
            expected_doc_type=doc_type_hint,
            detected_class_id=class_id,
            confidence=confidence,
            source="OCR Classification",
        )

    data = _parse_text_with_fallback(class_id_normalized, ocr_payload["combined"])
    data = _attach_bilingual_output(data)

    return {
        **data,
        "class_id": class_id,
        "confidence": confidence,
        "source": source,
    }


def process_pdf(pdf_path: str, doc_type: str | None = None) -> dict[str, Any]:
    doc_type_hint = _normalize_doc_type(doc_type)

    first_page_texts, total_pages = _extract_pdf_text(pdf_path, max_pages=1)
    first_page_text = "\n\n".join(t.strip() for t in first_page_texts if t and t.strip())

    if len(re.sub(r"\s+", "", first_page_text)) >= 30:
        if doc_type_hint:
            first_label, first_confidence = _predict_label_from_text(first_page_text, doc_type_hint=doc_type_hint)
            first_class_id = _extract_class_id(first_label)
            detected_doc_type = _normalize_doc_type(first_class_id.strip().lower())
            if detected_doc_type and detected_doc_type != doc_type_hint:
                return _build_mismatch_payload(
                    expected_doc_type=doc_type_hint,
                    detected_class_id=first_class_id,
                    confidence=first_confidence,
                    source="Digital PDF",
                    extra={"pdf": {"kind": "digital", "page_count": total_pages}},
                )

        page_texts, total_pages = _extract_pdf_text(pdf_path)
        combined_text = "\n\n".join(t.strip() for t in page_texts if t and t.strip())
        label, confidence = _predict_label_from_text(combined_text, doc_type_hint=doc_type_hint)
        class_id = _extract_class_id(label)
        class_id_normalized = class_id.strip().lower()
        data = _parse_text_with_fallback(class_id_normalized, combined_text)
        data = _attach_bilingual_output(data)
        return {
            **data,
            "class_id": class_id,
            "confidence": confidence,
            "source": "Digital PDF",
            "pdf": {"kind": "digital", "page_count": total_pages},
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        if doc_type_hint:
            first_page_images, total_pages = _render_pdf_pages_to_images(pdf_path, tmpdir, dpi=220, max_pages=1)
            if first_page_images:
                first_page_result = process_document(first_page_images[0], doc_type_hint)
                if first_page_result.get("mismatch"):
                    return {
                        **first_page_result,
                        "source": "Scanned PDF",
                        "pdf": {
                            "kind": "scanned",
                            "page_count": total_pages,
                            "processed_pages": 1,
                            "checked_pages": 1,
                        },
                        "mismatch_page": 1,
                    }

        page_images, total_pages = _render_pdf_pages_to_images(pdf_path, tmpdir, dpi=220)

        if doc_type_hint:
            for page_number, image_path in enumerate(page_images, start=1):
                page_result = process_document(image_path, doc_type_hint)
                if page_result.get("mismatch"):
                    return {
                        **page_result,
                        "source": "Scanned PDF",
                        "pdf": {
                            "kind": "scanned",
                            "page_count": total_pages,
                            "processed_pages": page_number,
                        },
                        "mismatch_page": page_number,
                    }

        page_results = []
        for page_number, image_path in enumerate(page_images, start=1):
            page_results.append({"page": page_number, "result": process_document(image_path, doc_type_hint)})

        if page_results:
            valid_results = [p["result"] for p in page_results if isinstance(p.get("result"), dict)]
            best_result = max(valid_results, key=lambda r: float(r.get("confidence") or 0.0))
        else:
            best_result = {
                "error": "No pages rendered from PDF",
                "class_id": "Others",
                "confidence": 0.0,
                "source": "Scanned PDF",
            }

        return {
            **best_result,
            "source": "Scanned PDF",
            "pdf": {
                "kind": "scanned",
                "page_count": total_pages,
                "processed_pages": len(page_results),
                "pages": page_results,
            },
        }


def output_filename_for_class_id(class_id: str | None) -> str:
    normalized = _normalize_doc_type(class_id)
    return f"{normalized or 'others'}.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR + classification + parsing on an image or PDF.")
    parser.add_argument("image_path", help="Path to an input image (jpg/png) or PDF.")
    parser.add_argument("--doc-type", default=None, help="Optional expected document type.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    input_path = Path(args.image_path)
    if input_path.suffix.lower() == ".pdf":
        payload = process_pdf(str(input_path), args.doc_type)
    else:
        payload = process_document(str(input_path), args.doc_type)

    output_path = args.output
    if output_path:
        Path(output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
