from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.core.credentials import AzureKeyCredential


@dataclass(frozen=True)
class OcrTextBundle:
    english: str
    hindi: str
    combined: str


@lru_cache(maxsize=1)
def _get_client() -> ImageAnalysisClient:
    endpoint = os.getenv("AZURE_VISION_ENDPOINT") or os.getenv("AZURE_ENDPOINT")
    key = os.getenv("AZURE_VISION_KEY") or os.getenv("AZURE_KEY")

    if not endpoint or not key:
        raise RuntimeError(
            "Azure Vision is not configured. Set AZURE_VISION_ENDPOINT and AZURE_VISION_KEY environment variables."
        )

    return ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))


def _dedupe_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            result.append(line)
    return result


def _extract_read_text(result: object) -> str:
    lines: list[str] = []

    read_result = getattr(result, "read", None)
    blocks = getattr(read_result, "blocks", None) or []
    for block in blocks:
        for line in (getattr(block, "lines", None) or []):
            text = str(getattr(line, "text", "") or "").strip()
            if text:
                lines.append(text)

    return "\n".join(lines).strip()


def _analyze_read(image_data: bytes, *, language: str | None) -> str:
    kwargs = {"image_data": image_data, "visual_features": ["Read"]}
    if language:
        kwargs["language"] = language

    try:
        result = _get_client().analyze(**kwargs)
    except TypeError:
        if language is None:
            raise
        result = _get_client().analyze(image_data=image_data, visual_features=["Read"])

    return _extract_read_text(result)


def _merge_text_blocks(*text_blocks: str) -> str:
    merged_lines: list[str] = []
    for block in text_blocks:
        for line in str(block or "").splitlines():
            value = line.strip()
            if value:
                merged_lines.append(value)
    return "\n".join(_dedupe_lines(merged_lines)).strip()


def read_text_multilingual(image_path: str) -> OcrTextBundle:
    with open(image_path, "rb") as f:
        image_data = f.read()

    try:
        english = _analyze_read(image_data, language="en")
    except Exception:
        english = _analyze_read(image_data, language=None)

    try:
        hindi = _analyze_read(image_data, language="hi")
    except Exception:
        hindi = ""

    combined = _merge_text_blocks(english, hindi)
    if not combined:
        combined = _analyze_read(image_data, language=None)
        if not english:
            english = combined

    return OcrTextBundle(english=english, hindi=hindi, combined=combined)


def read_text(image_path: str) -> str:
    return read_text_multilingual(image_path).combined
