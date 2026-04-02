from __future__ import annotations

from pathlib import Path
import threading
from typing import Any

from converted_savedmodel.prediction import predict_document as predict_tf_document

_ROOT_DIR = Path(__file__).resolve().parents[1]


_SPECIALIZED_YOLO_MODELS: dict[str, Path] = {
    "voter_id": _ROOT_DIR / "trained_models" / "voter_id_best" / "weights" / "best.pt",
    "driving_licence": _ROOT_DIR / "trained_models" / "driving_licence_best" / "weights" / "best.pt",
    "resume": _ROOT_DIR / "trained_models" / "resume_best" / "weights" / "best.pt",
}

_SPECIALIZED_POSITIVE_CLASS: dict[str, str] = {
    "voter_id": "voter_id",
    "driving_licence": "driving_licence",
    "resume": "resume",
}

_SPECIALIZED_LABEL: dict[str, str] = {
    "voter_id": "Voters",
    "driving_licence": "Driving License",
    "resume": "Resume",
}

_SPECIALIZED_THRESHOLD: dict[str, float] = {
    "voter_id": 0.65,
    "driving_licence": 0.65,
    "resume": 0.65,
}

_yolo_models: dict[str, Any] = {}
_yolo_locks: dict[str, threading.Lock] = {}
_yolo_init_lock = threading.Lock()


def _try_import_yolo() -> Any | None:
    try:
        from ultralytics import YOLO  # type: ignore

        return YOLO
    except Exception:
        return None


def _get_yolo_model(doc_type_hint: str) -> Any | None:
    model_path = _SPECIALIZED_YOLO_MODELS.get(doc_type_hint)
    if not model_path or not model_path.exists():
        return None

    YOLO = _try_import_yolo()
    if YOLO is None:
        return None

    with _yolo_init_lock:
        if doc_type_hint in _yolo_models:
            return _yolo_models[doc_type_hint]

        model = YOLO(str(model_path))
        _yolo_models[doc_type_hint] = model
        _yolo_locks[doc_type_hint] = threading.Lock()
        return model


def _predict_one_vs_all(doc_type_hint: str, image_path: str) -> tuple[str, float] | None:
    model = _get_yolo_model(doc_type_hint)
    if model is None:
        return None

    lock = _yolo_locks.get(doc_type_hint)
    if lock is None:
        lock = threading.Lock()
        _yolo_locks[doc_type_hint] = lock

    with lock:
        results = model.predict(source=image_path, verbose=False)

    if not results:
        return None

    result = results[0]
    probs = getattr(result, "probs", None)
    names = getattr(result, "names", None) or getattr(model, "names", None) or {}

    positive = _SPECIALIZED_POSITIVE_CLASS.get(doc_type_hint)
    if not positive or probs is None:
        return None

    positive_index: int | None = None
    if isinstance(names, dict):
        for i, n in names.items():
            if str(n) == positive:
                positive_index = int(i)
                break

    if positive_index is None:
        return None

    data = getattr(probs, "data", None)
    if data is None:
        return None

    try:
        positive_conf = float(data[positive_index])
    except Exception:
        return None

    threshold = float(_SPECIALIZED_THRESHOLD.get(doc_type_hint, 0.65))
    if positive_conf < threshold:
        return None

    return _SPECIALIZED_LABEL.get(doc_type_hint, positive), positive_conf


def predict_document_routed(image_path: str, *, doc_type_hint: str | None = None) -> tuple[str, float]:
    """
    Predicts document type using a routed multi-model approach.

    - If doc_type_hint matches a specialized one-vs-all YOLO classifier, use it.
      If it's not confidently the requested document type, fall back to the existing
      TensorFlow multi-class classifier to preserve current behavior.
    - If no doc_type_hint is provided, use the existing TensorFlow model (unchanged).
    """

    if doc_type_hint:
        yolo_prediction = _predict_one_vs_all(doc_type_hint, image_path)
        if yolo_prediction is not None:
            return yolo_prediction

    return predict_tf_document(image_path)

