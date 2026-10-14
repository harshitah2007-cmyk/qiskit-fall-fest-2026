from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import pandas as pd
import wfdb
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

FEATURES = [f"f{i}" for i in range(1, 9)]
MAX_BEATS_PER_REQUEST = 100
MODEL_PATH = Path(os.getenv("MODEL_BUNDLE_PATH", Path(__file__).parent / "model_bundle.pkl"))

app = FastAPI(
    title="VQC-ECG Inference API",
    description="Educational ECG signal classification prototype. Not for medical use.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    # Public static website + public demo API; no cookies or credentials are used.
    # A wildcard avoids browser preflight failures if the site is opened from a
    # preview URL or another allowed demo page.
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,
)

_bundle: dict[str, Any] | None = None


def get_bundle() -> dict[str, Any]:
    """Load the model bundle on the first inference request."""
    global _bundle
    if _bundle is not None:
        return _bundle
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Model bundle is missing. Export model_bundle.pkl from the "
                "trained Colab notebook and place it in the api folder."
            ),
        )
    try:
        with MODEL_PATH.open("rb") as f:
            loaded = cloudpickle.load(f)
        required = {"vqc", "scaler", "pca", "angle_scaler"}
        if not isinstance(loaded, dict) or not required.issubset(loaded):
            raise ValueError(f"Bundle must contain keys: {sorted(required)}")
        _bundle = loaded
        return _bundle
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not load model bundle: {type(exc).__name__}: {exc}",
        ) from exc


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "VQC-ECG Inference API",
        "health": "/health",
        "docs": "/docs",
        "note": "Educational prototype only; not for diagnosis or treatment.",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_bundle_present": MODEL_PATH.exists(),
        "model_loaded": _bundle is not None,
        "expected_features": FEATURES,
    }


def parse_feature_csv(payload: bytes) -> tuple[pd.DataFrame, np.ndarray | None] | None:
    """Parse f1...f8 rows and, optionally, a known label column for evaluation."""
    try:
        table = pd.read_csv(io.BytesIO(payload), sep=None, engine="python")
    except Exception:
        return None
    normalized = {str(col).strip().lower(): col for col in table.columns}
    if not all(name in normalized for name in FEATURES):
        return None

    feature_table = table[[normalized[name] for name in FEATURES]].copy()
    feature_table.columns = FEATURES
    for col in FEATURES:
        feature_table[col] = pd.to_numeric(feature_table[col], errors="coerce")
    feature_table = feature_table.replace([np.inf, -np.inf], np.nan)

    # If present, label/true_label is the dataset ground truth (0=Normal, 1=Abnormal).
    label_key = "label" if "label" in normalized else ("true_label" if "true_label" in normalized else None)
    labels: pd.Series | None = None
    if label_key is not None:
        labels = pd.to_numeric(table[normalized[label_key]], errors="coerce")
        valid = feature_table.notna().all(axis=1) & labels.notna()
        feature_table = feature_table.loc[valid].copy()
        labels = labels.loc[valid]
        if labels.empty:
            raise HTTPException(
                status_code=400,
                detail="The CSV contains f1...f8 and a label column, but no complete numeric rows.",
            )
        if not labels.isin([0, 1]).all():
            raise HTTPException(
                status_code=400,
                detail="Labels must be 0 (Normal) or 1 (Abnormal).",
            )
        labels = labels.astype(int)
    else:
        feature_table = feature_table.dropna()

    if feature_table.empty:
        raise HTTPException(
            status_code=400,
            detail="The CSV has f1...f8 columns, but no complete numeric feature rows.",
        )
    return feature_table, (labels.to_numpy(dtype=int) if labels is not None else None)


def parse_signal(payload: bytes) -> np.ndarray:
    """Read a plain numeric signal file; the first column must be the ECG channel."""
    try:
        table = pd.read_csv(io.BytesIO(payload), header=None, comment="#", sep=None, engine="python")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not read the signal file. Try a CSV or TXT with one numeric ECG column.",
        ) from exc
    if table.empty:
        raise HTTPException(status_code=400, detail="The uploaded signal file is empty.")
    # The first column is intentionally used: upload a single ECG channel, not timestamp + ECG columns.
    signal = pd.to_numeric(table.iloc[:, 0], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if signal.size < 3:
        raise HTTPException(
            status_code=400,
            detail="Not enough numeric samples were found in the first column.",
        )
    if not np.all(np.isfinite(signal)):
        raise HTTPException(status_code=400, detail="The signal contains non-finite values.")
    return signal


def beat_features(signal: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, list[int]]:
    """Detect candidate R-peaks and compute the same eight statistics used in training."""
    if not np.isfinite(sampling_rate) or sampling_rate < 50 or sampling_rate > 2000:
        raise HTTPException(status_code=400, detail="Sampling rate must be between 50 and 2000 Hz.")
    if signal.size < int(sampling_rate * 2):
        raise HTTPException(status_code=400, detail="Please upload at least 2 seconds of ECG samples.")
    try:
        peaks = wfdb.processing.xqrs_detect(sig=signal.astype(float), fs=float(sampling_rate), verbose=False)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not detect candidate beats. Check the ECG signal and sampling rate.",
        ) from exc

    before = int(0.20 * sampling_rate)
    after = int(0.40 * sampling_rate)
    rows: list[list[float]] = []
    kept_peaks: list[int] = []
    for peak in np.asarray(peaks, dtype=int):
        start, end = int(peak) - before, int(peak) + after
        if start < 0 or end > len(signal):
            continue
        segment = signal[start:end]
        if len(segment) < 3:
            continue
        rows.append([
            float(np.mean(segment)),
            float(np.std(segment)),
            float(np.min(segment)),
            float(np.max(segment)),
            float(np.ptp(segment)),
            float(np.sqrt(np.mean(segment ** 2))),
            float(np.mean(np.abs(segment))),
            float(np.mean(np.abs(np.diff(segment)))),
        ])
        kept_peaks.append(int(peak))
    if not rows:
        raise HTTPException(
            status_code=400,
            detail="No complete beat windows were found. Upload a longer signal with a correct sampling rate.",
        )
    return np.asarray(rows, dtype=float), kept_peaks


def run_vqc(features: np.ndarray) -> np.ndarray:
    bundle = get_bundle()
    try:
        standardized = bundle["scaler"].transform(features)
        components = bundle["pca"].transform(standardized)
        angles = bundle["angle_scaler"].transform(components)
        predictions = np.asarray(bundle["vqc"].predict(angles)).astype(int).ravel()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {type(exc).__name__}: {exc}",
        ) from exc
    if not np.isin(predictions, [0, 1]).all():
        raise HTTPException(status_code=500, detail="The model returned an unexpected class.")
    return predictions


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    sampling_rate: float = Form(360.0),
) -> dict[str, Any]:
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    feature_result = parse_feature_csv(payload)
    ground_truth: np.ndarray | None = None
    if feature_result is not None:
        feature_table, ground_truth = feature_result
        input_features = feature_table[FEATURES].to_numpy(dtype=float)
        source_type = "precomputed_features"
        peak_indices: list[int] = []
    else:
        signal = parse_signal(payload)
        input_features, peak_indices = beat_features(signal, sampling_rate)
        source_type = "raw_signal"

    if len(input_features) > MAX_BEATS_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Please upload at most {MAX_BEATS_PER_REQUEST} beat rows per request; this demo limits batch size to keep inference responsive.",
        )

    predictions = run_vqc(input_features)
    result_rows = []
    for i, pred in enumerate(predictions):
        row: dict[str, Any] = {
            "beat": i + 1,
            "class_id": int(pred),
            "class_label": "Normal-labelled" if int(pred) == 0 else "Abnormal-labelled",
        }
        if peak_indices:
            row["sample_index"] = peak_indices[i]
            row["time_seconds"] = round(peak_indices[i] / sampling_rate, 3)
        result_rows.append(row)

    normal_count = int(np.sum(predictions == 0))
    abnormal_count = int(np.sum(predictions == 1))
    evaluation: dict[str, Any] | None = None
    if ground_truth is not None:
        y_true = ground_truth.astype(int)
        y_pred = predictions.astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        evaluation = {
            "evaluated_examples": int(len(y_true)),
            "accuracy": float(np.mean(y_true == y_pred)),
            "abnormal_f1": float((2 * tp) / max(2 * tp + fp + fn, 1)),
            "abnormal_recall": float(tp / max(tp + fn, 1)),
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "note": "Metrics compare predictions with the uploaded dataset labels. Use held-out examples not used to fit the model.",
        }
    return {
        "ok": True,
        "source_type": source_type,
        "beats_classified": len(result_rows),
        "normal_labelled_count": normal_count,
        "abnormal_labelled_count": abnormal_count,
        "sampling_rate_hz": sampling_rate if source_type == "raw_signal" else None,
        "predictions": result_rows,
        "evaluation": evaluation,
        "disclaimer": (
            "Research prototype only. Labels are experimental model outputs, not a diagnosis. "
            "The model was evaluated on a small balanced subset and is not clinically validated."
        ),
    }
