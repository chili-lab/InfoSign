#!/usr/bin/env python3
"""Build auditable, task-independent AI-READI ECG waveform artifacts (V2).

The V2 contract intentionally keeps ECG samples in physical millivolts.  It
does not perform participant-wise centering, z-scoring, or amplitude
normalization.  Per-record/per-lead sufficient statistics are written so a
downstream job can fit a scaler on the *training split only*.

Default behaviour preserves the complete native recording and native sample
rate.  ``--target-fs`` and ``--target-samples`` are optional.  Requested
resampling uses SciPy's anti-aliased polyphase implementation when available;
the default is to fail rather than silently fall back to alias-prone linear
interpolation.

Outputs under ``--out-dir``:

  ecg_waveforms/                 one config-aware ``.pt`` artifact per record
  ecg_index.csv                  one row per successfully built/cached record
  ecg_numeric_features_wide.csv Rate/PR/QRSD/QT/QTc/P/QRS/T values + masks
  ecg_channel_stats.csv          sufficient statistics for train-only scaling
  ecg_duplicates.csv             deterministic duplicate ranking/audit
  ecg_failures.json              explicit read/resample/unit/cache failures
  ecg_schema.json                tensor/mask/unit/QC contract and run counts
  artifact_config.json           canonical preprocessing config + SHA-256 hash

Participant position is retained as metadata (including parsed degrees) but is
not included in the model numeric vector.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
try:
    import torch
except ImportError:  # Manifest/QC helpers can still be inspected without PyTorch.
    torch = None  # type: ignore

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None

try:  # Support both ``python ot2/file.py`` and ``import ot2.file``.
    from .preprocessing_utils import (
        artifact_config,
        as_person_id,
        canonical_json_hash,
        source_file_metadata,
    )
except ImportError:
    from preprocessing_utils import (  # type: ignore
        artifact_config,
        as_person_id,
        canonical_json_hash,
        source_file_metadata,
    )


CANONICAL_LEADS: Tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)
NUMERIC_FEATURES: Tuple[str, ...] = (
    "Rate", "PR", "QRSD", "QT", "QTc", "P", "QRS", "T",
)
REQUIRED_MANIFEST_COLUMNS = {
    "person_id", "wfdb_hea_filepath", "wfdb_dat_filepath",
}
MANIFEST_METADATA_COLUMNS: Tuple[str, ...] = (
    "modality",
    "participant_position",
    "machine_text",
    "machine_detail_description",
    "device_documentation_type_and_version",
    "interpretation_criteriaversion",
    "patient_criteriaversion",
    "internalmeasurements_version",
    "report_description",
    "manufacturer",
    "manufacturers_model_name",
)


@dataclass(frozen=True)
class LeadQCThresholds:
    """Conservative thresholds intended to reject only clearly unusable leads."""

    min_finite_fraction: float = 0.98
    min_std_mv: float = 0.005
    min_robust_range_mv: float = 0.02
    flat_step_epsilon_mv: float = 1e-5
    max_flat_step_fraction: float = 0.995
    max_abs_amplitude_mv: float = 25.0
    max_abs_baseline_mv: float = 5.0
    max_baseline_wander_mv: float = 5.0
    max_noise_mad_diff_mv: float = 2.5
    baseline_window_seconds: float = 2.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_finite_fraction <= 1.0:
            raise ValueError("min_finite_fraction must be in [0, 1]")
        if not 0.0 <= self.max_flat_step_fraction <= 1.0:
            raise ValueError("max_flat_step_fraction must be in [0, 1]")
        for name, value in asdict(self).items():
            if name not in {"min_finite_fraction", "max_flat_step_fraction"} and value < 0:
                raise ValueError(f"{name} must be non-negative")


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_scalar(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_scalar(v) for v in value]
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_scalar(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _optional_positive_float(value: str) -> Optional[float]:
    if str(value).strip().lower() in {"none", "null", "native", "preserve"}:
        return None
    out = float(value)
    if not np.isfinite(out) or out <= 0:
        raise argparse.ArgumentTypeError("value must be positive or 'none'")
    return out


def _optional_positive_int(value: str) -> Optional[int]:
    if str(value).strip().lower() in {"none", "null", "full", "preserve"}:
        return None
    out = int(value)
    if out <= 0:
        raise argparse.ArgumentTypeError("value must be positive or 'none'")
    return out


def _safe_file_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    if not token:
        raise ValueError("empty/unsafe artifact filename token")
    return token


def resolve_path(dataset_root: Path, value: Any) -> Path:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "none", "not reported"}:
        # Do not return Path(""): that aliases the current directory and would
        # incorrectly pass an exists() check.
        return dataset_root / "__missing_manifest_path__"
    path = Path(text)
    if path.is_absolute():
        # AI-READI manifest paths are normally dataset-root-relative despite the
        # leading slash.  Preserve a genuinely existing absolute path.
        if path.exists():
            return path
        return dataset_root / text.lstrip("/\\")
    return dataset_root / path


def parse_participant_position_degrees(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip().lower()
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*degrees?", text)
    if match:
        return float(match.group(1))
    semantic = (
        ("slight recline", 60.0),
        ("supine", 0.0),
        ("reclined", 30.0),
        ("sitting", 90.0),
    )
    for phrase, degrees in semantic:
        if phrase in text:
            return degrees
    return None


def numeric_values_and_mask(row: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    values = np.zeros(len(NUMERIC_FEATURES), dtype=np.float32)
    mask = np.zeros(len(NUMERIC_FEATURES), dtype=bool)
    for index, name in enumerate(NUMERIC_FEATURES):
        value = pd.to_numeric(row.get(name, np.nan), errors="coerce")
        if not pd.isna(value) and np.isfinite(float(value)):
            values[index] = float(value)
            mask[index] = True
    return values, mask


def import_wfdb():
    try:
        import wfdb  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency varies by runtime
        raise ImportError(
            "wfdb is required to read AI-READI .hea/.dat recordings; "
            "install the 'wfdb' package in the Python environment"
        ) from exc
    return wfdb


def read_wfdb_physical(hea_path: Path, dat_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not hea_path.is_file():
        raise FileNotFoundError(f"WFDB header not found: {hea_path}")
    if not dat_path.is_file():
        raise FileNotFoundError(f"WFDB signal file not found: {dat_path}")
    wfdb = import_wfdb()
    record_base = str(hea_path.with_suffix(""))
    signal, fields = wfdb.rdsamp(record_base)
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[0] == 0 or signal.shape[1] == 0:
        raise ValueError(f"Expected non-empty [samples, leads] WFDB data, got {signal.shape}")
    fs = _safe_float(fields.get("fs"))
    if fs is None or fs <= 0:
        raise ValueError(f"WFDB record has invalid sampling frequency: {fields.get('fs')!r}")
    header_samples = fields.get("sig_len", signal.shape[0])
    try:
        header_samples = int(header_samples)
    except (TypeError, ValueError):
        header_samples = int(signal.shape[0])
    metadata = {
        "record_base": record_base,
        "fs_raw": fs,
        "n_samples_loaded_raw": int(signal.shape[0]),
        "n_samples_header_raw": max(header_samples, 0),
        "n_signals_loaded": int(signal.shape[1]),
        "n_signals_header": int(fields.get("n_sig", signal.shape[1])),
        "sig_name_raw": [str(x) for x in (fields.get("sig_name") or [])],
        "units_raw": [str(x) for x in (fields.get("units") or [])],
        "comments": [str(x) for x in (fields.get("comments") or [])],
        "base_date": _json_scalar(fields.get("base_date")),
        "base_time": _json_scalar(fields.get("base_time")),
    }
    return signal, metadata


def normalize_lead_name(value: Any) -> str:
    text = str(value).strip().upper()
    for prefix in ("MDC_ECG_LEAD_", "ECG_LEAD_", "LEAD_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = re.sub(r"[^A-Z0-9]", "", text)
    aliases = {
        "AVR": "AVR", "AVL": "AVL", "AVF": "AVF",
        "VR": "AVR", "VL": "AVL", "VF": "AVF",
    }
    return aliases.get(text, text)


def canonicalize_leads(
    signal: np.ndarray,
    signal_names: Sequence[str],
    units: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[Optional[str]], List[Optional[int]], Dict[str, Any]]:
    """Map raw columns to canonical leads while retaining NaNs for true masks."""
    n_samples, n_columns = signal.shape
    output = np.full((n_samples, len(CANONICAL_LEADS)), np.nan, dtype=np.float64)
    header_mask = np.zeros(len(CANONICAL_LEADS), dtype=bool)
    source_units: List[Optional[str]] = [None] * len(CANONICAL_LEADS)
    source_indices: List[Optional[int]] = [None] * len(CANONICAL_LEADS)
    assumed_order = False
    duplicate_names: Dict[str, List[int]] = {}

    names = [str(x) for x in signal_names]
    if not names and n_columns == len(CANONICAL_LEADS):
        names = list(CANONICAL_LEADS)
        assumed_order = True
    if len(names) != n_columns:
        raise ValueError(
            f"WFDB sig_name length ({len(names)}) does not match signal columns ({n_columns}); "
            "refusing to guess lead identities"
        )

    by_name: Dict[str, List[int]] = {}
    for index, name in enumerate(names):
        by_name.setdefault(normalize_lead_name(name), []).append(index)
    duplicate_names = {name: idxs for name, idxs in by_name.items() if len(idxs) > 1}

    for output_index, lead in enumerate(CANONICAL_LEADS):
        candidates = by_name.get(normalize_lead_name(lead), [])
        if not candidates:
            continue
        source_index = candidates[0]
        output[:, output_index] = signal[:, source_index]
        header_mask[output_index] = True
        source_indices[output_index] = int(source_index)
        if source_index < len(units):
            source_units[output_index] = str(units[source_index])

    audit = {
        "lead_order_assumed_canonical": assumed_order,
        "duplicate_normalized_lead_names": duplicate_names,
        "unmapped_source_leads": [
            names[i]
            for i in range(n_columns)
            if normalize_lead_name(names[i]) not in {normalize_lead_name(x) for x in CANONICAL_LEADS}
        ],
    }
    return output, header_mask, source_units, source_indices, audit


def _unit_to_mv_factor(unit: Optional[str]) -> Optional[float]:
    if unit is None:
        return None
    text = str(unit).strip().lower().replace(" ", "")
    text = text.replace("μ", "u").replace("µ", "u")
    aliases = {
        "mv": 1.0,
        "millivolt": 1.0,
        "millivolts": 1.0,
        "uv": 1e-3,
        "microvolt": 1e-3,
        "microvolts": 1e-3,
        "v": 1e3,
        "volt": 1e3,
        "volts": 1e3,
    }
    return aliases.get(text)


def convert_canonical_to_mv(
    signal: np.ndarray,
    header_lead_mask: np.ndarray,
    source_units: Sequence[Optional[str]],
    unknown_units: str,
) -> Tuple[np.ndarray, List[Optional[float]], List[str]]:
    output = signal.astype(np.float64, copy=True)
    factors: List[Optional[float]] = [None] * len(CANONICAL_LEADS)
    assumed: List[str] = []
    for index, lead in enumerate(CANONICAL_LEADS):
        if not header_lead_mask[index]:
            continue
        factor = _unit_to_mv_factor(source_units[index])
        if factor is None:
            if unknown_units == "assume_mV":
                factor = 1.0
                assumed.append(lead)
            else:
                raise ValueError(
                    f"Unknown physical unit {source_units[index]!r} for lead {lead}; "
                    "use --unknown-units assume_mV only after verifying the header"
                )
        output[:, index] *= factor
        factors[index] = float(factor)
    return output, factors, assumed


def _finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def compute_lead_qc(
    signal_mv: np.ndarray,
    header_lead_mask: np.ndarray,
    fs: float,
    thresholds: LeadQCThresholds,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, List[str]]]:
    quality = np.zeros(len(CANONICAL_LEADS), dtype=bool)
    metrics: List[Dict[str, Any]] = []
    failures: Dict[str, List[str]] = {}

    for index, lead in enumerate(CANONICAL_LEADS):
        raw = np.asarray(signal_mv[:, index], dtype=np.float64)
        finite_mask = np.isfinite(raw)
        finite = raw[finite_mask]
        reasons: List[str] = []
        n_total = int(raw.size)
        n_finite = int(finite.size)
        finite_fraction = float(n_finite / max(n_total, 1))

        metric: Dict[str, Any] = {
            "lead": lead,
            "header_present": bool(header_lead_mask[index]),
            "n_total": n_total,
            "n_finite": n_finite,
            "finite_fraction": finite_fraction,
        }
        if not header_lead_mask[index]:
            reasons.append("missing_from_header")
        elif n_finite < 2:
            reasons.append("insufficient_finite_samples")
        else:
            median = float(np.median(finite))
            std = float(np.std(finite))
            p01, p99 = np.percentile(finite, [1.0, 99.0])
            robust_range = float(p99 - p01)
            max_abs = float(np.max(np.abs(finite)))

            adjacent_valid = finite_mask[:-1] & finite_mask[1:]
            adjacent_diff = np.abs(np.diff(raw)[adjacent_valid])
            flat_fraction = (
                float(np.mean(adjacent_diff <= thresholds.flat_step_epsilon_mv))
                if adjacent_diff.size else 1.0
            )
            if adjacent_diff.size:
                diff_center = np.median(adjacent_diff)
                noise_mad = float(1.4826 * np.median(np.abs(adjacent_diff - diff_center)))
            else:
                noise_mad = 0.0

            window = max(1, int(round(float(fs) * thresholds.baseline_window_seconds)))
            block_medians = []
            for start in range(0, n_total, window):
                block = _finite_values(raw[start:start + window])
                if block.size:
                    block_medians.append(float(np.median(block)))
            if len(block_medians) >= 2:
                lo, hi = np.percentile(np.asarray(block_medians), [5.0, 95.0])
                baseline_wander = float(hi - lo)
            else:
                baseline_wander = 0.0

            metric.update({
                "mean_mv": float(np.mean(finite)),
                "median_mv": median,
                "std_mv": std,
                "min_mv": float(np.min(finite)),
                "max_mv": float(np.max(finite)),
                "p01_mv": float(p01),
                "p99_mv": float(p99),
                "robust_range_mv": robust_range,
                "max_abs_mv": max_abs,
                "flat_step_fraction": flat_fraction,
                "noise_mad_diff_mv": noise_mad,
                "baseline_wander_mv": baseline_wander,
                "baseline_window_samples": window,
            })

            if finite_fraction < thresholds.min_finite_fraction:
                reasons.append("excess_nonfinite")
            if (
                std < thresholds.min_std_mv
                or robust_range < thresholds.min_robust_range_mv
                or flat_fraction > thresholds.max_flat_step_fraction
            ):
                reasons.append("flatline_or_near_flatline")
            if max_abs > thresholds.max_abs_amplitude_mv:
                reasons.append("extreme_amplitude")
            if abs(median) > thresholds.max_abs_baseline_mv:
                reasons.append("extreme_baseline_offset")
            if baseline_wander > thresholds.max_baseline_wander_mv:
                reasons.append("extreme_baseline_wander")
            if noise_mad > thresholds.max_noise_mad_diff_mv:
                reasons.append("extreme_high_frequency_noise")

        quality[index] = bool(header_lead_mask[index] and not reasons)
        metric["quality_pass"] = bool(quality[index])
        metric["failure_reasons"] = list(reasons)
        metrics.append(_json_scalar(metric))
        if reasons:
            failures[lead] = reasons

    return quality, metrics, failures


def fill_nonfinite_for_signal_processing(
    signal_mv: np.ndarray,
    header_lead_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill gaps only for filtering/resampling; return the untouched validity mask."""
    value_mask = np.isfinite(signal_mv) & header_lead_mask[None, :]
    output = np.zeros_like(signal_mv, dtype=np.float64)
    x_index = np.arange(signal_mv.shape[0], dtype=np.float64)
    for lead_index in range(signal_mv.shape[1]):
        valid = value_mask[:, lead_index]
        if not valid.any():
            continue
        values = signal_mv[valid, lead_index]
        positions = x_index[valid]
        if values.size == 1:
            output[:, lead_index] = values[0]
        else:
            output[:, lead_index] = np.interp(x_index, positions, values)
    return output, value_mask


def _match_first_axis(array: np.ndarray, target_length: int, fill_value: Any = 0) -> np.ndarray:
    if array.shape[0] == target_length:
        return array
    shape = (target_length,) + array.shape[1:]
    output = np.full(shape, fill_value, dtype=array.dtype)
    n_copy = min(target_length, array.shape[0])
    output[:n_copy] = array[:n_copy]
    return output


def _linear_resample(
    signal: np.ndarray,
    value_mask: np.ndarray,
    old_fs: float,
    new_fs: float,
    new_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    old_t = np.arange(signal.shape[0], dtype=np.float64) / old_fs
    new_t = np.arange(new_length, dtype=np.float64) / new_fs
    output = np.zeros((new_length, signal.shape[1]), dtype=np.float64)
    mask_float = np.zeros_like(output)
    for lead_index in range(signal.shape[1]):
        output[:, lead_index] = np.interp(new_t, old_t, signal[:, lead_index])
        mask_float[:, lead_index] = np.interp(
            new_t, old_t, value_mask[:, lead_index].astype(np.float64)
        )
    return output, mask_float >= 0.999999


def resample_waveform_and_mask(
    signal: np.ndarray,
    value_mask: np.ndarray,
    raw_fs: float,
    target_fs: Optional[float],
    fallback: str,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    if target_fs is None or math.isclose(raw_fs, target_fs, rel_tol=0.0, abs_tol=1e-9):
        return signal, value_mask, float(raw_fs), {
            "was_resampled": False,
            "resampling_method": "none",
            "resampling_warning": None,
        }
    if not np.isfinite(raw_fs) or raw_fs <= 0 or target_fs <= 0:
        raise ValueError(f"Cannot resample with raw_fs={raw_fs!r}, target_fs={target_fs!r}")

    new_length = max(1, int(round(signal.shape[0] * float(target_fs) / float(raw_fs))))
    ratio = Fraction(float(target_fs) / float(raw_fs)).limit_denominator(10000)
    try:
        import scipy  # type: ignore
        from scipy.signal import resample_poly  # type: ignore

        output = resample_poly(
            signal, ratio.numerator, ratio.denominator, axis=0, padtype="line"
        )
        mask_float = resample_poly(
            value_mask.astype(np.float64),
            ratio.numerator,
            ratio.denominator,
            axis=0,
            padtype="constant",
        )
        output = _match_first_axis(np.asarray(output, dtype=np.float64), new_length, 0.0)
        output_mask = _match_first_axis(np.asarray(mask_float >= 0.999, dtype=bool), new_length, False)
        return output, output_mask, float(target_fs), {
            "was_resampled": True,
            "resampling_method": "scipy.signal.resample_poly",
            "resampling_warning": None,
            "resample_up": int(ratio.numerator),
            "resample_down": int(ratio.denominator),
            "scipy_version": str(getattr(scipy, "__version__", "unknown")),
        }
    except Exception as exc:
        if fallback != "linear":
            raise RuntimeError(
                "Anti-aliased resampling was requested but scipy.signal.resample_poly "
                f"was unavailable or failed ({exc!r}). Install SciPy, preserve native "
                "sampling with --target-fs none, or explicitly opt into the unsafe "
                "fallback with --resample-fallback linear."
            ) from exc
        output, output_mask = _linear_resample(
            signal, value_mask, raw_fs, float(target_fs), new_length
        )
        return output, output_mask, float(target_fs), {
            "was_resampled": True,
            "resampling_method": "linear_interpolation_non_antialiased",
            "resampling_warning": (
                "EXPLICIT FALLBACK: linear interpolation is not anti-aliased and may "
                "distort/downsample high-frequency ECG content"
            ),
            "anti_alias_error": repr(exc),
        }


def _resample_boolean_time_mask(
    mask: np.ndarray,
    old_length: int,
    new_length: int,
) -> np.ndarray:
    if old_length == new_length:
        return mask.astype(bool, copy=True)
    old_x = np.arange(old_length, dtype=np.float64)
    new_x = np.linspace(0.0, max(old_length - 1, 0), new_length)
    interpolated = np.interp(new_x, old_x, mask.astype(np.float64))
    return interpolated >= 0.999999


def apply_length_policy(
    signal: np.ndarray,
    value_mask: np.ndarray,
    sample_mask: np.ndarray,
    header_sample_mask: np.ndarray,
    target_samples: Optional[int],
    crop_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    n_samples = int(signal.shape[0])
    if target_samples is None:
        return signal, value_mask, sample_mask, header_sample_mask, {
            "length_policy": "preserve_full",
            "crop_start": 0,
            "n_samples_before_length_policy": n_samples,
            "n_samples_output": n_samples,
            "n_padding_samples": 0,
        }
    if target_samples <= 0:
        raise ValueError("target_samples must be positive or None")

    if n_samples >= target_samples:
        if crop_mode == "start":
            start = 0
        elif crop_mode == "end":
            start = n_samples - target_samples
        elif crop_mode == "center":
            start = (n_samples - target_samples) // 2
        else:
            raise ValueError(f"Unknown crop_mode={crop_mode!r}")
        stop = start + target_samples
        return (
            signal[start:stop],
            value_mask[start:stop],
            sample_mask[start:stop],
            header_sample_mask[start:stop],
            {
                "length_policy": "crop" if n_samples > target_samples else "exact",
                "crop_start": int(start),
                "n_samples_before_length_policy": n_samples,
                "n_samples_output": int(target_samples),
                "n_padding_samples": 0,
            },
        )

    output = _match_first_axis(signal, target_samples, 0.0)
    output_value_mask = _match_first_axis(value_mask, target_samples, False)
    output_sample_mask = _match_first_axis(sample_mask, target_samples, False)
    output_header_mask = _match_first_axis(header_sample_mask, target_samples, False)
    return output, output_value_mask, output_sample_mask, output_header_mask, {
        "length_policy": "right_pad",
        "crop_start": 0,
        "n_samples_before_length_policy": n_samples,
        "n_samples_output": int(target_samples),
        "n_padding_samples": int(target_samples - n_samples),
    }


def compute_channel_stats(
    signal: np.ndarray,
    value_mask: np.ndarray,
    time_mask: np.ndarray,
    header_lead_mask: np.ndarray,
    quality_lead_mask: np.ndarray,
    person_id: str,
    record_id: str,
    artifact_key: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, lead in enumerate(CANONICAL_LEADS):
        valid = value_mask[:, index] & time_mask & bool(header_lead_mask[index])
        values = np.asarray(signal[valid, index], dtype=np.float64)
        row: Dict[str, Any] = {
            "person_id": person_id,
            "record_id": record_id,
            "artifact_key": artifact_key,
            "lead": lead,
            "header_present": bool(header_lead_mask[index]),
            "quality_pass": bool(quality_lead_mask[index]),
            "include_for_train_scaler": bool(quality_lead_mask[index] and values.size > 0),
            "valid_count": int(values.size),
            "sum_mv": float(values.sum(dtype=np.float64)) if values.size else 0.0,
            "sum_squares_mv2": float(np.square(values).sum(dtype=np.float64)) if values.size else 0.0,
            "mean_mv": float(values.mean()) if values.size else None,
            "std_mv": float(values.std()) if values.size else None,
            "min_mv": float(values.min()) if values.size else None,
            "max_mv": float(values.max()) if values.size else None,
        }
        rows.append(_json_scalar(row))
    return rows


def _artifact_key(row: Mapping[str, Any], duplicate_policy: str) -> str:
    pid = _safe_file_token(str(row["person_id"]))
    if duplicate_policy == "select":
        return pid
    return f"{pid}__{row['record_id']}"


def prepare_manifest(
    manifest: pd.DataFrame,
    dataset_root: Path,
    duplicate_policy: str,
    selection_policy: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = sorted(REQUIRED_MANIFEST_COLUMNS - set(manifest.columns))
    if missing:
        raise ValueError(f"ECG manifest missing required columns: {missing}")
    frame = manifest.copy().reset_index(drop=True)
    frame["manifest_row_index"] = np.arange(len(frame), dtype=np.int64)
    frame["person_id"] = frame["person_id"].map(as_person_id)
    if (frame["person_id"] == "").any():
        bad = frame.loc[frame["person_id"] == "", "manifest_row_index"].tolist()
        raise ValueError(f"ECG manifest contains empty person_id at rows {bad[:10]}")

    for column in ("wfdb_hea_filepath", "wfdb_dat_filepath"):
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    frame["hea_exists"] = frame["wfdb_hea_filepath"].map(
        lambda value: resolve_path(dataset_root, value).is_file()
    )
    frame["dat_exists"] = frame["wfdb_dat_filepath"].map(
        lambda value: resolve_path(dataset_root, value).is_file()
    )
    frame["path_pair_exists"] = frame["hea_exists"] & frame["dat_exists"]
    numeric_columns = [column for column in NUMERIC_FEATURES if column in frame.columns]
    if numeric_columns:
        frame["numeric_complete_count"] = frame[numeric_columns].apply(
            lambda row: int(pd.to_numeric(row, errors="coerce").notna().sum()), axis=1
        )
    else:
        frame["numeric_complete_count"] = 0

    if selection_policy == "completeness_then_path":
        sort_columns = [
            "person_id", "path_pair_exists", "numeric_complete_count",
            "wfdb_hea_filepath", "wfdb_dat_filepath", "manifest_row_index",
        ]
        ascending = [True, False, False, True, True, True]
    elif selection_policy == "lexical":
        sort_columns = [
            "person_id", "wfdb_hea_filepath", "wfdb_dat_filepath", "manifest_row_index",
        ]
        ascending = [True, True, True, True]
    else:
        raise ValueError(f"Unknown selection_policy={selection_policy!r}")
    ranked = frame.sort_values(sort_columns, ascending=ascending, kind="mergesort").copy()
    ranked["duplicate_rank"] = ranked.groupby("person_id", sort=False).cumcount() + 1
    ranked["n_records_for_person"] = ranked.groupby("person_id")["person_id"].transform("size")
    ranked["is_primary_record"] = ranked["duplicate_rank"] == 1
    ranked["record_id"] = ranked.apply(
        lambda row: canonical_json_hash({
            "hea": row["wfdb_hea_filepath"],
            "dat": row["wfdb_dat_filepath"],
            "manifest_row_index": int(row["manifest_row_index"]),
        })[:16],
        axis=1,
    )

    if duplicate_policy == "select":
        selected = ranked[ranked["is_primary_record"]].copy()
        ranked["kept_for_processing"] = ranked["is_primary_record"]
    elif duplicate_policy == "keep_all":
        selected = ranked.copy()
        ranked["kept_for_processing"] = True
    else:
        raise ValueError(f"Unknown duplicate_policy={duplicate_policy!r}")
    selected["artifact_key"] = selected.apply(
        lambda row: _artifact_key(row, duplicate_policy), axis=1
    )
    duplicate_audit = ranked[ranked["n_records_for_person"] > 1].copy()
    return selected.reset_index(drop=True), duplicate_audit.reset_index(drop=True), ranked.reset_index(drop=True)


def _torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch versions
        return torch.load(path, map_location="cpu")


def _atomic_torch_save(payload: Dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _source_fingerprint(files: Iterable[Mapping[str, Any]]) -> str:
    return canonical_json_hash({"source_files": list(files)})


def _cache_status(
    output_path: Path,
    config_hash: str,
    source_files: List[Dict[str, Any]],
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    if not output_path.exists():
        return False, None, "missing"
    try:
        cached = _torch_load(output_path)
    except Exception as exc:
        return False, None, f"unreadable:{exc!r}"
    if cached.get("artifact_config_hash") != config_hash:
        return False, cached, "config_hash_mismatch"
    if cached.get("source_fingerprint") != _source_fingerprint(source_files):
        return False, cached, "source_metadata_mismatch"
    return True, cached, "valid"


def _summary_from_artifact(
    artifact: Mapping[str, Any],
    output_path: Path,
    status: str,
    row: Mapping[str, Any],
) -> Dict[str, Any]:
    record = artifact.get("record_metadata", {})
    lead_mask = np.asarray(artifact["ecg_lead_mask"], dtype=bool)
    header_lead_mask = np.asarray(artifact["ecg_header_lead_mask"], dtype=bool)
    quality_mask = np.asarray(artifact["ecg_lead_quality_mask"], dtype=bool)
    sample_mask = np.asarray(artifact["ecg_sample_mask"], dtype=bool)
    header_sample_mask = np.asarray(artifact["ecg_header_sample_mask"], dtype=bool)
    value_mask = np.asarray(artifact["ecg_value_mask"], dtype=bool)
    numeric_mask = np.asarray(artifact["ecg_numeric_mask"], dtype=bool)
    return {
        "status": status,
        "person_id": str(artifact["person_id"]),
        "record_id": str(artifact["record_id"]),
        "artifact_key": str(artifact["artifact_key"]),
        "artifact_path": str(output_path),
        "duplicate_rank": int(row["duplicate_rank"]),
        "n_records_for_person": int(row["n_records_for_person"]),
        "is_primary_record": bool(row["is_primary_record"]),
        "wfdb_hea_filepath": str(row["wfdb_hea_filepath"]),
        "wfdb_dat_filepath": str(row["wfdb_dat_filepath"]),
        "fs_raw": record.get("fs_raw"),
        "fs_output": record.get("fs_output"),
        "n_samples_header_raw": record.get("n_samples_header_raw"),
        "n_samples_loaded_raw": record.get("n_samples_loaded_raw"),
        "n_samples_after_resample": record.get("n_samples_after_resample"),
        "n_samples_output": int(np.asarray(artifact["ecg_waveform"]).shape[0]),
        "n_sample_positions_loaded": int(sample_mask.sum()),
        "n_sample_positions_in_header": int(header_sample_mask.sum()),
        "n_valid_values": int(value_mask.sum()),
        "n_header_leads": int(header_lead_mask.sum()),
        "n_quality_leads": int(quality_mask.sum()),
        "n_model_leads": int(lead_mask.sum()),
        "n_numeric_values": int(numeric_mask.sum()),
        "header_lead_mask": "".join("1" if x else "0" for x in header_lead_mask),
        "quality_lead_mask": "".join("1" if x else "0" for x in quality_mask),
        "model_lead_mask": "".join("1" if x else "0" for x in lead_mask),
        "resampling_method": record.get("resampling_method"),
        "resampling_warning": record.get("resampling_warning"),
        "participant_position": artifact.get("manifest_metadata", {}).get("participant_position"),
        "participant_position_degrees_metadata": artifact.get("manifest_metadata", {}).get(
            "participant_position_degrees"
        ),
        "artifact_config_hash": artifact.get("artifact_config_hash"),
        "source_fingerprint": artifact.get("source_fingerprint"),
        "channel_stats": artifact.get("channel_stats", []),
    }


def process_record(
    row: Dict[str, Any],
    dataset_root_text: str,
    manifest_path_text: str,
    waveform_dir_text: str,
    run_config: Dict[str, Any],
    qc_payload: Dict[str, Any],
    cache_policy: str,
) -> Dict[str, Any]:
    if torch is None:
        raise ImportError("PyTorch is required to serialize ECG .pt artifacts")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    person_id = str(row["person_id"])
    record_id = str(row["record_id"])
    artifact_key = str(row["artifact_key"])
    output_path = Path(waveform_dir_text) / f"{artifact_key}.pt"
    dataset_root = Path(dataset_root_text)
    hea_path = resolve_path(dataset_root, row["wfdb_hea_filepath"])
    dat_path = resolve_path(dataset_root, row["wfdb_dat_filepath"])
    manifest_path = Path(manifest_path_text)
    sources = source_file_metadata([manifest_path, hea_path, dat_path])
    config_hash = str(run_config["config_hash"])

    try:
        if cache_policy != "rebuild":
            valid, cached, reason = _cache_status(output_path, config_hash, sources)
            if valid and cached is not None:
                return _summary_from_artifact(cached, output_path, "cached", row)
            if output_path.exists() and cache_policy == "error_on_stale":
                raise RuntimeError(f"Stale ECG cache {output_path}: {reason}")

        raw_signal, record_metadata = read_wfdb_physical(hea_path, dat_path)
        canonical, header_lead_mask, source_units, source_indices, lead_audit = canonicalize_leads(
            raw_signal,
            record_metadata.get("sig_name_raw", []),
            record_metadata.get("units_raw", []),
        )
        canonical_mv, unit_factors, assumed_unit_leads = convert_canonical_to_mv(
            canonical,
            header_lead_mask,
            source_units,
            str(run_config["unknown_units"]),
        )

        thresholds = LeadQCThresholds(**qc_payload)
        quality_mask, qc_metrics, qc_failures = compute_lead_qc(
            canonical_mv,
            header_lead_mask,
            float(record_metadata["fs_raw"]),
            thresholds,
        )
        processing_signal, raw_value_mask = fill_nonfinite_for_signal_processing(
            canonical_mv, header_lead_mask
        )
        output_signal, output_value_mask, output_fs, resample_meta = resample_waveform_and_mask(
            processing_signal,
            raw_value_mask,
            float(record_metadata["fs_raw"]),
            run_config.get("target_fs"),
            str(run_config["resample_fallback"]),
        )

        raw_loaded_mask = np.ones(raw_signal.shape[0], dtype=bool)
        raw_header_length = int(record_metadata["n_samples_header_raw"])
        raw_header_mask = np.arange(raw_signal.shape[0]) < raw_header_length
        loaded_mask = _resample_boolean_time_mask(
            raw_loaded_mask, raw_signal.shape[0], output_signal.shape[0]
        )
        header_sample_mask = _resample_boolean_time_mask(
            raw_header_mask, raw_signal.shape[0], output_signal.shape[0]
        )

        (
            output_signal,
            output_value_mask,
            loaded_mask,
            header_sample_mask,
            length_meta,
        ) = apply_length_policy(
            output_signal,
            output_value_mask,
            loaded_mask,
            header_sample_mask,
            run_config.get("target_samples"),
            str(run_config["crop_mode"]),
        )
        output_signal = np.asarray(output_signal, dtype=np.float32)
        output_value_mask = np.asarray(output_value_mask, dtype=bool)
        time_mask = np.asarray(loaded_mask & header_sample_mask, dtype=bool)
        # Header and loaded-sample support are part of value validity, not only
        # an advisory downstream mask. This prevents suspicious samples beyond
        # a declared header length from remaining non-zero in the cache.
        output_value_mask &= time_mask[:, None]
        output_signal[~output_value_mask] = 0.0

        output_has_values = (output_value_mask & time_mask[:, None]).any(axis=0)
        quality_mask = quality_mask & output_has_values
        for lead_index, lead in enumerate(CANONICAL_LEADS):
            if header_lead_mask[lead_index] and not output_has_values[lead_index]:
                qc_failures.setdefault(lead, []).append("no_valid_output_samples")
                qc_metrics[lead_index]["quality_pass"] = False
                qc_metrics[lead_index]["failure_reasons"] = list(qc_failures[lead])
        model_lead_mask = header_lead_mask & quality_mask & output_has_values

        numeric_values, numeric_mask = numeric_values_and_mask(row)
        manifest_metadata = {
            column: _json_scalar(row.get(column)) for column in MANIFEST_METADATA_COLUMNS
        }
        manifest_metadata["participant_position_degrees"] = parse_participant_position_degrees(
            row.get("participant_position")
        )
        manifest_metadata["participant_position_is_model_feature"] = False

        record_metadata.update({
            "fs_output": float(output_fs),
            "n_samples_after_resample": int(
                length_meta["n_samples_before_length_policy"]
            ),
            "n_samples_output": int(output_signal.shape[0]),
            "duration_seconds_raw": float(raw_signal.shape[0] / record_metadata["fs_raw"]),
            "duration_seconds_output": float(output_signal.shape[0] / output_fs),
            "source_units_by_canonical_lead": source_units,
            "canonical_units": ["mV"] * len(CANONICAL_LEADS),
            "unit_to_mv_factors": unit_factors,
            "unit_conversion_assumed_for_leads": assumed_unit_leads,
            "source_indices_by_canonical_lead": source_indices,
            "lead_mapping_audit": lead_audit,
            **resample_meta,
            **length_meta,
        })

        channel_stats = compute_channel_stats(
            output_signal,
            output_value_mask,
            time_mask,
            header_lead_mask,
            quality_mask,
            person_id,
            record_id,
            artifact_key,
        )
        source_fingerprint = _source_fingerprint(sources)
        artifact = {
            "artifact_format_version": 2,
            "artifact_config": run_config,
            "artifact_config_hash": config_hash,
            "source_files": sources,
            "source_fingerprint": source_fingerprint,
            "person_id": person_id,
            "record_id": record_id,
            "artifact_key": artifact_key,
            "ecg_waveform": torch.from_numpy(output_signal),
            "waveform_units": "mV",
            "ecg_sample_mask": torch.from_numpy(np.asarray(loaded_mask, dtype=bool)),
            "ecg_header_sample_mask": torch.from_numpy(
                np.asarray(header_sample_mask, dtype=bool)
            ),
            "ecg_time_mask": torch.from_numpy(time_mask),
            "ecg_value_mask": torch.from_numpy(output_value_mask),
            "ecg_header_lead_mask": torch.from_numpy(header_lead_mask),
            "ecg_lead_quality_mask": torch.from_numpy(quality_mask),
            "ecg_lead_mask": torch.from_numpy(model_lead_mask),
            "ecg_numeric": torch.from_numpy(numeric_values),
            "ecg_numeric_mask": torch.from_numpy(numeric_mask),
            "lead_names": list(CANONICAL_LEADS),
            "numeric_feature_names": list(NUMERIC_FEATURES),
            "lead_qc_thresholds": asdict(thresholds),
            "lead_qc_metrics": qc_metrics,
            "lead_qc_failure_reasons": qc_failures,
            "channel_stats": channel_stats,
            "source_paths": {
                "manifest": str(manifest_path),
                "wfdb_hea_filepath": str(hea_path),
                "wfdb_dat_filepath": str(dat_path),
            },
            "manifest_metadata": manifest_metadata,
            "record_metadata": _json_scalar(record_metadata),
        }
        _atomic_torch_save(artifact, output_path)
        return _summary_from_artifact(artifact, output_path, "processed", row)
    except Exception as exc:
        return {
            "status": "failed",
            "person_id": person_id,
            "record_id": record_id,
            "artifact_key": artifact_key,
            "artifact_path": str(output_path),
            "duplicate_rank": int(row.get("duplicate_rank", 0)),
            "wfdb_hea_filepath": str(row.get("wfdb_hea_filepath", "")),
            "wfdb_dat_filepath": str(row.get("wfdb_dat_filepath", "")),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
            "artifact_config_hash": config_hash,
        }


def _numeric_audit_table(
    selected: pd.DataFrame, artifact_config_hash: str
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for row in selected.to_dict("records"):
        values, mask = numeric_values_and_mask(row)
        item: Dict[str, Any] = {
            "person_id": str(row["person_id"]),
            "record_id": str(row["record_id"]),
            "artifact_key": str(row["artifact_key"]),
            "duplicate_rank": int(row["duplicate_rank"]),
            "is_primary_record": bool(row["is_primary_record"]),
            "artifact_config_hash": artifact_config_hash,
            "participant_position": _json_scalar(row.get("participant_position")),
            "participant_position_degrees_metadata": parse_participant_position_degrees(
                row.get("participant_position")
            ),
        }
        for index, name in enumerate(NUMERIC_FEATURES):
            item[f"ecg__{name}"] = float(values[index]) if mask[index] else np.nan
            item[f"ecg__{name}__mask"] = int(mask[index])
        rows.append(item)
    return pd.DataFrame(rows)


def _build_run_config(args: argparse.Namespace, qc: LeadQCThresholds) -> Dict[str, Any]:
    return artifact_config(
        modality="cardiac_ecg",
        extra={
            "builder": "build_ecg_waveform_features.py",
            "representation": "raw_physical_millivolts",
            "target_fs": args.target_fs,
            "target_samples": args.target_samples,
            "crop_mode": args.crop_mode,
            "resample_fallback": args.resample_fallback,
            "unknown_units": args.unknown_units,
            "duplicate_policy": args.duplicate_policy,
            "selection_policy": args.selection_policy,
            "canonical_leads": list(CANONICAL_LEADS),
            "numeric_features": list(NUMERIC_FEATURES),
            "participant_position_is_model_feature": False,
            "lead_qc_thresholds": asdict(qc),
        },
    )


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Path to ECG manifest TSV")
    parser.add_argument("--dataset-root", "--dataset_root", dest="dataset_root", required=True)
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir", required=True)
    parser.add_argument(
        "--target-fs", "--target_fs", dest="target_fs",
        type=_optional_positive_float, default=None,
        help="Output Hz; default none preserves native fs",
    )
    parser.add_argument(
        "--target-samples", "--target_samples", dest="target_samples",
        type=_optional_positive_int, default=None,
        help="Optional crop/pad length; default none preserves the full recording",
    )
    parser.add_argument("--crop-mode", choices=("center", "start", "end"), default="center")
    parser.add_argument(
        "--resample-fallback", choices=("error", "linear"), default="error",
        help="Default error prevents silent non-antialiased resampling",
    )
    parser.add_argument(
        "--unknown-units", choices=("error", "assume_mV"), default="error",
        help="Default error prevents silently treating digital/unknown units as mV",
    )
    parser.add_argument(
        "--duplicate-policy", choices=("select", "keep_all"), default="select",
    )
    parser.add_argument(
        "--selection-policy", choices=("completeness_then_path", "lexical"),
        default="completeness_then_path",
    )
    parser.add_argument(
        "--cache-policy", choices=("reuse", "rebuild", "error_on_stale"), default="reuse",
        help="reuse skips only when config hash and source file metadata both match",
    )
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-progress", "--no_progress", dest="no_progress", action="store_true")

    # Conservative QC thresholds remain configurable and are hashed.
    parser.add_argument("--qc-min-finite-fraction", type=float, default=0.98)
    parser.add_argument("--qc-min-std-mv", type=float, default=0.005)
    parser.add_argument("--qc-min-robust-range-mv", type=float, default=0.02)
    parser.add_argument("--qc-flat-step-epsilon-mv", type=float, default=1e-5)
    parser.add_argument("--qc-max-flat-step-fraction", type=float, default=0.995)
    parser.add_argument("--qc-max-abs-amplitude-mv", type=float, default=25.0)
    parser.add_argument("--qc-max-abs-baseline-mv", type=float, default=5.0)
    parser.add_argument("--qc-max-baseline-wander-mv", type=float, default=5.0)
    parser.add_argument("--qc-max-noise-mad-diff-mv", type=float, default=2.5)
    parser.add_argument("--qc-baseline-window-seconds", type=float, default=2.0)
    return parser


def main() -> None:
    args = _arg_parser().parse_args()
    if torch is None:
        raise SystemExit("PyTorch is required to serialize ECG .pt artifacts")
    if args.num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be >= 1")

    qc = LeadQCThresholds(
        min_finite_fraction=args.qc_min_finite_fraction,
        min_std_mv=args.qc_min_std_mv,
        min_robust_range_mv=args.qc_min_robust_range_mv,
        flat_step_epsilon_mv=args.qc_flat_step_epsilon_mv,
        max_flat_step_fraction=args.qc_max_flat_step_fraction,
        max_abs_amplitude_mv=args.qc_max_abs_amplitude_mv,
        max_abs_baseline_mv=args.qc_max_abs_baseline_mv,
        max_baseline_wander_mv=args.qc_max_baseline_wander_mv,
        max_noise_mad_diff_mv=args.qc_max_noise_mad_diff_mv,
        baseline_window_seconds=args.qc_baseline_window_seconds,
    )
    run_config = _build_run_config(args, qc)

    manifest_path = Path(args.manifest).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    waveform_dir = out_dir / "ecg_waveforms"
    out_dir.mkdir(parents=True, exist_ok=True)
    waveform_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "artifact_config.json", run_config)

    manifest = pd.read_csv(
        manifest_path, sep="\t", low_memory=False, dtype={"person_id": "string"}
    )
    selected, duplicate_audit, ranked = prepare_manifest(
        manifest,
        dataset_root,
        duplicate_policy=args.duplicate_policy,
        selection_policy=args.selection_policy,
    )
    if args.limit is not None:
        selected = selected.head(args.limit).copy()

    duplicate_audit.to_csv(out_dir / "ecg_duplicates.csv", index=False)
    numeric_table = _numeric_audit_table(
        selected, str(run_config["config_hash"])
    )

    records = selected.to_dict("records")
    results: List[Dict[str, Any]] = []
    worker_args = (
        str(dataset_root), str(manifest_path), str(waveform_dir),
        run_config, asdict(qc), args.cache_policy,
    )

    if args.num_workers == 1:
        iterator: Iterable[Any] = records
        if tqdm is not None and not args.no_progress:
            iterator = tqdm(
                records, total=len(records), desc="ECG V2 preprocessing",
                unit="record", dynamic_ncols=True,
            )
        for row in iterator:
            results.append(process_record(row, *worker_args))
            if tqdm is not None and not args.no_progress:
                failed = sum(x["status"] == "failed" for x in results)
                cached = sum(x["status"] == "cached" for x in results)
                iterator.set_postfix(processed=len(results) - failed - cached, cached=cached, failed=failed)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_identity = {
                executor.submit(process_record, row, *worker_args): (
                    str(row["person_id"]), str(row["record_id"])
                )
                for row in records
            }
            iterator = as_completed(future_to_identity)
            if tqdm is not None and not args.no_progress:
                iterator = tqdm(
                    iterator, total=len(future_to_identity), desc="ECG V2 preprocessing",
                    unit="record", dynamic_ncols=True,
                )
            for future in iterator:
                identity = future_to_identity[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # process_record normally catches its own failures
                    results.append({
                        "status": "failed",
                        "person_id": identity[0],
                        "record_id": identity[1],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=8),
                        "artifact_config_hash": run_config["config_hash"],
                    })
                if tqdm is not None and not args.no_progress:
                    failed = sum(x["status"] == "failed" for x in results)
                    cached = sum(x["status"] == "cached" for x in results)
                    iterator.set_postfix(processed=len(results) - failed - cached, cached=cached, failed=failed)

    results.sort(key=lambda row: (
        str(row.get("person_id", "")), int(row.get("duplicate_rank", 0)),
        str(row.get("record_id", "")),
    ))
    failures = [row for row in results if row.get("status") == "failed"]
    successful = [row for row in results if row.get("status") in {"processed", "cached"}]

    channel_stats: List[Dict[str, Any]] = []
    for row in successful:
        for item in row.pop("channel_stats", []):
            item["artifact_config_hash"] = row.get("artifact_config_hash")
            channel_stats.append(item)
    index = pd.DataFrame(successful)
    if not index.empty:
        index = index.sort_values(["person_id", "duplicate_rank", "record_id"], kind="mergesort")
    index.to_csv(out_dir / "ecg_index.csv", index=False)
    pd.DataFrame(channel_stats).to_csv(out_dir / "ecg_channel_stats.csv", index=False)

    status_by_record = {
        (str(row.get("person_id", "")), str(row.get("record_id", ""))): row.get("status")
        for row in results
    }
    if not numeric_table.empty:
        numeric_table["waveform_status"] = numeric_table.apply(
            lambda row: status_by_record.get((str(row["person_id"]), str(row["record_id"])), "not_run"),
            axis=1,
        )
    numeric_table.to_csv(out_dir / "ecg_numeric_features_wide.csv", index=False)
    _write_json(out_dir / "ecg_failures.json", failures)

    n_processed = sum(row.get("status") == "processed" for row in results)
    n_cached = sum(row.get("status") == "cached" for row in results)
    schema = {
        "artifact_config": run_config,
        "raw_modality": "cardiac_ecg_12lead_waveform",
        "model_representation": "variable_or_fixed_length_raw_physical_mv",
        "waveform": {
            "key": "ecg_waveform",
            "dtype": "float32",
            "shape": [args.target_samples, 12],
            "units": "mV",
            "normalization": "none; fit scaler from train-only ecg_channel_stats.csv",
        },
        "masks": {
            "ecg_sample_mask": "positions loaded from the WFDB signal before optional padding",
            "ecg_header_sample_mask": "positions declared inside the WFDB header",
            "ecg_time_mask": "sample_mask AND header_sample_mask",
            "ecg_value_mask": "finite source value support per sample and canonical lead",
            "ecg_header_lead_mask": "canonical lead explicitly found/assumed from WFDB header",
            "ecg_lead_quality_mask": "canonical lead passes conservative signal QC",
            "ecg_lead_mask": "header, quality and output-value combined model mask",
            "ecg_numeric_mask": "observed Rate/interval/axis value",
        },
        "canonical_leads": list(CANONICAL_LEADS),
        "numeric_feature_names": list(NUMERIC_FEATURES),
        "participant_position": {
            "retained_as_metadata": True,
            "included_in_numeric_model_features": False,
        },
        "lead_qc_thresholds": asdict(qc),
        "duplicate_policy": args.duplicate_policy,
        "selection_policy": args.selection_policy,
        "cache_policy": args.cache_policy,
        "n_manifest_rows": int(len(manifest)),
        "n_unique_people_manifest": int(manifest["person_id"].map(as_person_id).nunique()),
        "n_people_with_multiple_records": int(
            (ranked.groupby("person_id").size() > 1).sum()
        ),
        "n_duplicate_audit_rows": int(len(duplicate_audit)),
        "n_records_selected_for_run": int(len(selected)),
        "n_processed": int(n_processed),
        "n_cached": int(n_cached),
        "n_failed": int(len(failures)),
        "output_files": {
            "waveform_dir": str(waveform_dir),
            "index": str(out_dir / "ecg_index.csv"),
            "numeric": str(out_dir / "ecg_numeric_features_wide.csv"),
            "channel_stats": str(out_dir / "ecg_channel_stats.csv"),
            "duplicates": str(out_dir / "ecg_duplicates.csv"),
            "failures": str(out_dir / "ecg_failures.json"),
        },
        "notes": [
            "No participant-specific waveform z-score or amplitude normalization is applied.",
            "Aggregate channel statistics only after joining participants.tsv and retaining train rows.",
            "A cached artifact is reused only when both config hash and source size/mtime metadata match.",
            "Resampling is anti-aliased unless the operator explicitly requests the linear fallback.",
            "MoCA/CVD labels are not read or used by this task-independent builder.",
        ],
    }
    _write_json(out_dir / "ecg_schema.json", schema)

    print(json.dumps({
        "out_dir": str(out_dir),
        "artifact_config_hash": run_config["config_hash"],
        "n_manifest_rows": len(manifest),
        "n_records_selected_for_run": len(selected),
        "n_processed": n_processed,
        "n_cached": n_cached,
        "n_failed": len(failures),
        "target_fs": args.target_fs,
        "target_samples": args.target_samples,
        "duplicate_policy": args.duplicate_policy,
    }, indent=2, allow_nan=False))
    if failures[:5]:
        print("first failures:", [
            {key: item.get(key) for key in ("person_id", "record_id", "error_type", "error")}
            for item in failures[:5]
        ])


if __name__ == "__main__":
    main()
