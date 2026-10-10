#!/usr/bin/env python3
"""Fit downstream normalization statistics from official train participants.

Canonical OT2 V2 artifacts deliberately retain raw units.  This script is the
split-aware second stage: it reads ``participants.tsv``, retains only rows with
``recommended_split == train``, and fits reusable normalization parameters.

Supported optional sources:

* clinical wide table: per-feature median and IQR;
* ECG numeric wide table: per-feature median and IQR, respecting numeric masks;
* ECG channel sufficient-statistics table: sample-weighted mean/std per lead;
* environment artifacts: streaming mean/std using observed feature support;
* wearable artifacts: separate streaming mean/std for CGM and activity using
  their exact feature masks.

Storage fill values are never treated as observations.  Deterministic
``time__*`` features are explicitly marked as not scaled.  No canonical input
is modified, and longitudinal tensors are loaded one participant at a time.
PyTorch is imported lazily only when environment or wearable artifacts are
requested, so tabular/ECG-statistics-only use does not require PyTorch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .preprocessing_utils import (
        PREPROCESSING_VERSION,
        as_person_id,
        canonical_json_hash,
    )
except ImportError:  # Allow ``python preprocessing/fit_train_scalers.py``.
    from preprocessing_utils import (  # type: ignore
        PREPROCESSING_VERSION,
        as_person_id,
        canonical_json_hash,
    )


SCALER_FORMAT_VERSION = "aireadi_ot2_train_scalers_v2"
EPSILON = 1e-12
NON_SCALING_PREFIX = "time__"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                _json_safe(payload),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path, source: str) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{source} JSON does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Could not parse {source} JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{source} JSON must contain an object: {path}")
    return payload


def _normalise_id_series(values: pd.Series, source: str) -> pd.Series:
    result = values.map(as_person_id)
    if (result == "").any():
        raise ValueError(f"{source} contains an empty person_id")
    return result


@dataclass(frozen=True)
class FitPopulation:
    all_ids: frozenset[str]
    train_ids: frozenset[str]
    fit_person_ids_hash: str
    n_participants: int


def load_fit_population(path: str | Path) -> FitPopulation:
    participants = pd.read_csv(
        path,
        sep="\t",
        dtype={"person_id": "string"},
        keep_default_na=False,
    )
    required = {"person_id", "recommended_split"}
    missing = sorted(required - set(participants.columns))
    if missing:
        raise ValueError(f"participants.tsv missing required columns: {missing}")
    participants = participants.copy()
    participants["person_id"] = _normalise_id_series(
        participants["person_id"], "participants.tsv"
    )
    duplicate_ids = sorted(
        participants.loc[
            participants["person_id"].duplicated(keep=False), "person_id"
        ].unique()
    )
    if duplicate_ids:
        raise ValueError(
            "participants.tsv must have one row per person_id; duplicates: "
            f"{duplicate_ids[:10]}"
        )
    split = participants["recommended_split"].astype(str).str.strip().str.lower()
    train_ids = frozenset(participants.loc[split == "train", "person_id"])
    if not train_ids:
        raise ValueError("participants.tsv contains no recommended_split=train rows")
    all_ids = frozenset(participants["person_id"])
    fit_hash = canonical_json_hash({"person_ids": sorted(train_ids)})
    return FitPopulation(
        all_ids=all_ids,
        train_ids=train_ids,
        fit_person_ids_hash=fit_hash,
        n_participants=int(len(participants)),
    )


def _validate_known_ids(
    frame: pd.DataFrame, population: FitPopulation, source: str
) -> pd.DataFrame:
    if "person_id" not in frame.columns:
        raise ValueError(f"{source} is missing person_id")
    frame = frame.copy()
    frame["person_id"] = _normalise_id_series(frame["person_id"], source)
    unknown = sorted(set(frame["person_id"]) - set(population.all_ids))
    if unknown:
        raise ValueError(
            f"{source} contains person_id values absent from participants.tsv: "
            f"{unknown[:10]}"
        )
    return frame


def _feature_entry_robust(values: np.ndarray, semantics: str) -> Dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    count = int(values.size)
    if count == 0:
        return {
            "count": 0,
            "center": 0.0,
            "scale": 1.0,
            "fallback": "no_observed_train_values",
            "scale_applied": True,
            "semantics": semantics,
        }
    q1, median, q3 = np.percentile(values, [25.0, 50.0, 75.0])
    iqr = float(q3 - q1)
    fallback: Optional[str] = None
    scale = iqr
    if not np.isfinite(scale) or scale <= EPSILON:
        scale = 1.0
        fallback = "unit_scale_zero_or_nonfinite_iqr"
    return {
        "count": count,
        "center": float(median),
        "scale": float(scale),
        "fallback": fallback,
        "scale_applied": True,
        "q1": float(q1),
        "q3": float(q3),
        "semantics": semantics,
    }


def fit_robust_features(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    masks: Optional[Mapping[str, np.ndarray]] = None,
    semantics: str = "subtract train median and divide by train IQR",
) -> Dict[str, Any]:
    features: Dict[str, Any] = {}
    for feature_name in feature_names:
        if feature_name not in frame.columns:
            raise ValueError(f"Missing requested feature column: {feature_name}")
        numeric = pd.to_numeric(frame[feature_name], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if masks is not None and feature_name in masks:
            mask = validated_boolean_mask(
                masks[feature_name], f"robust feature mask {feature_name}"
            )
            if mask.shape != numeric.shape:
                raise ValueError(f"Mask shape mismatch for feature {feature_name}")
            numeric = numeric[mask]
        features[str(feature_name)] = _feature_entry_robust(numeric, semantics)
    return {
        "method": "median_iqr",
        "feature_order": [str(name) for name in feature_names],
        "features": features,
        "semantics": semantics,
        "missing_value_application": (
            "impute a missing tabular value with center before scaling, yielding normalized zero"
        ),
    }


def validated_boolean_mask(value: Any, source: str) -> np.ndarray:
    """Accept only native booleans or finite numeric 0/1 mask values."""
    array = np.asarray(value)
    if array.dtype.kind == "b":
        return array.astype(bool, copy=False)
    if array.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(
            f"{source}: mask dtype must be bool or numeric 0/1, got {array.dtype}"
        )
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError(f"{source}: mask contains NaN or infinity")
    if not np.isin(array, [0, 1]).all():
        raise ValueError(f"{source}: mask contains values other than 0/1")
    return array.astype(bool, copy=False)


class StreamingMoments:
    """Per-feature float64 sufficient statistics updated one artifact at a time."""

    def __init__(self, feature_names: Sequence[str]) -> None:
        names = [str(name) for name in feature_names]
        if not names or len(names) != len(set(names)):
            raise ValueError("Streaming feature names must be non-empty and unique")
        self.feature_names = names
        self.count = np.zeros(len(names), dtype=np.int64)
        self.total = np.zeros(len(names), dtype=np.float64)
        self.total_squares = np.zeros(len(names), dtype=np.float64)

    def update(self, values: np.ndarray, mask: np.ndarray, source: str) -> None:
        values = np.asarray(values)
        mask = validated_boolean_mask(mask, f"{source} feature mask")
        if values.ndim != 2:
            raise ValueError(f"{source}: expected a 2D feature tensor, got {values.shape}")
        if values.shape != mask.shape:
            raise ValueError(
                f"{source}: feature/mask shape mismatch {values.shape} vs {mask.shape}"
            )
        if values.shape[1] != len(self.feature_names):
            raise ValueError(
                f"{source}: expected {len(self.feature_names)} features, got {values.shape[1]}"
            )
        numeric = values.astype(np.float64, copy=False)
        invalid_observed = mask & ~np.isfinite(numeric)
        if np.any(invalid_observed):
            bad = np.argwhere(invalid_observed)[0]
            raise ValueError(
                f"{source}: observed mask includes a non-finite value at "
                f"row={int(bad[0])}, feature={self.feature_names[int(bad[1])]!r}"
            )
        valid = mask & np.isfinite(numeric)
        self.count += valid.sum(axis=0, dtype=np.int64)
        safe = np.where(valid, numeric, 0.0)
        self.total += safe.sum(axis=0, dtype=np.float64)
        self.total_squares += np.square(safe).sum(axis=0, dtype=np.float64)

    def result(
        self,
        semantics: str,
        non_scaling_prefix: str = NON_SCALING_PREFIX,
    ) -> Dict[str, Any]:
        features: Dict[str, Any] = {}
        for index, feature_name in enumerate(self.feature_names):
            count = int(self.count[index])
            if feature_name.startswith(non_scaling_prefix):
                features[feature_name] = {
                    "count": count,
                    "center": 0.0,
                    "scale": 1.0,
                    "fallback": "not_scaled_deterministic_time_feature",
                    "scale_applied": False,
                    "semantics": (
                        "identity transform; deterministic local-time feature is not normalized"
                    ),
                }
                continue
            if count == 0:
                features[feature_name] = {
                    "count": 0,
                    "center": 0.0,
                    "scale": 1.0,
                    "fallback": "no_observed_train_values",
                    "scale_applied": True,
                    "semantics": semantics,
                }
                continue
            mean = float(self.total[index] / count)
            variance = float(self.total_squares[index] / count - mean * mean)
            if variance < 0.0 and variance > -1e-10 * max(mean * mean, 1.0):
                variance = 0.0
            if not np.isfinite(variance) or variance < 0.0:
                raise ValueError(
                    f"Invalid streaming variance for {feature_name}: {variance}"
                )
            std = float(np.sqrt(variance))
            fallback: Optional[str] = None
            scale = std
            if not np.isfinite(scale) or scale <= EPSILON:
                scale = 1.0
                fallback = "unit_scale_zero_or_nonfinite_std"
            features[feature_name] = {
                "count": count,
                "center": mean,
                "scale": float(scale),
                "fallback": fallback,
                "scale_applied": True,
                "semantics": semantics,
            }
        return {
            "method": "mean_population_std",
            "feature_order": list(self.feature_names),
            "features": features,
            "semantics": semantics,
        }


def _find_schema(
    explicit: Optional[str], candidates: Sequence[Path], source: str, required: bool
) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    if explicit:
        path = Path(explicit)
        return path, _read_json(path, f"{source} schema")
    existing = []
    for candidate in candidates:
        if candidate.exists() and candidate not in existing:
            existing.append(candidate)
    if len(existing) == 1:
        return existing[0], _read_json(existing[0], f"{source} schema")
    if len(existing) > 1:
        raise ValueError(
            f"Multiple possible {source} schemas found; pass --{source}-schema: "
            f"{[str(path) for path in existing]}"
        )
    if required:
        raise ValueError(
            f"Could not infer {source} schema; pass --{source}-schema explicitly"
        )
    return None, None


def _declared_hash(payload: Mapping[str, Any], source: str) -> Tuple[str, str]:
    primary_candidates: List[Tuple[str, Any]] = [
        ("config_hash", payload.get("config_hash")),
        ("artifact_config_hash", payload.get("artifact_config_hash")),
        (
            "artifact_config.config_hash",
            payload.get("artifact_config", {}).get("config_hash")
            if isinstance(payload.get("artifact_config"), Mapping)
            else None,
        ),
        (
            "preprocessing_config.config_hash",
            payload.get("preprocessing_config", {}).get("config_hash")
            if isinstance(payload.get("preprocessing_config"), Mapping)
            else None,
        ),
    ]
    present = [
        (kind, str(value).strip())
        for kind, value in primary_candidates
        if str(value or "").strip()
    ]
    # Clinical V2 originally exposed only task_config_hash.  Prefer a complete
    # artifact/preprocessing config hash when a later schema exposes both;
    # task_config_hash identifies the exclusion spec, not the entire builder.
    if not present and str(payload.get("task_config_hash") or "").strip():
        present = [("task_config_hash", str(payload["task_config_hash"]).strip())]
    if not present:
        raise ValueError(f"{source} does not declare a config hash")
    hashes = {value for _, value in present}
    if len(hashes) != 1:
        raise ValueError(f"{source} declares conflicting config hashes: {present}")
    return present[0][1], present[0][0]


def _check_expected_hash(actual: str, expected: Optional[str], source: str) -> None:
    if expected is not None and str(expected).strip() != actual:
        raise ValueError(
            f"{source} config hash mismatch: actual={actual!r}, expected={expected!r}"
        )


def fit_clinical(
    wide_path: Path,
    schema_path_arg: Optional[str],
    population: FitPopulation,
    expected_hash: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Path]]:
    stem = wide_path.name
    schema_candidates: List[Path] = []
    if stem.startswith("clinical_features_") and stem.endswith("_wide.csv"):
        task = stem[len("clinical_features_") : -len("_wide.csv")]
        schema_candidates.append(wide_path.parent / f"clinical_schema_{task}.json")
    # A task-specific filename is unambiguous even when several task schemas
    # share the directory.  Only fall back to discovery when it is absent.
    if not any(path.exists() for path in schema_candidates):
        generic = wide_path.parent / "clinical_schema.json"
        schema_candidates = (
            [generic]
            if generic.exists()
            else list(wide_path.parent.glob("clinical_schema_*.json"))
        )
    schema_path, schema = _find_schema(
        schema_path_arg, schema_candidates, "clinical", required=True
    )
    assert schema_path is not None and schema is not None
    config_hash, hash_kind = _declared_hash(schema, "clinical schema")
    _check_expected_hash(config_hash, expected_hash, "clinical")
    if str(schema.get("fit_split", "")).lower() != "train":
        raise ValueError("clinical schema must declare fit_split=train")
    schema_fit_hash = str(schema.get("fit_person_ids_hash", ""))
    if schema_fit_hash != population.fit_person_ids_hash:
        raise ValueError(
            "clinical schema fit_person_ids_hash does not match official train IDs"
        )

    frame = _validate_known_ids(
        pd.read_csv(
            wide_path,
            low_memory=False,
            dtype={"person_id": "string"},
            keep_default_na=False,
        ),
        population,
        "clinical wide",
    )
    if frame["person_id"].duplicated().any():
        raise ValueError("clinical wide must contain at most one row per person_id")
    missing_train_people = sorted(
        set(population.train_ids) - set(frame["person_id"])
    )
    if missing_train_people:
        raise ValueError(
            "clinical V2 wide must cover every official train participant; "
            f"missing: {missing_train_people[:10]}"
        )
    feature_names = [str(value) for value in schema.get("feature_names", [])]
    if not feature_names:
        raise ValueError("clinical schema has no feature_names")
    table_features = [column for column in frame.columns if column != "person_id"]
    if table_features != feature_names:
        raise ValueError(
            "clinical wide feature columns/order do not exactly match clinical schema"
        )
    train = frame[frame["person_id"].isin(population.train_ids)].copy()
    scaler = fit_robust_features(
        train,
        feature_names,
        semantics="subtract official-train median and divide by official-train IQR",
    )
    scaler.update(
        {
            "n_train_rows": int(len(train)),
            "n_train_participants_with_rows": int(train["person_id"].nunique()),
            "fit_participant_ids_hash": canonical_json_hash(
                {"person_ids": sorted(set(train["person_id"]))}
            ),
        }
    )
    source_info = {
        "config_hash": config_hash,
        "hash_kind": hash_kind,
        "schema_hash": canonical_json_hash(schema),
        "task": schema.get("task"),
        "task_config_hash": schema.get("task_config_hash"),
    }
    return scaler, source_info, [wide_path, schema_path]


def _ecg_schema_candidates(paths: Iterable[Optional[Path]]) -> List[Path]:
    result: List[Path] = []
    for path in paths:
        if path is not None:
            candidate = path.parent / "ecg_schema.json"
            if candidate not in result:
                result.append(candidate)
    return result


def _validate_ecg_table_binding(
    schema: Mapping[str, Any],
    schema_path: Path,
    output_key: str,
    actual_path: Path,
) -> None:
    """Bind a hash-declaring ECG schema to its declared CSV output."""
    output_files = schema.get("output_files")
    if not isinstance(output_files, Mapping):
        raise ValueError(
            "ECG schema lacks output_files, so CSV inputs cannot be bound to its config hash"
        )
    declared_value = str(output_files.get(output_key) or "").strip()
    if not declared_value:
        raise ValueError(f"ECG schema does not declare output_files.{output_key}")
    declared = Path(declared_value)
    actual_resolved = actual_path.resolve()
    declared_resolved = declared.resolve()
    bundled_resolved = (schema_path.parent / declared.name).resolve()
    if actual_resolved not in {declared_resolved, bundled_resolved}:
        raise ValueError(
            f"ECG {output_key} CSV is not the file declared by ecg_schema.json: "
            f"actual={actual_resolved}, declared={declared_resolved}"
        )


def _validate_ecg_csv_config_hash(
    frame: pd.DataFrame,
    expected_config_hash: str,
    source: str,
    allow_legacy_unbound: bool,
) -> str:
    if "artifact_config_hash" not in frame.columns:
        if allow_legacy_unbound:
            return "legacy_schema_output_path_binding"
        raise ValueError(
            f"{source} lacks artifact_config_hash. Regenerate it with the current ECG V2 "
            "builder, or explicitly pass --allow-legacy-unbound-ecg-csv to use the "
            "weaker schema output-path binding."
        )
    column = frame["artifact_config_hash"]
    if column.isna().any() or (column.astype(str).str.strip() == "").any():
        raise ValueError(f"{source} contains an empty artifact_config_hash")
    hashes = set(column.astype(str).str.strip())
    if hashes != {expected_config_hash}:
        raise ValueError(
            f"{source} artifact_config_hash mismatch: found={sorted(hashes)}, "
            f"schema={expected_config_hash!r}"
        )
    return "row_level_artifact_config_hash"


def fit_ecg_numeric(
    wide_path: Path,
    schema: Mapping[str, Any],
    population: FitPopulation,
    expected_config_hash: Optional[str] = None,
    allow_legacy_unbound: bool = False,
    schema_path: Optional[Path] = None,
) -> Dict[str, Any]:
    frame = _validate_known_ids(
        pd.read_csv(
            wide_path,
            low_memory=False,
            dtype={
                "person_id": "string",
                "record_id": "string",
                "artifact_key": "string",
                "artifact_config_hash": "string",
            },
            keep_default_na=False,
        ),
        population,
        "ECG numeric wide",
    )
    schema_config_hash, _ = _declared_hash(schema, "ECG schema")
    _check_expected_hash(schema_config_hash, expected_config_hash, "ECG")
    if "artifact_config_hash" not in frame.columns and allow_legacy_unbound:
        if schema_path is None:
            raise ValueError(
                "Legacy ECG numeric binding requires schema_path so output_files.numeric "
                "can be verified"
            )
        _validate_ecg_table_binding(schema, schema_path, "numeric", wide_path)
    binding = _validate_ecg_csv_config_hash(
        frame,
        schema_config_hash,
        "ECG numeric wide",
        allow_legacy_unbound,
    )
    identity_columns = {"record_id", "artifact_key"}
    missing_identity = sorted(identity_columns - set(frame.columns))
    if missing_identity:
        raise ValueError(
            f"ECG numeric wide missing identity columns: {missing_identity}"
        )
    for column in sorted(identity_columns):
        if frame[column].isna().any() or (
            frame[column].astype(str).str.strip() == ""
        ).any():
            raise ValueError(f"ECG numeric wide contains an empty {column}")
    if frame["artifact_key"].astype(str).duplicated().any():
        raise ValueError("ECG numeric wide has duplicate artifact_key rows")
    if frame.duplicated(["person_id", "record_id"]).any():
        raise ValueError("ECG numeric wide has duplicate person_id/record_id rows")
    numeric_names = [str(value) for value in schema.get("numeric_feature_names", [])]
    if not numeric_names:
        raise ValueError("ECG schema has no numeric_feature_names")
    feature_names = [f"ecg__{name}" for name in numeric_names]
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"ECG numeric wide missing schema features: {missing}")

    train_selector = frame["person_id"].isin(population.train_ids).to_numpy()
    train = frame.loc[train_selector].copy()
    masks: Dict[str, np.ndarray] = {}
    for feature_name in feature_names:
        mask_name = f"{feature_name}__mask"
        if mask_name not in frame.columns:
            raise ValueError(
                f"ECG numeric wide is missing required V2 observation mask: {mask_name}"
            )
        raw_mask = pd.to_numeric(frame[mask_name], errors="coerce").to_numpy()
        all_rows_mask = validated_boolean_mask(raw_mask, f"ECG numeric {mask_name}")
        numeric_values = pd.to_numeric(
            frame[feature_name], errors="coerce"
        ).to_numpy(dtype=np.float64)
        if np.any(all_rows_mask & ~np.isfinite(numeric_values)):
            raise ValueError(
                f"ECG numeric mask marks a non-finite value observed: {feature_name}"
            )
        masks[feature_name] = all_rows_mask[train_selector]
    scaler = fit_robust_features(
        train,
        feature_names,
        masks=masks,
        semantics=(
            "subtract official-train record median and divide by official-train record IQR; "
            "explicit ECG numeric masks take precedence over storage values"
        ),
    )
    scaler.update(
        {
            "n_train_rows": int(len(train)),
            "n_train_participants_with_rows": int(train["person_id"].nunique()),
            "fit_participant_ids_hash": canonical_json_hash(
                {"person_ids": sorted(set(train["person_id"]))}
            ),
            "config_binding": binding,
        }
    )
    return scaler


def _strict_bool_series(values: pd.Series, source: str) -> pd.Series:
    def parse(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"true", "1"}:
            return True
        if text in {"false", "0"}:
            return False
        raise ValueError(f"{source} contains a non-boolean value: {value!r}")

    return values.map(parse)


def fit_ecg_waveform_stats(
    stats_path: Path,
    schema: Mapping[str, Any],
    population: FitPopulation,
    expected_config_hash: Optional[str] = None,
    allow_legacy_unbound: bool = False,
    schema_path: Optional[Path] = None,
) -> Dict[str, Any]:
    stats = _validate_known_ids(
        pd.read_csv(
            stats_path,
            low_memory=False,
            dtype={
                "person_id": "string",
                "record_id": "string",
                "artifact_key": "string",
                "artifact_config_hash": "string",
            },
            keep_default_na=False,
        ),
        population,
        "ECG channel stats",
    )
    schema_config_hash, _ = _declared_hash(schema, "ECG schema")
    _check_expected_hash(schema_config_hash, expected_config_hash, "ECG")
    if "artifact_config_hash" not in stats.columns and allow_legacy_unbound:
        if schema_path is None:
            raise ValueError(
                "Legacy ECG channel-stats binding requires schema_path so "
                "output_files.channel_stats can be verified"
            )
        _validate_ecg_table_binding(
            schema, schema_path, "channel_stats", stats_path
        )
    binding = _validate_ecg_csv_config_hash(
        stats,
        schema_config_hash,
        "ECG channel stats",
        allow_legacy_unbound,
    )
    required = {
        "record_id",
        "artifact_key",
        "lead",
        "include_for_train_scaler",
        "valid_count",
        "sum_mv",
        "sum_squares_mv2",
    }
    missing = sorted(required - set(stats.columns))
    if missing:
        raise ValueError(f"ECG channel stats missing columns: {missing}")
    for column in ["record_id", "artifact_key", "lead"]:
        if stats[column].isna().any() or (
            stats[column].astype(str).str.strip() == ""
        ).any():
            raise ValueError(f"ECG channel stats contains an empty {column}")
    if stats.duplicated(["artifact_key", "lead"]).any():
        raise ValueError("ECG channel stats has duplicate artifact_key/lead rows")
    identity = stats[["person_id", "record_id", "artifact_key"]].drop_duplicates()
    if identity["artifact_key"].astype(str).duplicated().any():
        raise ValueError(
            "ECG channel stats maps one artifact_key to multiple record identities"
        )
    if identity.duplicated(["person_id", "record_id"]).any():
        raise ValueError(
            "ECG channel stats maps one person_id/record_id to multiple artifact keys"
        )

    canonical_leads = [str(value) for value in schema.get("canonical_leads", [])]
    if not canonical_leads:
        raise ValueError("ECG schema has no canonical_leads")
    unknown_leads = sorted(set(stats["lead"].astype(str)) - set(canonical_leads))
    if unknown_leads:
        raise ValueError(f"ECG channel stats contains unknown leads: {unknown_leads}")
    expected_leads = set(canonical_leads)
    incomplete_artifacts = [
        str(artifact_key)
        for artifact_key, group in stats.groupby("artifact_key", sort=False)
        if set(group["lead"].astype(str)) != expected_leads
    ]
    if incomplete_artifacts:
        raise ValueError(
            "ECG channel stats must contain every canonical lead exactly once per "
            f"artifact; incomplete artifacts: {incomplete_artifacts[:10]}"
        )
    stats["include_for_train_scaler"] = _strict_bool_series(
        stats["include_for_train_scaler"], "ECG include_for_train_scaler"
    )
    for column in ["valid_count", "sum_mv", "sum_squares_mv2"]:
        stats[column] = pd.to_numeric(stats[column], errors="coerce")
        if stats[column].isna().any() or not np.isfinite(stats[column]).all():
            raise ValueError(f"ECG channel stats has invalid {column}")
    if (stats["valid_count"] < 0).any():
        raise ValueError("ECG valid_count cannot be negative")
    if not np.equal(stats["valid_count"], np.floor(stats["valid_count"])).all():
        raise ValueError("ECG valid_count must contain integer counts")
    if (stats["sum_squares_mv2"] < 0).any():
        raise ValueError("ECG sum_squares_mv2 cannot be negative")
    counts = stats["valid_count"].to_numpy(dtype=np.float64)
    totals = stats["sum_mv"].to_numpy(dtype=np.float64)
    observed_squares = stats["sum_squares_mv2"].to_numpy(dtype=np.float64)
    zero_count = counts == 0
    if np.any(
        zero_count
        & ((np.abs(totals) > EPSILON) | (np.abs(observed_squares) > EPSILON))
    ):
        raise ValueError(
            "ECG sufficient statistics with valid_count=0 must have zero sums"
        )
    positive_count = counts > 0
    minimum_squares = np.zeros_like(observed_squares)
    minimum_squares[positive_count] = (
        np.square(totals[positive_count]) / counts[positive_count]
    )
    tolerance = 1e-10 * np.maximum(np.abs(minimum_squares), 1.0)
    impossible = positive_count & (observed_squares + tolerance < minimum_squares)
    if np.any(impossible):
        bad_index = int(np.flatnonzero(impossible)[0])
        raise ValueError(
            "ECG sufficient statistics violate sum_squares >= sum^2/count "
            f"at row offset {bad_index}"
        )
    if np.any(stats["include_for_train_scaler"].to_numpy() & zero_count):
        raise ValueError("Included ECG channel stats rows must have valid_count > 0")
    train = stats[
        stats["person_id"].isin(population.train_ids)
        & stats["include_for_train_scaler"]
    ].copy()

    features: Dict[str, Any] = {}
    for lead in canonical_leads:
        selected = train[train["lead"].astype(str) == lead]
        count = int(selected["valid_count"].sum()) if len(selected) else 0
        semantics = (
            "subtract sample-weighted official-train lead mean and divide by "
            "sample-weighted population std; only include_for_train_scaler rows contribute"
        )
        if count == 0:
            features[lead] = {
                "count": 0,
                "center": 0.0,
                "scale": 1.0,
                "fallback": "no_quality_pass_train_samples",
                "scale_applied": True,
                "semantics": semantics,
            }
            continue
        total = float(selected["sum_mv"].sum())
        total_squares = float(selected["sum_squares_mv2"].sum())
        mean = total / count
        variance = total_squares / count - mean * mean
        if variance < 0.0 and variance > -1e-10 * max(mean * mean, 1.0):
            variance = 0.0
        if not np.isfinite(variance) or variance < 0.0:
            raise ValueError(f"Invalid combined ECG variance for lead {lead}: {variance}")
        std = float(np.sqrt(variance))
        fallback: Optional[str] = None
        scale = std
        if scale <= EPSILON:
            scale = 1.0
            fallback = "unit_scale_zero_std"
        features[lead] = {
            "count": count,
            "center": float(mean),
            "scale": float(scale),
            "fallback": fallback,
            "scale_applied": True,
            "n_included_channel_stat_rows": int(len(selected)),
            "semantics": semantics,
        }
    return {
        "method": "sample_weighted_mean_population_std_from_sufficient_statistics",
        "units": "mV",
        "feature_order": canonical_leads,
        "features": features,
        "n_train_channel_stat_rows_included": int(len(train)),
        "n_train_participants_included": int(train["person_id"].nunique()),
        "fit_participant_ids_hash": canonical_json_hash(
            {"person_ids": sorted(set(train["person_id"]))}
        ),
        "config_binding": binding,
        "semantics": "quality-pass ECG waveform samples from official train participants only",
    }


def _import_torch(source: str) -> Any:
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            f"{source} artifact scaling requires PyTorch to read .pt files. "
            "Install/use the OT2 training environment with torch, or omit this source."
        ) from exc
    return torch


def _torch_load(torch: Any, path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch before weights_only was added
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact must contain a dictionary: {path}")
    return payload


def _as_numpy(value: Any, source: str) -> np.ndarray:
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)
    except Exception as exc:
        raise ValueError(f"Could not convert tensor to numpy for {source}") from exc


def _resolve_artifact_path(
    raw_value: Any,
    index_path: Path,
    artifact_dir: Optional[Path],
    person_id: str,
) -> Path:
    if raw_value is None:
        text = ""
    else:
        try:
            text = "" if pd.isna(raw_value) else str(raw_value).strip()
        except (TypeError, ValueError):
            text = str(raw_value).strip()
    if not text or text.lower() == "nan":
        raise ValueError(f"Index row for person_id={person_id} has no artifact path")
    raw = Path(text)
    candidates: List[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([raw, index_path.parent / raw])
    if artifact_dir is not None:
        candidates.extend([artifact_dir / raw.name, artifact_dir / f"{person_id}.pt"])
    candidates.append(index_path.parent / raw.name)
    unique: List[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    for candidate in unique:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Artifact for person_id={person_id} was not found; tried "
        f"{[str(candidate) for candidate in unique]}"
    )


def _index_config_hash(index: pd.DataFrame, source: str) -> str:
    if "config_hash" not in index.columns:
        raise ValueError(f"{source} index is missing config_hash")
    if index["config_hash"].isna().any() or (
        index["config_hash"].astype(str).str.strip() == ""
    ).any():
        raise ValueError(f"{source} index contains an empty config_hash")
    hashes = set(index["config_hash"].astype(str).str.strip())
    if len(hashes) != 1:
        raise ValueError(f"{source} index contains multiple config hashes: {sorted(hashes)}")
    return next(iter(hashes))


def _artifact_config_hash(payload: Mapping[str, Any], source: str) -> str:
    values = []
    for candidate in [
        payload.get("config_hash"),
        payload.get("artifact_config_hash"),
        payload.get("preprocessing_config", {}).get("config_hash")
        if isinstance(payload.get("preprocessing_config"), Mapping)
        else None,
        payload.get("artifact_config", {}).get("config_hash")
        if isinstance(payload.get("artifact_config"), Mapping)
        else None,
    ]:
        text = str(candidate or "").strip()
        if text:
            values.append(text)
    if not values:
        raise ValueError(f"{source} artifact does not declare a config hash")
    if len(set(values)) != 1:
        raise ValueError(f"{source} artifact declares conflicting config hashes: {values}")
    return values[0]


def environment_feature_mask(
    feature_names: Sequence[str],
    sensor_columns: Sequence[str],
    channel_mask: np.ndarray,
    row_count: np.ndarray,
) -> np.ndarray:
    """Construct exact feature support for V2 environment artifacts."""
    names = [str(name) for name in feature_names]
    channel_mask = validated_boolean_mask(
        channel_mask, "environment_channel_observed_mask"
    )
    row_count = np.asarray(row_count)
    if channel_mask.ndim != 2 or row_count.ndim != 1:
        raise ValueError("Environment channel mask must be 2D and row_count must be 1D")
    if row_count.dtype.kind not in {"i", "u", "f"} or row_count.dtype.kind == "b":
        raise ValueError("Environment row_count must be numeric nonnegative integers")
    if not np.isfinite(row_count).all() or np.any(row_count < 0):
        raise ValueError("Environment row_count contains non-finite or negative values")
    if not np.equal(row_count, np.floor(row_count)).all():
        raise ValueError("Environment row_count contains non-integer values")
    if channel_mask.shape[0] != row_count.shape[0]:
        raise ValueError("Environment channel mask and row_count length mismatch")
    if channel_mask.shape[1] != len(sensor_columns):
        raise ValueError("Environment channel mask width does not match sensor_columns")
    rows_present = row_count > 0
    if np.any(channel_mask.any(axis=1) & ~rows_present):
        raise ValueError(
            "Environment channel mask marks an observed sensor where row_count is zero"
        )
    if len(names) != len(set(names)):
        raise ValueError("Environment feature_names contains duplicates")
    index = {name: i for i, name in enumerate(names)}
    mask = np.zeros((channel_mask.shape[0], len(names)), dtype=bool)
    supported_names: set[str] = set()
    for channel_index, column in enumerate(sensor_columns):
        safe = str(column).replace(".", "_")
        mean_name = f"env__{safe}__mean"
        std_name = f"env__{safe}__std"
        missing_name = f"env__{safe}__missing_rate"
        for name in [mean_name, std_name, missing_name]:
            if name not in index:
                raise ValueError(f"Environment artifact missing expected feature {name}")
            supported_names.add(name)
        mask[:, index[mean_name]] = channel_mask[:, channel_index]
        mask[:, index[std_name]] = channel_mask[:, channel_index]
        mask[:, index[missing_name]] = rows_present
    if "env__row_count" not in index:
        raise ValueError("Environment artifact missing env__row_count")
    supported_names.add("env__row_count")
    mask[:, index["env__row_count"]] = rows_present
    for name in names:
        if name.startswith(NON_SCALING_PREFIX):
            mask[:, index[name]] = True
            supported_names.add(name)
    unsupported = sorted(set(names) - supported_names)
    if unsupported:
        raise ValueError(f"No feature-mask semantics defined for environment features: {unsupported}")
    return mask


def _schema_string_list(
    payload: Mapping[str, Any], key: str, source: str
) -> List[str]:
    raw = payload.get(key)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{source} must declare a non-empty {key} list")
    values = [str(value).strip() for value in raw]
    if any(not value for value in values):
        raise ValueError(f"{source} {key} contains an empty name")
    if len(values) != len(set(values)):
        raise ValueError(f"{source} {key} contains duplicates")
    return values


def _schema_shape(value: Any, rank: int, source: str) -> Tuple[int, ...]:
    if not isinstance(value, list) or len(value) != rank:
        raise ValueError(f"{source} must be a rank-{rank} shape list")
    shape: List[int] = []
    for dimension in value:
        if isinstance(dimension, (bool, np.bool_)):
            raise ValueError(f"{source} shape dimensions must be positive integers")
        try:
            numeric = float(dimension)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{source} shape dimensions must be positive integers"
            ) from exc
        if (
            not np.isfinite(numeric)
            or numeric <= 0
            or numeric != np.floor(numeric)
        ):
            raise ValueError(f"{source} shape dimensions must be positive integers")
        shape.append(int(numeric))
    return tuple(shape)


def _validate_environment_schema_structure(
    schema: Mapping[str, Any],
) -> Tuple[List[str], List[str], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    feature_names = _schema_string_list(
        schema, "feature_names", "environment schema"
    )
    sensor_columns = _schema_string_list(
        schema, "sensor_columns", "environment schema"
    )
    tensor_shape = _schema_shape(
        schema.get("shape"), 2, "environment schema tensor"
    )
    channel_shape = _schema_shape(
        schema.get("channel_mask_shape"), 2, "environment schema channel mask"
    )
    row_shape = _schema_shape(
        schema.get("row_count_shape"), 1, "environment schema row count"
    )
    if tensor_shape[1] != len(feature_names):
        raise ValueError(
            "Environment schema tensor width does not match feature_names"
        )
    if channel_shape[1] != len(sensor_columns):
        raise ValueError(
            "Environment schema channel-mask width does not match sensor_columns"
        )
    if tensor_shape[0] != channel_shape[0] or tensor_shape[0] != row_shape[0]:
        raise ValueError(
            "Environment schema sequence lengths disagree across tensor/masks"
        )
    time_names = _schema_string_list(
        schema, "time_feature_names", "environment schema"
    )
    expected_time_names = [
        name for name in feature_names if name.startswith(NON_SCALING_PREFIX)
    ]
    if time_names != expected_time_names:
        raise ValueError(
            "Environment schema time_feature_names do not match feature_names order"
        )
    # Validate the complete sensor-to-feature naming contract without allocating
    # a full sequence-sized mask.
    environment_feature_mask(
        feature_names,
        sensor_columns,
        np.zeros((0, len(sensor_columns)), dtype=bool),
        np.zeros((0,), dtype=np.int64),
    )
    return feature_names, sensor_columns, tensor_shape, channel_shape, row_shape


def _validate_wearable_schema_structure(
    schema: Mapping[str, Any],
) -> Tuple[Dict[str, List[str]], Dict[str, Tuple[int, ...]]]:
    feature_names: Dict[str, List[str]] = {}
    shapes: Dict[str, Tuple[int, ...]] = {}
    for group in ["cgm", "activity"]:
        group_schema = schema.get(group)
        if not isinstance(group_schema, Mapping):
            raise ValueError(f"Wearable schema must declare a {group} object")
        feature_names[group] = _schema_string_list(
            group_schema, "feature_names", f"wearable schema {group}"
        )
        shapes[group] = _schema_shape(
            group_schema.get("shape"), 2, f"wearable schema {group} tensor"
        )
        if shapes[group][1] != len(feature_names[group]):
            raise ValueError(
                f"Wearable schema {group} tensor width does not match feature_names"
            )
        mask_keys = group_schema.get("mask_keys")
        expected_mask = f"{group}_feature_mask"
        if not isinstance(mask_keys, list) or expected_mask not in {
            str(value) for value in mask_keys
        }:
            raise ValueError(
                f"Wearable schema {group} must declare {expected_mask} in mask_keys"
            )
    if shapes["cgm"][0] != shapes["activity"][0]:
        raise ValueError("Wearable schema CGM/activity sequence lengths disagree")
    time_grid = schema.get("time_grid")
    if isinstance(time_grid, Mapping) and time_grid.get("seq_len") is not None:
        grid_shape = _schema_shape(
            [time_grid.get("seq_len")], 1, "wearable schema time_grid"
        )
        if grid_shape[0] != shapes["cgm"][0]:
            raise ValueError(
                "Wearable schema time_grid.seq_len disagrees with tensor shapes"
            )
    return feature_names, shapes


def _validate_index(
    index_path: Path, population: FitPopulation, source: str
) -> pd.DataFrame:
    index = _validate_known_ids(
        pd.read_csv(
            index_path,
            low_memory=False,
            dtype={"person_id": "string"},
            keep_default_na=False,
        ),
        population,
        f"{source} index",
    )
    if index.empty:
        raise ValueError(f"{source} index is empty")
    if index["person_id"].duplicated().any():
        raise ValueError(f"{source} index must have at most one row per person_id")
    return index


def _is_failed_status(value: Any) -> bool:
    return str(value).strip().lower() in {"failed", "failure", "error"}


def fit_environment_streaming(
    index_path: Path,
    artifact_dir: Optional[Path],
    schema_path_arg: Optional[str],
    population: FitPopulation,
    expected_hash: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Path]]:
    index = _validate_index(index_path, population, "environment")
    config_hash = _index_config_hash(index, "environment")
    _check_expected_hash(config_hash, expected_hash, "environment")
    schema_path, schema = _find_schema(
        schema_path_arg,
        [index_path.parent / "environment_5min_schema.json"],
        "environment",
        required=True,
    )
    assert schema_path is not None and schema is not None
    schema_hash, _ = _declared_hash(schema, "environment schema")
    if schema_hash != config_hash:
        raise ValueError("Environment schema/index config hash mismatch")
    (
        schema_feature_names,
        schema_sensor_columns,
        schema_tensor_shape,
        schema_channel_shape,
        schema_row_shape,
    ) = _validate_environment_schema_structure(schema)

    torch = _import_torch("Environment")
    train_rows = index[index["person_id"].isin(population.train_ids)].copy()
    if "status" in train_rows.columns:
        train_rows = train_rows[~train_rows["status"].map(_is_failed_status)]
    path_column = "artifact_path" if "artifact_path" in index.columns else "out_path"
    if path_column not in index.columns:
        raise ValueError("Environment index has neither artifact_path nor out_path")

    accumulator: Optional[StreamingMoments] = None
    feature_names: Optional[List[str]] = None
    observed_tensor_shape: Optional[Tuple[int, ...]] = None
    n_loaded = 0
    loaded_person_ids: List[str] = []
    artifact_paths: List[Path] = []
    for row in train_rows.to_dict("records"):
        person_id = str(row["person_id"])
        artifact_path = _resolve_artifact_path(
            row.get(path_column), index_path, artifact_dir, person_id
        )
        artifact = _torch_load(torch, artifact_path)
        if as_person_id(artifact.get("person_id")) != person_id:
            raise ValueError(f"Environment artifact/index person_id mismatch: {artifact_path}")
        if _artifact_config_hash(artifact, "Environment") != config_hash:
            raise ValueError(f"Environment artifact config hash mismatch: {artifact_path}")
        current_names = [str(name) for name in artifact.get("feature_names", [])]
        if not current_names:
            raise ValueError(f"Environment artifact has no feature_names: {artifact_path}")
        if current_names != schema_feature_names:
            raise ValueError(
                f"Environment artifact/schema feature_names mismatch: {artifact_path}"
            )
        if feature_names is None:
            feature_names = current_names
            accumulator = StreamingMoments(feature_names)
        elif current_names != feature_names:
            raise ValueError("Environment feature_names differ across train artifacts")
        values = _as_numpy(artifact.get("environment"), str(artifact_path))
        current_shape = tuple(int(value) for value in values.shape)
        if observed_tensor_shape is None:
            observed_tensor_shape = current_shape
        elif current_shape != observed_tensor_shape:
            raise ValueError("Environment tensor shape differs across train artifacts")
        if current_shape != schema_tensor_shape:
            raise ValueError(
                f"Environment artifact/schema tensor shape mismatch: {artifact_path}"
            )
        artifact_sensors = [str(name) for name in artifact.get("sensor_columns", [])]
        if artifact_sensors != schema_sensor_columns:
            raise ValueError(
                f"Environment artifact/schema sensor_columns mismatch: {artifact_path}"
            )
        channel_mask = _as_numpy(
            artifact.get("environment_channel_observed_mask"), str(artifact_path)
        )
        row_count = _as_numpy(
            artifact.get("environment_row_count"), str(artifact_path)
        )
        if tuple(channel_mask.shape) != schema_channel_shape:
            raise ValueError(
                f"Environment channel-mask/schema shape mismatch: {artifact_path}"
            )
        if tuple(row_count.shape) != schema_row_shape:
            raise ValueError(
                f"Environment row-count/schema shape mismatch: {artifact_path}"
            )
        derived_mask = environment_feature_mask(
            current_names,
            artifact_sensors,
            channel_mask,
            row_count,
        )
        if "environment_feature_mask" in artifact:
            mask = validated_boolean_mask(
                _as_numpy(artifact["environment_feature_mask"], str(artifact_path)),
                "environment_feature_mask",
            )
            if mask.shape != derived_mask.shape or not np.array_equal(mask, derived_mask):
                raise ValueError(
                    f"Environment explicit/derived feature masks disagree: {artifact_path}"
                )
        else:
            mask = derived_mask
        assert accumulator is not None
        accumulator.update(values, mask, f"environment person_id={person_id}")
        n_loaded += 1
        loaded_person_ids.append(person_id)
        artifact_paths.append(artifact_path)
        del artifact, values, mask

    if accumulator is None:
        accumulator = StreamingMoments(schema_feature_names)
    result = accumulator.result(
        "subtract official-train observed-slot feature mean and divide by population std; "
        "storage fills are excluded by environment feature support"
    )
    result.update(
        {
            "n_train_artifacts_loaded": n_loaded,
            "n_official_train_ids_without_loaded_artifact": int(
                len(population.train_ids) - n_loaded
            ),
            "fit_participant_ids_hash": canonical_json_hash(
                {"person_ids": sorted(loaded_person_ids)}
            ),
        }
    )
    source_info = {
        "config_hash": config_hash,
        "hash_kind": "index.config_hash",
        "schema_hash": canonical_json_hash(schema),
    }
    inputs = [index_path, *artifact_paths]
    inputs.append(schema_path)
    return result, source_info, inputs


def fit_wearable_streaming(
    index_path: Path,
    artifact_dir: Optional[Path],
    schema_path_arg: Optional[str],
    population: FitPopulation,
    expected_hash: Optional[str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], List[Path]]:
    index = _validate_index(index_path, population, "wearable")
    config_hash = _index_config_hash(index, "wearable")
    _check_expected_hash(config_hash, expected_hash, "wearable")
    schema_path, schema = _find_schema(
        schema_path_arg,
        [index_path.parent / "wearable_5min_schema.json"],
        "wearable",
        required=True,
    )
    assert schema_path is not None and schema is not None
    schema_hash, _ = _declared_hash(schema, "wearable schema")
    if schema_hash != config_hash:
        raise ValueError("Wearable schema/index config hash mismatch")
    schema_feature_names, schema_shapes = _validate_wearable_schema_structure(schema)

    torch = _import_torch("Wearable/CGM")
    train_rows = index[index["person_id"].isin(population.train_ids)].copy()
    if "status" in train_rows.columns:
        train_rows = train_rows[~train_rows["status"].map(_is_failed_status)]
    path_column = "out_path" if "out_path" in index.columns else "artifact_path"
    if path_column not in index.columns:
        raise ValueError("Wearable index has neither out_path nor artifact_path")

    accumulators: Dict[str, StreamingMoments] = {}
    feature_names_by_group: Dict[str, List[str]] = {}
    observed_shapes: Dict[str, Tuple[int, ...]] = {}
    n_loaded = 0
    loaded_person_ids: List[str] = []
    artifact_paths: List[Path] = []
    for row in train_rows.to_dict("records"):
        person_id = str(row["person_id"])
        artifact_path = _resolve_artifact_path(
            row.get(path_column), index_path, artifact_dir, person_id
        )
        artifact = _torch_load(torch, artifact_path)
        if as_person_id(artifact.get("person_id")) != person_id:
            raise ValueError(f"Wearable artifact/index person_id mismatch: {artifact_path}")
        if _artifact_config_hash(artifact, "Wearable") != config_hash:
            raise ValueError(f"Wearable artifact config hash mismatch: {artifact_path}")
        for group in ["cgm", "activity"]:
            names_key = f"{group}_feature_names"
            mask_key = f"{group}_feature_mask"
            names = [str(name) for name in artifact.get(names_key, [])]
            if not names:
                raise ValueError(f"Wearable artifact missing {names_key}: {artifact_path}")
            if names != schema_feature_names[group]:
                raise ValueError(
                    f"Wearable {group} artifact/schema feature_names mismatch: {artifact_path}"
                )
            if group not in accumulators:
                feature_names_by_group[group] = names
                accumulators[group] = StreamingMoments(names)
            elif names != feature_names_by_group[group]:
                raise ValueError(f"{group} feature names differ across train artifacts")
            values = _as_numpy(artifact.get(group), str(artifact_path))
            mask = _as_numpy(artifact.get(mask_key), str(artifact_path))
            current_shape = tuple(int(value) for value in values.shape)
            if group not in observed_shapes:
                observed_shapes[group] = current_shape
            elif observed_shapes[group] != current_shape:
                raise ValueError(f"{group} tensor shape differs across train artifacts")
            if current_shape != schema_shapes[group]:
                raise ValueError(
                    f"Wearable {group} artifact/schema tensor shape mismatch: {artifact_path}"
                )
            accumulators[group].update(values, mask, f"{group} person_id={person_id}")
        n_loaded += 1
        loaded_person_ids.append(person_id)
        artifact_paths.append(artifact_path)
        del artifact

    if not accumulators:
        for group in ["cgm", "activity"]:
            accumulators[group] = StreamingMoments(schema_feature_names[group])

    result: Dict[str, Dict[str, Any]] = {}
    for group, accumulator in accumulators.items():
        group_result = accumulator.result(
            f"subtract official-train observed {group} feature mean and divide by population std; "
            f"storage fills are excluded by {group}_feature_mask"
        )
        group_result.update(
            {
                "n_train_artifacts_loaded": n_loaded,
                "n_official_train_ids_without_loaded_artifact": int(
                    len(population.train_ids) - n_loaded
                ),
                "fit_participant_ids_hash": canonical_json_hash(
                    {"person_ids": sorted(loaded_person_ids)}
                ),
            }
        )
        result[group] = group_result
    source_info = {
        "config_hash": config_hash,
        "hash_kind": "index.config_hash",
        "schema_hash": canonical_json_hash(schema),
    }
    inputs = [index_path, *artifact_paths]
    inputs.append(schema_path)
    return result, source_info, inputs


def _input_metadata(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    unique: List[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    fingerprints: List[Dict[str, Any]] = []
    for path in unique:
        if not path.is_file():
            raise FileNotFoundError(f"Scaler input file does not exist: {path}")
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise RuntimeError(f"Scaler input changed while fingerprinting: {path}")
        fingerprints.append(
            {
                "path": str(path),
                "exists": True,
                "size_bytes": int(after.st_size),
                "mtime_ns": int(after.st_mtime_ns),
                "sha256": digest.hexdigest(),
            }
        )
    return fingerprints


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", required=True)
    parser.add_argument(
        "--output", "--out-json", "--out_json", dest="output", required=True
    )

    parser.add_argument("--clinical-wide", "--clinical_wide", dest="clinical_wide")
    parser.add_argument("--clinical-schema", "--clinical_schema", dest="clinical_schema")
    parser.add_argument(
        "--expected-clinical-config-hash",
        "--expected_clinical_config_hash",
        dest="expected_clinical_config_hash",
    )

    parser.add_argument("--ecg-numeric-wide", "--ecg_numeric_wide", dest="ecg_numeric_wide")
    parser.add_argument(
        "--ecg-channel-stats", "--ecg_channel_stats", dest="ecg_channel_stats"
    )
    parser.add_argument("--ecg-schema", "--ecg_schema", dest="ecg_schema")
    parser.add_argument(
        "--expected-ecg-config-hash",
        "--expected_ecg_config_hash",
        dest="expected_ecg_config_hash",
    )
    parser.add_argument(
        "--allow-legacy-unbound-ecg-csv",
        "--allow_legacy_unbound_ecg_csv",
        dest="allow_legacy_unbound_ecg_csv",
        action="store_true",
        help=(
            "Allow older ECG V2 CSVs without artifact_config_hash, using weaker "
            "ecg_schema.json output-path binding. Default is a hard failure."
        ),
    )

    parser.add_argument(
        "--environment-index", "--environment_index", dest="environment_index"
    )
    parser.add_argument(
        "--environment-artifact-dir",
        "--environment_artifact_dir",
        dest="environment_artifact_dir",
    )
    parser.add_argument(
        "--environment-schema", "--environment_schema", dest="environment_schema"
    )
    parser.add_argument(
        "--expected-environment-config-hash",
        "--expected_environment_config_hash",
        dest="expected_environment_config_hash",
    )

    parser.add_argument("--wearable-index", "--wearable_index", dest="wearable_index")
    parser.add_argument(
        "--wearable-artifact-dir",
        "--wearable_artifact_dir",
        dest="wearable_artifact_dir",
    )
    parser.add_argument("--wearable-schema", "--wearable_schema", dest="wearable_schema")
    parser.add_argument(
        "--expected-wearable-config-hash",
        "--expected_wearable_config_hash",
        dest="expected_wearable_config_hash",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    source_args = [
        args.clinical_wide,
        args.ecg_numeric_wide,
        args.ecg_channel_stats,
        args.environment_index,
        args.wearable_index,
    ]
    if not any(source_args):
        raise ValueError("At least one scaler source input must be provided")

    participants_path = Path(args.participants)
    population = load_fit_population(participants_path)
    scalers: Dict[str, Any] = {}
    source_config_hashes: Dict[str, Any] = {}
    input_paths: List[Path] = [participants_path]

    if args.clinical_wide:
        clinical, source_info, paths = fit_clinical(
            Path(args.clinical_wide),
            args.clinical_schema,
            population,
            args.expected_clinical_config_hash,
        )
        scalers["clinical"] = clinical
        source_config_hashes["clinical"] = source_info
        input_paths.extend(paths)

    ecg_schema_path: Optional[Path] = None
    ecg_schema: Optional[Dict[str, Any]] = None
    if args.ecg_numeric_wide or args.ecg_channel_stats:
        numeric_path = Path(args.ecg_numeric_wide) if args.ecg_numeric_wide else None
        channel_path = Path(args.ecg_channel_stats) if args.ecg_channel_stats else None
        ecg_schema_path, ecg_schema = _find_schema(
            args.ecg_schema,
            _ecg_schema_candidates([numeric_path, channel_path]),
            "ecg",
            required=True,
        )
        assert ecg_schema_path is not None and ecg_schema is not None
        ecg_hash, ecg_hash_kind = _declared_hash(ecg_schema, "ECG schema")
        _check_expected_hash(ecg_hash, args.expected_ecg_config_hash, "ECG")
        source_config_hashes["ecg"] = {
            "config_hash": ecg_hash,
            "hash_kind": ecg_hash_kind,
            "schema_hash": canonical_json_hash(ecg_schema),
            "allow_legacy_unbound_ecg_csv": bool(
                args.allow_legacy_unbound_ecg_csv
            ),
            "csv_binding_policy": (
                "row-level artifact_config_hash must equal the schema config hash; "
                "legacy CSVs require explicit opt-in and schema output-path binding"
            ),
            "legacy_binding_limitation": (
                "schema output-path binding is weaker than row-level config binding "
                "and does not independently authenticate CSV row contents"
                if args.allow_legacy_unbound_ecg_csv
                else None
            ),
        }
        input_paths.append(ecg_schema_path)
        if numeric_path is not None:
            numeric_columns = pd.read_csv(numeric_path, nrows=0).columns
            if "artifact_config_hash" not in numeric_columns:
                if not args.allow_legacy_unbound_ecg_csv:
                    raise ValueError(
                        "ECG numeric CSV lacks artifact_config_hash; regenerate it or pass "
                        "--allow-legacy-unbound-ecg-csv explicitly"
                    )
                _validate_ecg_table_binding(
                    ecg_schema, ecg_schema_path, "numeric", numeric_path
                )
            scalers["ecg_numeric"] = fit_ecg_numeric(
                numeric_path,
                ecg_schema,
                population,
                expected_config_hash=ecg_hash,
                allow_legacy_unbound=args.allow_legacy_unbound_ecg_csv,
                schema_path=ecg_schema_path,
            )
            input_paths.append(numeric_path)
        if channel_path is not None:
            channel_columns = pd.read_csv(channel_path, nrows=0).columns
            if "artifact_config_hash" not in channel_columns:
                if not args.allow_legacy_unbound_ecg_csv:
                    raise ValueError(
                        "ECG channel-stats CSV lacks artifact_config_hash; regenerate it or "
                        "pass --allow-legacy-unbound-ecg-csv explicitly"
                    )
                _validate_ecg_table_binding(
                    ecg_schema, ecg_schema_path, "channel_stats", channel_path
                )
            scalers["ecg_waveform"] = fit_ecg_waveform_stats(
                channel_path,
                ecg_schema,
                population,
                expected_config_hash=ecg_hash,
                allow_legacy_unbound=args.allow_legacy_unbound_ecg_csv,
                schema_path=ecg_schema_path,
            )
            input_paths.append(channel_path)

    if args.environment_index:
        environment, source_info, paths = fit_environment_streaming(
            Path(args.environment_index),
            Path(args.environment_artifact_dir)
            if args.environment_artifact_dir
            else None,
            args.environment_schema,
            population,
            args.expected_environment_config_hash,
        )
        scalers["environment"] = environment
        source_config_hashes["environment"] = source_info
        input_paths.extend(paths)

    if args.wearable_index:
        wearable, source_info, paths = fit_wearable_streaming(
            Path(args.wearable_index),
            Path(args.wearable_artifact_dir) if args.wearable_artifact_dir else None,
            args.wearable_schema,
            population,
            args.expected_wearable_config_hash,
        )
        scalers.update(wearable)
        source_config_hashes["wearable"] = source_info
        input_paths.extend(paths)

    input_fingerprints = _input_metadata(input_paths)
    output = {
        "scaler_format_version": SCALER_FORMAT_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "fit_split": "recommended_split=train",
        "fit_population": {
            "n_participants_tsv": population.n_participants,
            "n_official_train_participants": len(population.train_ids),
            "fit_person_ids_hash": population.fit_person_ids_hash,
            "hash_semantics": (
                "sha256 canonical JSON of sorted official train person_id values"
            ),
        },
        "source_config_hashes": source_config_hashes,
        "scalers": scalers,
        "input_files": input_fingerprints,
        "global_semantics": {
            "fit_scope": "official recommended_split=train participants only",
            "canonical_artifacts": "remain raw scale and are never modified",
            "storage_fill": "excluded wherever the canonical feature mask is false",
            "deterministic_time_features": (
                "all feature names beginning time__ use identity scaling"
            ),
            "application": "subtract center then divide by scale only when scale_applied=true",
        },
        "scaler_config_hash_semantics": (
            "sha256 canonical JSON of this complete payload excluding scaler_config_hash; "
            "therefore fitted statistics and input path/size/mtime/SHA-256 fingerprints "
            "are covered"
        ),
    }
    output = _json_safe(output)
    if not isinstance(output, dict):  # Defensive; the root above is always a mapping.
        raise TypeError("Internal scaler payload must be a JSON object")
    output["scaler_config_hash"] = canonical_json_hash(output)
    _atomic_write_json(Path(args.output), output)
    print(
        json.dumps(
            {
                "output": args.output,
                "n_official_train_participants": len(population.train_ids),
                "fit_person_ids_hash": population.fit_person_ids_hash,
                "sources": sorted(scalers),
                "scaler_config_hash": output["scaler_config_hash"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
