#!/usr/bin/env python3
"""Strict, mask-first data layer for OT2 Model V2.

The canonical preprocessing artifacts intentionally retain raw values.  This
module is the model-side boundary which applies the frozen, train-only scaler
and exposes the nine scientific modality groups independently.  Missingness is
always carried by an explicit mask and by ``source_present``; an all-zero value
tensor is never interpreted as evidence that a source is missing.

The loader is deliberately lazy for per-participant artifacts and deserializes
them with PyTorch's restricted weights-only loader; unsafe pickle fallback is
never allowed.  The clinical
wide table is the one exception: it is loaded once and aligned to the master
cohort during construction.  Retinal artifacts are reduced immediately to
their model views and masks.  In particular, the optional FLIO lossless decay
cube archive is never returned, decoded, copied into a sample, or cached by this
module.  For repeated training, use the archive-free index produced by
``build_retinal_model_view_sidecars.py``; directly loading the canonical
monolithic FLIO artifacts still materializes their compressed archive bytes.

Import the native artifact loader through ``infosign.native_data``.
"""
from __future__ import annotations

import json
import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # Let schema-only tooling import the module.
    torch = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        pass

PREPROCESSING_VERSION="aireadi-ot2-v2"
SCALER_FORMAT_VERSION="aireadi_ot2_train_scalers_v2"
NON_SCALING_PREFIX="time__"
def as_person_id(value: Any) -> str:
    """Normalize identifiers without converting valid leading-zero strings."""
    if isinstance(value, str):
        value = value.strip()
        return value[:-2] if value.endswith(".0") and value[:-2].isdigit() else value
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()

def canonical_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

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


PRIMARY_SOURCE_ORDER: Tuple[str, ...] = (
    "clinical_data",
    "cardiac_ecg",
    "environment",
    "retinal_flio",
    "retinal_oct",
    "retinal_octa",
    "retinal_photography",
    "wearable_activity_monitor",
    "wearable_blood_glucose",
)

RETINAL_GROUPS: Tuple[str, ...] = (
    "retinal_flio",
    "retinal_oct",
    "retinal_octa",
    "retinal_photography",
)

RETINAL_MODALITY_TO_GROUP: Mapping[str, str] = {
    "flio": "retinal_flio",
    "oct": "retinal_oct",
    "octa": "retinal_octa",
    "photography": "retinal_photography",
}

SOURCE_FLAG_BY_GROUP: Mapping[str, str] = {
    "clinical_data": "has_clinical",
    "cardiac_ecg": "has_ecg",
    "environment": "has_environment",
    "retinal_flio": "has_retinal_flio",
    "retinal_oct": "has_retinal_oct",
    "retinal_octa": "has_retinal_octa",
    "retinal_photography": "has_retinal_photography",
    "wearable_activity_monitor": "has_activity",
    "wearable_blood_glucose": "has_cgm",
}

REQUIRED_SCALERS = frozenset(
    {"clinical", "ecg_numeric", "ecg_waveform", "environment", "cgm", "activity"}
)

DEFAULT_RETINAL_ASSET_CAPS: Mapping[str, int] = {
    "retinal_flio": 4,
    "retinal_oct": 4,
    "retinal_octa": 12,
    "retinal_photography": 8,
}


def _require_torch() -> Any:
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required for OT2V2Dataset; install the model environment first"
        )
    return torch


def _read_json(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Could not parse {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _strict_bool(value: Any, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and value in {0.0, 1.0}:
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"{label} must be a strict boolean, got {value!r}")


def _as_bool_tensor(value: Any, label: str) -> Any:
    module = _require_torch()
    try:
        array = validated_boolean_mask(np.asarray(value), label)
    except ValueError as exc:
        raise ValueError(f"Invalid {label}") from exc
    return module.as_tensor(np.ascontiguousarray(array), dtype=module.bool)


def _as_float_tensor(value: Any, label: str, rank: Optional[int] = None) -> Any:
    module = _require_torch()
    if value is None:
        raise ValueError(f"Missing tensor {label}")
    if module.is_tensor(value):
        tensor = value.detach().cpu().to(dtype=module.float32)
    else:
        try:
            tensor = module.as_tensor(np.asarray(value), dtype=module.float32)
        except Exception as exc:
            raise ValueError(f"Could not convert {label} to float32") from exc
    if rank is not None and tensor.ndim != rank:
        raise ValueError(f"{label} must be rank {rank}, got shape {tuple(tensor.shape)}")
    return tensor.contiguous()


def _safe_torch_load(path: Path) -> MutableMapping[str, Any]:
    module = _require_torch()
    if not path.is_file():
        raise FileNotFoundError(f"Artifact does not exist: {path}")
    try:
        payload = module.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "Installed PyTorch does not support the required weights_only=True "
            "artifact loader; refusing unsafe pickle fallback"
        ) from exc
    if not isinstance(payload, MutableMapping):
        raise ValueError(f"Artifact must contain a mapping: {path}")
    return payload


def _declared_artifact_hash(payload: Mapping[str, Any], label: str) -> str:
    candidates: List[str] = []
    for value in (
        payload.get("config_hash"),
        payload.get("artifact_config_hash"),
        payload.get("preprocessing_config", {}).get("config_hash")
        if isinstance(payload.get("preprocessing_config"), Mapping)
        else None,
        payload.get("artifact_config", {}).get("config_hash")
        if isinstance(payload.get("artifact_config"), Mapping)
        else None,
    ):
        text = str(value or "").strip()
        if text:
            candidates.append(text)
    if not candidates:
        raise ValueError(f"{label} artifact does not declare a config hash")
    if len(set(candidates)) != 1:
        raise ValueError(f"{label} artifact declares conflicting config hashes: {candidates}")
    return candidates[0]


def _normalise_person_ids(values: pd.Series, label: str) -> pd.Series:
    result = values.map(as_person_id)
    if result.eq("").any():
        raise ValueError(f"{label} contains an empty person_id")
    return result


def _ecg_numeric_artifact_feature_order(
    scaler_feature_order: Sequence[str],
) -> Tuple[str, ...]:
    """Map the frozen ECG table namespace to the artifact tensor namespace.

    ECG numeric scaler entries are fitted from wide-table columns named
    ``ecg__Rate``, ``ecg__PR``, and so on, while the canonical participant
    artifacts store the same ordered tensor axis as ``Rate``, ``PR``, etc.
    Accept either the fitted, uniformly namespaced representation or the
    legacy unprefixed representation.  Mixed namespaces remain an error so a
    malformed scaler cannot silently change which statistic a position means.
    """

    raw = tuple(str(value) for value in scaler_feature_order)
    prefix = "ecg__"
    prefixed = tuple(name.startswith(prefix) for name in raw)
    if any(prefixed) and not all(prefixed):
        raise ValueError(
            "Scaler ecg_numeric feature_order mixes ecg__-prefixed and "
            "unprefixed names"
        )
    canonical = (
        tuple(name[len(prefix) :] for name in raw)
        if raw and all(prefixed)
        else raw
    )
    if any(not name for name in canonical) or len(canonical) != len(set(canonical)):
        raise ValueError(
            "Scaler ecg_numeric feature_order is empty or collides after "
            "removing the exact ecg__ namespace prefix"
        )
    return canonical


def _evenly_spaced_indices(length: int, cap: int) -> List[int]:
    if length <= cap:
        return list(range(length))
    # linspace keeps the endpoints and provides deterministic coverage of a
    # volume instead of silently taking only its anterior/first slices.
    return np.linspace(0, length - 1, num=cap).round().astype(int).tolist()


@dataclass(frozen=True)
class FrozenScalerGroup:
    """Validated scaler arrays for one final feature dimension."""

    name: str
    feature_order: Tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    scale_applied: np.ndarray

    @classmethod
    def from_payload(cls, name: str, payload: Mapping[str, Any]) -> "FrozenScalerGroup":
        order_raw = payload.get("feature_order")
        features_raw = payload.get("features")
        if not isinstance(order_raw, list) or not order_raw:
            raise ValueError(f"Scaler {name} has no non-empty feature_order")
        order = tuple(str(item) for item in order_raw)
        if any(not item for item in order) or len(order) != len(set(order)):
            raise ValueError(f"Scaler {name} feature_order must be non-empty and unique")
        if not isinstance(features_raw, Mapping) or set(features_raw) != set(order):
            raise ValueError(f"Scaler {name} features do not exactly match feature_order")
        centers: List[float] = []
        scales: List[float] = []
        applied: List[bool] = []
        for feature in order:
            entry = features_raw[feature]
            if not isinstance(entry, Mapping):
                raise ValueError(f"Scaler {name}:{feature} entry is not an object")
            center = float(entry.get("center"))
            scale = float(entry.get("scale"))
            if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
                raise ValueError(
                    f"Scaler {name}:{feature} has invalid center/scale {center}/{scale}"
                )
            centers.append(center)
            scales.append(scale)
            applied.append(_strict_bool(entry.get("scale_applied"), f"{name}:{feature}.scale_applied"))
        return cls(
            name=name,
            feature_order=order,
            center=np.asarray(centers, dtype=np.float32),
            scale=np.asarray(scales, dtype=np.float32),
            scale_applied=np.asarray(applied, dtype=bool),
        )

    def normalise(self, values: Any, observed_mask: Any, label: str) -> Any:
        """Scale observed entries and force unobserved storage to exact zero."""
        module = _require_torch()
        values_tensor = _as_float_tensor(values, label)
        mask_tensor = _as_bool_tensor(observed_mask, f"{label} observed mask")
        if values_tensor.shape != mask_tensor.shape:
            raise ValueError(
                f"{label} value/mask shape mismatch: {tuple(values_tensor.shape)} vs "
                f"{tuple(mask_tensor.shape)}"
            )
        if values_tensor.ndim < 1 or values_tensor.shape[-1] != len(self.feature_order):
            raise ValueError(
                f"{label} last dimension must be {len(self.feature_order)}, got "
                f"{tuple(values_tensor.shape)}"
            )
        if module.any(mask_tensor & ~module.isfinite(values_tensor)).item():
            raise ValueError(f"{label} has a non-finite value marked observed")
        center = module.as_tensor(self.center, dtype=module.float32)
        scale = module.as_tensor(self.scale, dtype=module.float32)
        applied = module.as_tensor(self.scale_applied, dtype=module.bool)
        transformed = module.where(applied, (values_tensor - center) / scale, values_tensor)
        if module.any(mask_tensor & ~module.isfinite(transformed)).item():
            raise ValueError(f"{label} normalization produced a non-finite observed value")
        return module.where(mask_tensor, transformed, module.zeros_like(transformed))


class OT2V2Dataset(Dataset):
    """Dataset over a task-specific V2 multimodal master index.

    Parameters are explicit because the master paths may be portable paths
    relative to the configured ``artifact_root``.  ``split`` is optional;
    one dataset instance can therefore also support cohort-wide audits.
    """

    def __init__(
        self,
        master_csv: str | Path,
        scaler_json: str | Path,
        *,
        task: str,
        artifact_root: str | Path | None,
        split: Optional[str] = None,
        clinical_features_csv: str | Path | None = None,
        master_metadata_json: str | Path | None = None,
        environment_schema_json: str | Path | None = None,
        exclude_default_environment_features: bool = True,
        max_retinal_assets_per_group: int | Mapping[str, int] = DEFAULT_RETINAL_ASSET_CAPS,
        max_retinal_views_per_asset: int = 8,
        default_retinal_size: int = 224,
        require_y: bool = True,
        validate_scaler_hash: bool = True,
        loaded_source_groups: Optional[Iterable[str]] = None,
    ) -> None:
        _require_torch()
        self.master_path = Path(master_csv).resolve()
        self.scaler_path = Path(scaler_json).resolve()
        self.artifact_root = Path(artifact_root).resolve() if artifact_root is not None else None
        self.task = str(task).strip()
        self.require_y = bool(require_y)
        requested_groups = (
            set(PRIMARY_SOURCE_ORDER)
            if loaded_source_groups is None
            else {str(group).strip() for group in loaded_source_groups}
        )
        unknown_groups = sorted(requested_groups - set(PRIMARY_SOURCE_ORDER))
        if unknown_groups:
            raise ValueError(
                f"loaded_source_groups contains unknown groups: {unknown_groups}"
            )
        if not requested_groups:
            raise ValueError("loaded_source_groups must retain at least one modality")
        self.loaded_source_groups = tuple(
            group for group in PRIMARY_SOURCE_ORDER if group in requested_groups
        )
        self.max_retinal_views_per_asset = int(max_retinal_views_per_asset)
        self.default_retinal_size = int(default_retinal_size)
        if not self.task:
            raise ValueError("task must be non-empty")
        if self.max_retinal_views_per_asset <= 0 or self.default_retinal_size <= 0:
            raise ValueError("Retinal view cap and default size must be positive")
        self.retinal_asset_caps = self._retinal_caps(max_retinal_assets_per_group)
        self._flio_monolithic_warning_emitted = False
        self.retinal_artifact_config_hash: Optional[str] = None
        self.retinal_model_view_config_hash: Optional[str] = None

        self.index = self._load_master(split)
        self.scaler_payload = _read_json(self.scaler_path, "scaler JSON")
        self.scalers = self._validate_and_load_scalers(validate_scaler_hash)
        self.ecg_numeric_artifact_feature_order = _ecg_numeric_artifact_feature_order(
            self.scalers["ecg_numeric"].feature_order
        )
        self._validate_wearable_path_lineage()
        self._validate_master_metadata(master_metadata_json)
        self._configure_environment_features(
            environment_schema_json,
            exclude_default=bool(exclude_default_environment_features),
        )
        self.clinical_values_raw = self._load_clinical_once(clinical_features_csv)
        self.data_contract_audit = {
            "primary_source_order": list(PRIMARY_SOURCE_ORDER),
            "loaded_source_groups": list(self.loaded_source_groups),
            "unloaded_source_groups": [
                group
                for group in PRIMARY_SOURCE_ORDER
                if group not in set(self.loaded_source_groups)
            ],
            "environment": {
                "canonical_feature_names": list(self.environment_canonical_feature_names),
                "model_feature_names": list(self.environment_feature_names),
                "model_selected_canonical_indices": list(self.environment_feature_indices),
                "default_excluded_feature_names": list(
                    self.environment_excluded_feature_names
                ),
            },
            "retinal": {
                "flio_archive_policy": "never decode, return, or cache lossless archive",
                "monolithic_torch_save_limitation": (
                    "torch.load materializes compressed archive bytes; use the slim "
                    "model-view index from build_retinal_model_view_sidecars.py "
                    "for archive-free repeated training I/O"
                ),
                "upstream_artifact_config_hash": self.retinal_artifact_config_hash,
                "model_view_config_hash": self.retinal_model_view_config_hash,
            },
            "wearable": {
                "path_resolution_priority": {
                    "cgm": ["cgm_path", "wearable_path"],
                    "activity": ["activity_path", "wearable_path"],
                },
                "deserialization": "torch.load(weights_only=True), no pickle fallback",
            },
            "ecg": {
                "numeric_scaler_feature_order": list(
                    self.scalers["ecg_numeric"].feature_order
                ),
                "numeric_artifact_feature_order": list(
                    self.ecg_numeric_artifact_feature_order
                ),
                "namespace_mapping": (
                    "strip one exact, uniform ecg__ prefix from scaler feature names"
                ),
            },
        }
        self.model_feature_dimensions = {
            "clinical_dim": len(self.scalers["clinical"].feature_order),
            "environment_dim": len(self.environment_feature_names),
            "activity_dim": len(self.scalers["activity"].feature_order),
            "cgm_dim": len(self.scalers["cgm"].feature_order),
            "ecg_waveform_dim": len(self.scalers["ecg_waveform"].feature_order),
            "ecg_numeric_dim": len(self.scalers["ecg_numeric"].feature_order),
        }

    @staticmethod
    def _retinal_caps(value: int | Mapping[str, int]) -> Dict[str, int]:
        if isinstance(value, Mapping):
            unknown = sorted(set(value) - set(RETINAL_GROUPS))
            if unknown:
                raise ValueError(f"Unknown retinal asset cap groups: {unknown}")
            caps = {group: int(value.get(group, DEFAULT_RETINAL_ASSET_CAPS[group])) for group in RETINAL_GROUPS}
        else:
            caps = {group: int(value) for group in RETINAL_GROUPS}
        if any(cap <= 0 for cap in caps.values()):
            raise ValueError("Every retinal asset cap must be positive")
        return caps

    def _load_master(self, split: Optional[str]) -> pd.DataFrame:
        if not self.master_path.is_file():
            raise FileNotFoundError(f"Master CSV does not exist: {self.master_path}")
        frame = pd.read_csv(
            self.master_path,
            dtype={"person_id": "string"},
            keep_default_na=False,
            low_memory=False,
        )
        required = {
            "person_id", "task", "split", "clinical_features_path",
            "retinal_asset_paths_json", "environment_path", "wearable_path", "ecg_path",
            *SOURCE_FLAG_BY_GROUP.values(),
        }
        if self.require_y:
            required.add("y")
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Master CSV missing required columns: {missing}")
        frame = frame.copy()
        # New masters may physically separate the two wearable groups.  Empty
        # optional columns preserve the legacy combined wearable_path contract.
        for column in ("cgm_path", "activity_path"):
            if column not in frame:
                frame[column] = ""
        frame["person_id"] = _normalise_person_ids(frame["person_id"], "master CSV")
        duplicate_ids = frame.loc[frame["person_id"].duplicated(keep=False), "person_id"].tolist()
        if duplicate_ids:
            raise ValueError(f"Master CSV must have one row per person_id: {duplicate_ids[:10]}")
        tasks = set(frame["task"].astype(str).str.strip())
        if tasks != {self.task}:
            raise ValueError(f"Master task mismatch: requested {self.task!r}, observed {sorted(tasks)}")
        frame["split"] = frame["split"].astype(str).str.strip().str.lower()
        if frame["split"].eq("").any():
            raise ValueError("Master CSV contains an empty split")
        unknown_splits = sorted(set(frame["split"]) - {"train", "val", "test"})
        if unknown_splits:
            raise ValueError(
                "Master CSV contains splits outside train/val/test: "
                f"{unknown_splits}"
            )
        for column in set(SOURCE_FLAG_BY_GROUP.values()):
            frame[column] = [
                _strict_bool(value, f"master {column}") for value in frame[column]
            ]
        if "y" in frame:
            numeric_y = pd.to_numeric(frame["y"], errors="coerce")
            present_y = frame["y"].astype(str).str.strip().ne("")
            if self.require_y and ((~np.isfinite(numeric_y)) | ~numeric_y.isin([0, 1])).any():
                raise ValueError("Master y must contain only finite binary labels 0/1")
            if (present_y & ((~np.isfinite(numeric_y)) | ~numeric_y.isin([0, 1]))).any():
                raise ValueError("Every present master y value must be binary 0/1")
            frame["y"] = numeric_y
        if "task_config_hash" in frame:
            hashes = set(frame["task_config_hash"].astype(str).str.strip()) - {""}
            if len(hashes) > 1:
                raise ValueError(f"Master contains multiple task_config_hash values: {sorted(hashes)}")
        if split is not None:
            requested = str(split).strip().lower()
            frame = frame[frame["split"].eq(requested)].reset_index(drop=True)
            if frame.empty:
                raise ValueError(f"Master contains no rows for split={requested!r}")
        else:
            frame = frame.reset_index(drop=True)
        return frame

    def _validate_and_load_scalers(self, validate_hash: bool) -> Dict[str, FrozenScalerGroup]:
        payload = self.scaler_payload
        if payload.get("scaler_format_version") != SCALER_FORMAT_VERSION:
            raise ValueError(
                f"Scaler format mismatch: {payload.get('scaler_format_version')!r} vs "
                f"{SCALER_FORMAT_VERSION!r}"
            )
        if payload.get("preprocessing_version") != PREPROCESSING_VERSION:
            raise ValueError("Scaler preprocessing_version does not match this code")
        if payload.get("fit_split") != "recommended_split=train":
            raise ValueError("Scaler must be frozen from recommended_split=train")
        declared_hash = str(payload.get("scaler_config_hash", "")).strip()
        if not declared_hash:
            raise ValueError("Scaler JSON does not declare scaler_config_hash")
        if validate_hash:
            hash_payload = dict(payload)
            hash_payload.pop("scaler_config_hash", None)
            actual_hash = canonical_json_hash(hash_payload)
            if actual_hash != declared_hash:
                raise ValueError(
                    f"Scaler config hash mismatch: declared {declared_hash}, computed {actual_hash}"
                )
        source_info = payload.get("source_config_hashes")
        if not isinstance(source_info, Mapping):
            raise ValueError("Scaler JSON has no source_config_hashes object")
        clinical_info = source_info.get("clinical")
        if not isinstance(clinical_info, Mapping) or str(clinical_info.get("task", "")) != self.task:
            raise ValueError("Scaler clinical task does not match requested task")
        master_task_hashes = set(self.index.get("task_config_hash", pd.Series(dtype=str)).astype(str).str.strip()) - {""}
        scaler_task_hash = str(clinical_info.get("task_config_hash", "")).strip()
        if master_task_hashes and master_task_hashes != {scaler_task_hash}:
            raise ValueError("Master/scaler task_config_hash mismatch")
        raw_scalers = payload.get("scalers")
        if not isinstance(raw_scalers, Mapping):
            raise ValueError("Scaler JSON has no scalers object")
        missing = sorted(REQUIRED_SCALERS - set(raw_scalers))
        if missing:
            raise ValueError(f"Scaler JSON missing required source scalers: {missing}")
        return {
            name: FrozenScalerGroup.from_payload(name, raw_scalers[name])
            for name in sorted(REQUIRED_SCALERS)
        }

    def _validate_wearable_path_lineage(self) -> None:
        """Bind optional wearable sidecar paths to the frozen source hash.

        Generated masters carry per-row hashes for every effective wearable
        path.  Legacy masters omitted these columns, so their artifact hash is
        still checked lazily when the participant artifact is opened.
        """
        source_info = self.scaler_payload.get("source_config_hashes", {})
        wearable_info = (
            source_info.get("wearable", {})
            if isinstance(source_info, Mapping)
            else {}
        )
        expected = (
            str(wearable_info.get("config_hash", "")).strip()
            if isinstance(wearable_info, Mapping)
            else ""
        )
        if not expected:
            raise ValueError("Frozen scaler has no wearable config hash")

        for group, path_column in (
            ("cgm", "cgm_path"),
            ("activity", "activity_path"),
        ):
            hash_column = f"{group}_config_hash"
            if hash_column not in self.index:
                continue
            paths = self.index[path_column].astype(str).str.strip()
            hashes = self.index[hash_column].astype(str).str.strip()
            missing = paths.ne("") & hashes.eq("")
            if missing.any():
                people = self.index.loc[missing, "person_id"].tolist()
                raise ValueError(
                    f"Master {group} sidecar path lacks config hash: {people[:10]}"
                )
            mismatch = hashes.ne("") & hashes.ne(expected)
            if mismatch.any():
                people = self.index.loc[mismatch, "person_id"].tolist()
                raise ValueError(
                    f"Master/scaler {group} sidecar config hash mismatch: "
                    f"{people[:10]}"
                )

        if "wearable_config_hash" in self.index:
            hashes = self.index["wearable_config_hash"].astype(str).str.strip()
            mismatch = hashes.ne("") & hashes.ne(expected)
            if mismatch.any():
                people = self.index.loc[mismatch, "person_id"].tolist()
                raise ValueError(
                    "Master/scaler wearable config hash mismatch: "
                    f"{people[:10]}"
                )

    def _validate_master_metadata(self, explicit_path: str | Path | None) -> None:
        if explicit_path is not None:
            path = Path(explicit_path).resolve()
        else:
            candidate = self.master_path.with_suffix(".metadata.json")
            path = candidate if candidate.is_file() else None
        if path is None:
            return
        metadata = _read_json(path, "master metadata JSON")
        if str(metadata.get("task", "")) != self.task:
            raise ValueError("Master metadata task does not match requested task")
        # Metadata describes the full CSV, so compare before/after split by
        # reading the cheap row count when a split subset was requested.
        expected_rows = metadata.get("n_rows")
        if expected_rows is not None:
            with self.master_path.open("rb") as stream:
                full_count = sum(1 for _ in stream) - 1
            if int(expected_rows) != full_count:
                raise ValueError("Master metadata n_rows does not match master CSV")
        source_indexes = metadata.get("source_indexes", {})
        scaler_sources = self.scaler_payload.get("source_config_hashes", {})
        for source in ("environment", "wearable", "ecg"):
            left = source_indexes.get(source, {}) if isinstance(source_indexes, Mapping) else {}
            right = scaler_sources.get(source, {}) if isinstance(scaler_sources, Mapping) else {}
            left_hash = str(left.get("config_hash", "")) if isinstance(left, Mapping) else ""
            right_hash = str(right.get("config_hash", "")) if isinstance(right, Mapping) else ""
            if left_hash and right_hash and left_hash != right_hash:
                raise ValueError(f"Master/scaler {source} config hash mismatch")
        retinal_info = (
            source_indexes.get("retinal", {})
            if isinstance(source_indexes, Mapping)
            else {}
        )
        if isinstance(retinal_info, Mapping):
            retinal_hash = str(retinal_info.get("config_hash", "")).strip()
            if retinal_hash:
                self.retinal_artifact_config_hash = retinal_hash
            model_view_hash = str(
                retinal_info.get("model_view_config_hash", "") or ""
            ).strip()
            if model_view_hash:
                self.retinal_model_view_config_hash = model_view_hash
        master_schema_hash = str(metadata.get("clinical_schema_hash", ""))
        clinical_source = scaler_sources.get("clinical", {}) if isinstance(scaler_sources, Mapping) else {}
        scaler_schema_hash = str(clinical_source.get("schema_hash", "")) if isinstance(clinical_source, Mapping) else ""
        if master_schema_hash and scaler_schema_hash and master_schema_hash != scaler_schema_hash:
            raise ValueError("Master/scaler clinical schema hash mismatch")

    def _configure_environment_features(
        self,
        explicit_path: str | Path | None,
        *,
        exclude_default: bool,
    ) -> None:
        """Bind the canonical 67-column schema and derive the default 64-column view."""
        canonical = tuple(self.scalers["environment"].feature_order)
        candidates: List[Path] = []
        if explicit_path is not None:
            explicit = Path(explicit_path)
            if not explicit.is_absolute():
                if self.artifact_root is None:
                    raise ValueError(
                        "Relative environment_schema_json requires artifact_root"
                    )
                explicit = self.artifact_root / explicit
            candidates.append(explicit.resolve())
        else:
            # The schema is emitted beside environment participant artifacts.
            for raw in self.index["environment_path"]:
                if str(raw).strip():
                    artifact_path = self._resolve_path(raw, "environment")
                    assert artifact_path is not None
                    candidates.append(artifact_path.parent / "environment_5min_schema.json")
                    break
            # Scaler provenance is a secondary discovery route when a filtered
            # cohort contains no environment artifact at all.
            for item in self.scaler_payload.get("input_files", []):
                if not isinstance(item, Mapping):
                    continue
                raw = str(item.get("path", "")).strip()
                if Path(raw).name == "environment_5min_schema.json":
                    candidates.append(Path(raw))
        schema_path = next((path for path in candidates if path.is_file()), None)
        if schema_path is None:
            if exclude_default:
                raise FileNotFoundError(
                    "Could not locate environment_5min_schema.json required for default "
                    "technical-feature exclusion; pass environment_schema_json explicitly"
                )
            excluded: Tuple[str, ...] = ()
        else:
            schema = _read_json(schema_path, "environment schema JSON")
            schema_names = tuple(str(name) for name in schema.get("feature_names", []))
            if schema_names != canonical:
                raise ValueError(
                    "Environment schema canonical feature order does not match frozen scaler"
                )
            expected_hash = str(
                self.scaler_payload.get("source_config_hashes", {})
                .get("environment", {})
                .get("config_hash", "")
            )
            observed_hash = str(schema.get("config_hash", ""))
            if expected_hash and observed_hash != expected_hash:
                raise ValueError("Environment schema/scaler config hash mismatch")
            expected_schema_hash = str(
                self.scaler_payload.get("source_config_hashes", {})
                .get("environment", {})
                .get("schema_hash", "")
            )
            if expected_schema_hash and canonical_json_hash(schema) != expected_schema_hash:
                raise ValueError("Environment schema content hash does not match scaler provenance")
            raw_excluded = schema.get("default_model_excluded_feature_names", [])
            if not isinstance(raw_excluded, list):
                raise ValueError(
                    "Environment schema default_model_excluded_feature_names must be a list"
                )
            excluded = tuple(str(name) for name in raw_excluded) if exclude_default else ()
        if len(excluded) != len(set(excluded)):
            raise ValueError("Environment default exclusion list contains duplicates")
        unknown = sorted(set(excluded) - set(canonical))
        if unknown:
            raise ValueError(f"Environment default exclusions are not canonical features: {unknown}")
        selected = tuple(index for index, name in enumerate(canonical) if name not in set(excluded))
        if not selected:
            raise ValueError("Environment default exclusion removed every feature")
        self.environment_canonical_feature_names = canonical
        self.environment_excluded_feature_names = excluded
        self.environment_feature_indices = selected
        self.environment_feature_names = tuple(canonical[index] for index in selected)

    def _resolve_path(self, value: Any, label: str, *, allow_empty: bool = False) -> Optional[Path]:
        text = str(value or "").strip()
        if not text:
            if allow_empty:
                return None
            raise ValueError(f"{label} path is empty")
        path = Path(text)
        if not path.is_absolute():
            if self.artifact_root is None:
                raise ValueError(
                    f"{label} uses relative path {text!r}, but artifact_root was not provided"
                )
            path = (self.artifact_root / path).resolve()
            try:
                path.relative_to(self.artifact_root)
            except ValueError as exc:
                raise ValueError(
                    f"{label} relative path escapes artifact_root: {text!r}"
                ) from exc
            return path
        return path.resolve()

    def _load_clinical_once(self, explicit_path: str | Path | None) -> np.ndarray:
        declared = {
            str(value).strip()
            for value in self.index["clinical_features_path"]
            if str(value).strip()
        }
        if explicit_path is None:
            if len(declared) != 1:
                raise ValueError(
                    "Master must declare exactly one clinical_features_path when no override is supplied"
                )
            path = self._resolve_path(next(iter(declared)), "clinical features")
            assert path is not None
        else:
            path = Path(explicit_path)
            if not path.is_absolute():
                path = self._resolve_path(path, "clinical features override")
                assert path is not None
            else:
                path = path.resolve()
        # A true modality-subset run must not read a disabled participant-level
        # source.  Keep the resolved path for lineage, but construct the exact
        # zero/absent placeholder directly from the frozen scaler width.
        if "clinical_data" not in set(self.loaded_source_groups):
            self.clinical_features_path = path
            return np.zeros(
                (len(self.index), len(self.scalers["clinical"].feature_order)),
                dtype=np.float32,
            )
        if not path.is_file():
            raise FileNotFoundError(f"Clinical wide CSV does not exist: {path}")
        # Expose the exact byte source for checkpoint provenance.  The table is
        # loaded once per split-specific Dataset, but all three must bind to
        # the same task-specific wide file.
        self.clinical_features_path = path
        frame = pd.read_csv(
            path,
            dtype={"person_id": "string"},
            keep_default_na=True,
            low_memory=False,
        )
        if "person_id" not in frame:
            raise ValueError("Clinical wide CSV is missing person_id")
        frame["person_id"] = _normalise_person_ids(frame["person_id"], "clinical wide CSV")
        if frame["person_id"].duplicated().any():
            raise ValueError("Clinical wide CSV must have at most one row per person_id")
        expected = list(self.scalers["clinical"].feature_order)
        observed = [column for column in frame.columns if column != "person_id"]
        if observed != expected:
            raise ValueError("Clinical wide feature columns/order do not match frozen scaler")
        aligned = frame.set_index("person_id").reindex(self.index["person_id"])
        missing_people = aligned.index[aligned[expected].isna().all(axis=1)].tolist()
        raw_ids = set(frame["person_id"])
        truly_missing = [person_id for person_id in missing_people if person_id not in raw_ids]
        if truly_missing:
            raise ValueError(f"Clinical wide CSV is missing cohort people: {truly_missing[:10]}")
        try:
            return aligned[expected].to_numpy(dtype=np.float32)
        except Exception as exc:
            raise ValueError("Clinical wide features must be numeric") from exc

    def __len__(self) -> int:
        return int(len(self.index))

    def _explicit_presence(self, row: pd.Series) -> Dict[str, bool]:
        return {
            group: _strict_bool(row[flag], f"master {flag}")
            for group, flag in SOURCE_FLAG_BY_GROUP.items()
        }

    def _load_environment(self, row: pd.Series, present: bool) -> Dict[str, Any]:
        module = _require_torch()
        feature_count = len(self.environment_feature_names)
        if not present:
            return {
                "environment": module.zeros((0, feature_count), dtype=module.float32),
                "environment_feature_mask": module.zeros((0, feature_count), dtype=module.bool),
                "environment_slot_mask": module.zeros((0,), dtype=module.bool),
            }
        path = self._resolve_path(row["environment_path"], "environment")
        assert path is not None
        artifact = _safe_torch_load(path)
        self._validate_artifact_person(artifact, row["person_id"], "Environment", path)
        self._validate_artifact_source_hash(artifact, "environment", "Environment", path)
        names = [str(value) for value in artifact.get("feature_names", [])]
        if names != list(self.scalers["environment"].feature_order):
            raise ValueError(f"Environment feature order disagrees with scaler: {path}")
        values = _as_float_tensor(artifact.get("environment"), "environment", rank=2)
        channel_mask = validated_boolean_mask(
            np.asarray(artifact.get("environment_channel_observed_mask")),
            "environment_channel_observed_mask",
        )
        row_count = np.asarray(artifact.get("environment_row_count"))
        feature_mask = environment_feature_mask(
            names,
            [str(value) for value in artifact.get("sensor_columns", [])],
            channel_mask,
            row_count,
        )
        explicit_feature_mask = artifact.get("environment_feature_mask")
        if explicit_feature_mask is not None:
            explicit_array = validated_boolean_mask(
                np.asarray(explicit_feature_mask), "environment_feature_mask"
            )
            if not np.array_equal(explicit_array, feature_mask):
                raise ValueError(f"Environment explicit and derived masks disagree: {path}")
        mask_tensor = _as_bool_tensor(feature_mask, "environment_feature_mask")
        slot_mask = _as_bool_tensor(
            artifact.get("environment_slot_mask"), "environment_slot_mask"
        )
        if values.shape[0] != slot_mask.shape[0]:
            raise ValueError(f"Environment slot mask length mismatch: {path}")
        if not bool(slot_mask.any().item()):
            raise ValueError(f"Master declares environment present but slot mask is empty: {path}")
        normalized = self.scalers["environment"].normalise(
            values, mask_tensor, "environment"
        )
        selected = module.as_tensor(self.environment_feature_indices, dtype=module.long)
        return {
            "environment": normalized.index_select(1, selected),
            "environment_feature_mask": mask_tensor.index_select(1, selected),
            "environment_slot_mask": slot_mask,
        }

    def _load_wearable(
        self, row: pd.Series, cgm_present: bool, activity_present: bool
    ) -> Dict[str, Any]:
        module = _require_torch()
        result: Dict[str, Any] = {}
        artifacts: Dict[Path, MutableMapping[str, Any]] = {}
        for group, present in (("cgm", cgm_present), ("activity", activity_present)):
            width = len(self.scalers[group].feature_order)
            if not present:
                result[group] = module.zeros((0, width), dtype=module.float32)
                result[f"{group}_feature_mask"] = module.zeros((0, width), dtype=module.bool)
                result[f"{group}_observed_mask"] = module.zeros((0,), dtype=module.bool)
                result[f"{group}_valid_mask"] = module.zeros((0,), dtype=module.bool)
                continue
            specific_path = str(row.get(f"{group}_path", "") or "").strip()
            path_value = specific_path or row["wearable_path"]
            path = self._resolve_path(path_value, f"wearable {group}")
            assert path is not None
            if path not in artifacts:
                artifact = _safe_torch_load(path)
                self._validate_artifact_person(
                    artifact, row["person_id"], f"Wearable {group}", path
                )
                self._validate_artifact_source_hash(
                    artifact, "wearable", f"Wearable {group}", path
                )
                artifacts[path] = artifact
            artifact = artifacts[path]
            names = [str(value) for value in artifact.get(f"{group}_feature_names", [])]
            if names != list(self.scalers[group].feature_order):
                raise ValueError(f"Wearable {group} feature order disagrees with scaler: {path}")
            values = _as_float_tensor(artifact.get(group), group, rank=2)
            feature_mask = _as_bool_tensor(
                artifact.get(f"{group}_feature_mask"), f"{group}_feature_mask"
            )
            observed_mask = _as_bool_tensor(
                artifact.get(f"{group}_observed_mask"), f"{group}_observed_mask"
            )
            valid_mask = _as_bool_tensor(
                artifact.get(f"{group}_valid_mask"), f"{group}_valid_mask"
            )
            if values.shape[0] != observed_mask.shape[0] or values.shape[0] != valid_mask.shape[0]:
                raise ValueError(f"Wearable {group} slot-mask length mismatch: {path}")
            if not bool(observed_mask.any().item()):
                raise ValueError(f"Master declares {group} present but observed mask is empty: {path}")
            result[group] = self.scalers[group].normalise(values, feature_mask, group)
            result[f"{group}_feature_mask"] = feature_mask
            result[f"{group}_observed_mask"] = observed_mask
            result[f"{group}_valid_mask"] = valid_mask
        return result

    def _load_ecg(self, row: pd.Series, present: bool) -> Dict[str, Any]:
        module = _require_torch()
        lead_count = len(self.scalers["ecg_waveform"].feature_order)
        numeric_count = len(self.scalers["ecg_numeric"].feature_order)
        if not present:
            return {
                "ecg_waveform": module.zeros((0, lead_count), dtype=module.float32),
                "ecg_value_mask": module.zeros((0, lead_count), dtype=module.bool),
                "ecg_sample_mask": module.zeros((0,), dtype=module.bool),
                "ecg_lead_mask": module.zeros((lead_count,), dtype=module.bool),
                "ecg_numeric": module.zeros((numeric_count,), dtype=module.float32),
                "ecg_numeric_mask": module.zeros((numeric_count,), dtype=module.bool),
            }
        path = self._resolve_path(row["ecg_path"], "ECG")
        assert path is not None
        artifact = _safe_torch_load(path)
        self._validate_artifact_person(artifact, row["person_id"], "ECG", path)
        self._validate_artifact_source_hash(artifact, "ecg", "ECG", path)
        leads = [str(value) for value in artifact.get("lead_names", [])]
        numeric_names = [str(value) for value in artifact.get("numeric_feature_names", [])]
        if leads != list(self.scalers["ecg_waveform"].feature_order):
            raise ValueError(f"ECG lead order disagrees with scaler: {path}")
        if numeric_names != list(self.ecg_numeric_artifact_feature_order):
            raise ValueError(
                "ECG numeric feature order disagrees with the scaler after "
                f"strict ecg__ namespace normalization: {path}"
            )
        waveform = _as_float_tensor(artifact.get("ecg_waveform"), "ecg_waveform", rank=2)
        value_mask = _as_bool_tensor(artifact.get("ecg_value_mask"), "ecg_value_mask")
        sample_mask = _as_bool_tensor(
            artifact.get("ecg_time_mask", artifact.get("ecg_sample_mask")),
            "ecg_time_mask",
        )
        lead_mask = _as_bool_tensor(artifact.get("ecg_lead_mask"), "ecg_lead_mask")
        numeric = _as_float_tensor(artifact.get("ecg_numeric"), "ecg_numeric", rank=1)
        numeric_mask = _as_bool_tensor(artifact.get("ecg_numeric_mask"), "ecg_numeric_mask")
        if tuple(value_mask.shape) != tuple(waveform.shape):
            raise ValueError(f"ECG value-mask shape mismatch: {path}")
        if sample_mask.shape != (waveform.shape[0],) or lead_mask.shape != (waveform.shape[1],):
            raise ValueError(f"ECG sample/lead mask shape mismatch: {path}")
        effective_value_mask = (
            value_mask & sample_mask.unsqueeze(-1) & lead_mask.unsqueeze(0)
        )
        if not bool(effective_value_mask.any().item()):
            raise ValueError(f"Master declares ECG present but no waveform values are usable: {path}")
        return {
            "ecg_waveform": self.scalers["ecg_waveform"].normalise(
                waveform, effective_value_mask, "ecg_waveform"
            ),
            "ecg_value_mask": effective_value_mask,
            "ecg_sample_mask": sample_mask,
            "ecg_lead_mask": lead_mask,
            "ecg_numeric": self.scalers["ecg_numeric"].normalise(
                numeric, numeric_mask, "ecg_numeric"
            ),
            "ecg_numeric_mask": numeric_mask,
        }

    def _validate_artifact_person(
        self, artifact: Mapping[str, Any], person_id: str, label: str, path: Path
    ) -> None:
        if as_person_id(artifact.get("person_id")) != person_id:
            raise ValueError(f"{label} artifact/master person_id mismatch: {path}")

    def _validate_artifact_source_hash(
        self, artifact: Mapping[str, Any], source: str, label: str, path: Path
    ) -> None:
        sources = self.scaler_payload.get("source_config_hashes", {})
        expected_entry = sources.get(source, {}) if isinstance(sources, Mapping) else {}
        expected = str(expected_entry.get("config_hash", "")) if isinstance(expected_entry, Mapping) else ""
        if not expected:
            raise ValueError(f"Frozen scaler has no {source} config hash")
        observed = _declared_artifact_hash(artifact, label)
        if observed != expected:
            raise ValueError(f"{label} artifact/scaler config hash mismatch: {path}")

    def _retinal_paths(self, value: Any) -> List[Path]:
        try:
            raw = json.loads(str(value))
        except Exception as exc:
            raise ValueError("retinal_asset_paths_json is not valid JSON") from exc
        if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
            raise ValueError("retinal_asset_paths_json must be a JSON list of strings")
        if len(raw) != len(set(raw)):
            raise ValueError("retinal_asset_paths_json contains duplicate paths")
        paths: List[Path] = []
        for item in sorted(raw):
            path = self._resolve_path(item, "retinal asset")
            assert path is not None
            paths.append(path)
        return paths

    def _load_retinal_groups(
        self, row: pd.Series, presence: Mapping[str, bool]
    ) -> Dict[str, List[Dict[str, Any]]]:
        result: Dict[str, List[Dict[str, Any]]] = {group: [] for group in RETINAL_GROUPS}
        loaded_retinal = set(RETINAL_GROUPS).intersection(self.loaded_source_groups)
        if not loaded_retinal:
            return result
        any_declared = any(
            presence[group] for group in RETINAL_GROUPS if group in loaded_retinal
        )
        paths = self._retinal_paths(row["retinal_asset_paths_json"])
        if not any_declared:
            # Assets belonging to disabled retinal acquisitions are allowed;
            # they are deliberately neither opened nor validated here.
            enabled_paths = [
                path
                for path in paths
                if RETINAL_MODALITY_TO_GROUP.get(path.parent.name.strip().lower())
                in loaded_retinal
            ]
            if enabled_paths:
                raise ValueError(
                    "Master declares no enabled retinal subgroup for person "
                    f"{row['person_id']} but lists enabled assets"
                )
            return result
        if not paths:
            raise ValueError(f"Master declares retinal data but lists no assets: {row['person_id']}")

        # Paths and view choices are sorted/deterministic.  We retain only model
        # tensors and compact metadata; the giant optional FLIO archive is not
        # propagated or decoded.
        for path in paths:
            # Canonical V2 artifacts are stored as
            # ``.../retinal_assets/<hash>/<modality>/<asset>.pt``.  Use that
            # audited path component to enforce caps *before* torch.load;
            # otherwise a capped FLIO item would still materialize its very
            # large lossless archive on every epoch.  Unknown layouts fall
            # back to protocol_metadata after loading.
            path_modality = path.parent.name.strip().lower()
            path_group = RETINAL_MODALITY_TO_GROUP.get(path_modality)
            if path_group is None and loaded_retinal != set(RETINAL_GROUPS):
                raise ValueError(
                    "Selective retinal loading requires the parent directory to "
                    f"identify a canonical modality before torch.load: {path}"
                )
            if path_group is not None and path_group not in loaded_retinal:
                continue
            if (
                path_group is not None
                and len(result[path_group]) >= self.retinal_asset_caps[path_group]
            ):
                continue
            artifact = _safe_torch_load(path)
            self._validate_artifact_person(artifact, row["person_id"], "Retinal", path)
            if self.retinal_artifact_config_hash is not None:
                observed_hash = _declared_artifact_hash(artifact, "Retinal")
                if observed_hash != self.retinal_artifact_config_hash:
                    raise ValueError(
                        f"Retinal artifact/master config hash mismatch: {path}"
                    )
            if self.retinal_model_view_config_hash is not None:
                observed_model_view_hash = str(
                    artifact.get("model_view_config_hash", "") or ""
                ).strip()
                if observed_model_view_hash != self.retinal_model_view_config_hash:
                    raise ValueError(
                        "Retinal model-view config hash mismatch: "
                        f"expected={self.retinal_model_view_config_hash}, "
                        f"observed={observed_model_view_hash!r}: {path}"
                    )
            protocol = artifact.get("protocol_metadata")
            if not isinstance(protocol, Mapping):
                raise ValueError(f"Retinal artifact lacks protocol_metadata: {path}")
            modality = str(protocol.get("modality", "")).strip().lower()
            if modality not in RETINAL_MODALITY_TO_GROUP:
                raise ValueError(f"Unknown retinal protocol modality {modality!r}: {path}")
            group = RETINAL_MODALITY_TO_GROUP[modality]
            if group not in loaded_retinal:
                raise ValueError(
                    "Selective retinal path classification disagrees with loaded "
                    f"groups for artifact: {path}"
                )
            if path_group is not None and path_group != group:
                raise ValueError(
                    f"Retinal artifact path modality {path_modality!r} disagrees "
                    f"with protocol_metadata {modality!r}: {path}"
                )
            if (
                group == "retinal_flio"
                and "flio_decay_cube_lossless_archive" in artifact
                and not self._flio_monolithic_warning_emitted
            ):
                warnings.warn(
                    "FLIO artifact is a monolithic torch.save payload: torch.load has "
                    "materialized its compressed lossless archive bytes even though the "
                    "Model V2 loader will neither decode nor retain them. Generate slim "
                    "model-view sidecars with build_retinal_model_view_sidecars.py "
                    "for efficient repeated training I/O.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._flio_monolithic_warning_emitted = True
            if len(result[group]) >= self.retinal_asset_caps[group]:
                del artifact
                continue
            views = _as_float_tensor(artifact.get("model_views"), "retinal model_views", rank=4)
            pixel_mask = _as_bool_tensor(
                artifact.get("model_pixel_mask"), "retinal model_pixel_mask"
            )
            quality_mask = _as_bool_tensor(
                artifact.get("view_quality_mask"), "retinal view_quality_mask"
            )
            if views.shape[1] != 3:
                raise ValueError(f"Retinal model views must have three channels: {path}")
            expected_pixel_shape = (views.shape[0], 1, views.shape[2], views.shape[3])
            if tuple(pixel_mask.shape) != expected_pixel_shape:
                raise ValueError(f"Retinal pixel-mask shape mismatch: {path}")
            if quality_mask.shape != (views.shape[0],):
                raise ValueError(f"Retinal quality-mask shape mismatch: {path}")
            if views.numel() and (
                not bool(_require_torch().isfinite(views).all().item())
                or bool(((views < 0) | (views > 1)).any().item())
            ):
                raise ValueError(f"Retinal model views must be finite and within [0,1]: {path}")
            selected = _evenly_spaced_indices(
                int(views.shape[0]), self.max_retinal_views_per_asset
            )
            selected_tensor = _require_torch().as_tensor(selected, dtype=_require_torch().long)
            views = views.index_select(0, selected_tensor)
            pixel_mask = pixel_mask.index_select(0, selected_tensor)
            quality_mask = quality_mask.index_select(0, selected_tensor)
            view_mask = quality_mask & pixel_mask.flatten(1).any(dim=1)
            all_view_names = [str(value) for value in artifact.get("model_view_names", [])]
            if all_view_names and len(all_view_names) != int(artifact["model_views"].shape[0]):
                raise ValueError(f"Retinal model_view_names length mismatch: {path}")
            view_names = [all_view_names[index] for index in selected] if all_view_names else []
            result[group].append(
                {
                    "views": views,
                    "pixel_mask": pixel_mask,
                    "view_mask": view_mask,
                    "metadata": {
                        "artifact_path": str(path),
                        "asset_id": str(artifact.get("asset_id", "")),
                        "asset_key": str(artifact.get("asset_key", "")),
                        "product_kind": str(artifact.get("product_kind", "")),
                        "protocol_metadata": dict(protocol),
                        "model_view_names": view_names,
                    },
                }
            )
            # Never retain or inspect flio_decay_cube_lossless_archive.
            del artifact

        for group in RETINAL_GROUPS:
            if group not in loaded_retinal:
                continue
            declared = bool(presence[group])
            observed = bool(result[group])
            if declared != observed:
                raise ValueError(
                    f"Master retinal flag {SOURCE_FLAG_BY_GROUP[group]}={declared} disagrees "
                    f"with loaded protocol metadata for person {row['person_id']}"
                )
        return result

    def __getitem__(self, index: int) -> Dict[str, Any]:
        module = _require_torch()
        row = self.index.iloc[int(index)]
        person_id = str(row["person_id"])
        acquired_presence = self._explicit_presence(row)
        loaded = set(self.loaded_source_groups)
        presence = {
            group: bool(acquired_presence[group] and group in loaded)
            for group in PRIMARY_SOURCE_ORDER
        }

        clinical_raw = module.as_tensor(
            np.ascontiguousarray(self.clinical_values_raw[int(index)]), dtype=module.float32
        )
        clinical_mask = module.isfinite(clinical_raw)
        if not presence["clinical_data"]:
            clinical_mask = module.zeros_like(clinical_mask)
        clinical_values = self.scalers["clinical"].normalise(
            clinical_raw, clinical_mask, "clinical_values"
        )

        sample: Dict[str, Any] = {
            "person_id": person_id,
            "task": self.task,
            "split": str(row["split"]),
            "clinical_values": clinical_values,
            "clinical_mask": clinical_mask,
            "environment_feature_names": list(self.environment_feature_names),
            "environment_canonical_feature_names": list(
                self.environment_canonical_feature_names
            ),
            "environment_excluded_feature_names": list(
                self.environment_excluded_feature_names
            ),
            "environment_feature_indices": list(self.environment_feature_indices),
        }
        sample.update(self._load_environment(row, presence["environment"]))
        sample.update(
            self._load_wearable(
                row,
                presence["wearable_blood_glucose"],
                presence["wearable_activity_monitor"],
            )
        )
        sample.update(self._load_ecg(row, presence["cardiac_ecg"]))
        sample.update(self._load_retinal_groups(row, presence))
        source_present = module.tensor(
            [presence[group] for group in PRIMARY_SOURCE_ORDER], dtype=module.bool
        )
        sample["source_present"] = source_present
        for group, flag in SOURCE_FLAG_BY_GROUP.items():
            sample[flag] = module.tensor(presence[group], dtype=module.bool)
        if "y" in row.index and np.isfinite(float(row["y"])):
            sample["y"] = module.tensor(float(row["y"]), dtype=module.float32)
        elif self.require_y:
            raise ValueError(f"Missing y for person_id={person_id}")
        if "label_value" in row.index:
            sample["label_value"] = row["label_value"]
        return sample


def _pad_temporal(
    samples: Sequence[Mapping[str, Any]], value_key: str, mask_keys: Sequence[str]
) -> Dict[str, Any]:
    module = _require_torch()
    values = [sample[value_key] for sample in samples]
    widths = {int(value.shape[1]) for value in values}
    if len(widths) != 1:
        raise ValueError(f"Batch {value_key} feature widths disagree: {sorted(widths)}")
    width = next(iter(widths))
    max_length = max(1, max(int(value.shape[0]) for value in values))
    batch_values = module.zeros((len(samples), max_length, width), dtype=module.float32)
    batch_masks: Dict[str, Any] = {}
    for key in mask_keys:
        exemplar = samples[0][key]
        suffix = tuple(int(size) for size in exemplar.shape[1:])
        batch_masks[key] = module.zeros(
            (len(samples), max_length, *suffix), dtype=module.bool
        )
    for batch_index, sample in enumerate(samples):
        length = int(sample[value_key].shape[0])
        if length == 0:
            continue
        batch_values[batch_index, :length] = sample[value_key]
        for key in mask_keys:
            mask = sample[key]
            if int(mask.shape[0]) != length:
                raise ValueError(f"Batch {key} length disagrees with {value_key}")
            batch_masks[key][batch_index, :length] = mask
    return {value_key: batch_values, **batch_masks}


def _pad_retinal_group(
    samples: Sequence[Mapping[str, Any]], group: str, default_size: int = 224
) -> Dict[str, Any]:
    module = _require_torch()
    asset_lists = [sample[group] for sample in samples]
    max_assets = max(1, max(len(assets) for assets in asset_lists))
    max_views = max(
        1,
        max(
            (int(asset["views"].shape[0]) for assets in asset_lists for asset in assets),
            default=0,
        ),
    )
    heights = {
        int(asset["views"].shape[2]) for assets in asset_lists for asset in assets
    }
    widths = {
        int(asset["views"].shape[3]) for assets in asset_lists for asset in assets
    }
    if len(heights) > 1 or len(widths) > 1:
        raise ValueError(f"Batch {group} assets disagree on model-view spatial size")
    height = next(iter(heights), int(default_size))
    width = next(iter(widths), int(default_size))
    views = module.zeros(
        (len(samples), max_assets, max_views, 3, height, width), dtype=module.float32
    )
    pixel_mask = module.zeros(
        (len(samples), max_assets, max_views, 1, height, width), dtype=module.bool
    )
    view_mask = module.zeros((len(samples), max_assets, max_views), dtype=module.bool)
    asset_mask = module.zeros((len(samples), max_assets), dtype=module.bool)
    metadata: List[List[Optional[Dict[str, Any]]]] = [
        [None] * max_assets for _ in samples
    ]
    for batch_index, assets in enumerate(asset_lists):
        for asset_index, asset in enumerate(assets):
            n_views = int(asset["views"].shape[0])
            if tuple(asset["views"].shape[1:]) != (3, height, width):
                raise ValueError(f"{group} view shape changed within batch")
            if tuple(asset["pixel_mask"].shape) != (n_views, 1, height, width):
                raise ValueError(f"{group} pixel-mask shape changed within batch")
            if tuple(asset["view_mask"].shape) != (n_views,):
                raise ValueError(f"{group} view-mask shape changed within batch")
            views[batch_index, asset_index, :n_views] = asset["views"]
            pixel_mask[batch_index, asset_index, :n_views] = asset["pixel_mask"]
            view_mask[batch_index, asset_index, :n_views] = asset["view_mask"]
            asset_mask[batch_index, asset_index] = True
            metadata[batch_index][asset_index] = asset["metadata"]
    return {
        group: views,
        f"{group}_pixel_mask": pixel_mask,
        f"{group}_view_mask": view_mask,
        f"{group}_asset_mask": asset_mask,
        f"{group}_metadata": metadata,
    }


def task2_v2_collate(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Pad variable sequences/assets while preserving every explicit mask."""
    module = _require_torch()
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    batch: Dict[str, Any] = {
        "person_id": [str(sample["person_id"]) for sample in samples],
        "task": [str(sample["task"]) for sample in samples],
        "split": [str(sample["split"]) for sample in samples],
        "clinical_values": module.stack([sample["clinical_values"] for sample in samples]),
        "clinical_mask": module.stack([sample["clinical_mask"] for sample in samples]),
        "ecg_lead_mask": module.stack([sample["ecg_lead_mask"] for sample in samples]),
        "ecg_numeric": module.stack([sample["ecg_numeric"] for sample in samples]),
        "ecg_numeric_mask": module.stack([sample["ecg_numeric_mask"] for sample in samples]),
        "source_present": module.stack([sample["source_present"] for sample in samples]),
    }
    for key in (
        "environment_feature_names",
        "environment_canonical_feature_names",
        "environment_excluded_feature_names",
        "environment_feature_indices",
    ):
        expected = samples[0][key]
        if any(sample[key] != expected for sample in samples[1:]):
            raise ValueError(f"Batch {key} metadata disagrees across samples")
        batch[key] = list(expected)
    for flag in sorted(set(SOURCE_FLAG_BY_GROUP.values())):
        batch[flag] = module.stack([sample[flag] for sample in samples])
    batch.update(
        _pad_temporal(
            samples,
            "ecg_waveform",
            ("ecg_value_mask", "ecg_sample_mask"),
        )
    )
    batch.update(
        _pad_temporal(
            samples,
            "environment",
            ("environment_feature_mask", "environment_slot_mask"),
        )
    )
    batch.update(
        _pad_temporal(
            samples,
            "activity",
            ("activity_feature_mask", "activity_observed_mask", "activity_valid_mask"),
        )
    )
    batch.update(
        _pad_temporal(
            samples,
            "cgm",
            ("cgm_feature_mask", "cgm_observed_mask", "cgm_valid_mask"),
        )
    )
    for group in RETINAL_GROUPS:
        batch.update(_pad_retinal_group(samples, group))
    if all("y" in sample for sample in samples):
        batch["y"] = module.stack([sample["y"] for sample in samples])
    elif any("y" in sample for sample in samples):
        raise ValueError("A batch cannot mix labeled and unlabeled samples")
    if any("label_value" in sample for sample in samples):
        batch["label_value"] = [sample.get("label_value") for sample in samples]
    return batch


def move_task2_v2_batch_to_device(
    batch: Mapping[str, Any], device: Any, *, non_blocking: bool = False
) -> Dict[str, Any]:
    """Move tensor leaves to a device without touching metadata containers."""
    module = _require_torch()
    return {
        key: value.to(device, non_blocking=non_blocking) if module.is_tensor(value) else value
        for key, value in batch.items()
    }


__all__ = [
    "DEFAULT_RETINAL_ASSET_CAPS",
    "FrozenScalerGroup",
    "OT2V2Dataset",
    "PRIMARY_SOURCE_ORDER",
    "RETINAL_GROUPS",
    "SOURCE_FLAG_BY_GROUP",
    "move_task2_v2_batch_to_device",
    "task2_v2_collate",
]
