#!/usr/bin/env python3
"""Extract compact training-only retinal sidecars from canonical V2 artifacts.

The canonical FLIO artifacts intentionally preserve a lossless decay-cube
archive for provenance.  Because ``torch.load`` materializes every object in a
``.pt`` file, reading those artifacts inside every training epoch is extremely
expensive even when the model only consumes ``model_views``.  This one-time
adapter loads each selected canonical artifact once and writes an explicitly
whitelisted, float16 model-view sidecar.  Canonical artifacts are never
modified or deleted.

The output index remains compatible with ``build_multimodal_index.py``:
``artifact_config_hash`` continues to identify the upstream pixel
preprocessing contract, while ``model_view_config_hash`` separately identifies
this lossless-metadata/lower-precision-storage adapter.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import numbers
import os
import tempfile
import traceback
from collections import Counter
from collections.abc import Mapping as ABCMapping
from collections.abc import Sequence as ABCSequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

import numpy as np
import pandas as pd
import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - progress is optional
    tqdm = None

try:
    from .preprocessing_utils import canonical_json_hash
except ImportError:
    from preprocessing_utils import canonical_json_hash  # type: ignore


FORMAT_VERSION = "retinal-model-view-sidecar-v2.1"
METADATA_SERIALIZATION_VERSION = "strict-json-primitives-v1"
SUCCESS_STATUSES = {
    "processed",
    "cached",
    "skipped",
    "skipped_current",
    "processed_with_source_errors",
}
DEFAULT_CAPS = {
    "photography": 8,
    "oct": 4,
    "octa": 12,
    "flio": 4,
}
REQUIRED_INDEX_COLUMNS = {
    "status",
    "person_id",
    "asset_key",
    "artifact_path",
    "modality",
    "artifact_config_hash",
}
MODEL_KEYS = (
    "person_id",
    "asset_id",
    "asset_key",
    "product_kind",
    "model_views",
    "model_pixel_mask",
    "view_quality_mask",
    "model_view_names",
    "protocol_metadata",
    "spacing_metadata",
    "pixel_interpretation_audit",
    "interpretation_warnings",
    "asset_metadata",
    "artifact_config",
    "artifact_config_hash",
    "input_index_config_hash",
    "input_index_source_fingerprint",
    "input_row_fingerprint",
    "source_fingerprint",
    "source_files",
)


def _norm(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _safe_token(value: Any, fallback: str) -> str:
    text = _norm(value) or fallback
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in text)


def _file_signature(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _torch_load(path: Path) -> MutableMapping[str, Any]:
    """Load a trusted canonical artifact inside the one-time adapter only."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, MutableMapping):
        raise ValueError(f"Canonical retinal artifact is not a mapping: {path}")
    return payload


def _torch_load_sidecar(path: Path) -> MutableMapping[str, Any]:
    """Load a generated sidecar without permitting arbitrary pickle globals."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:  # pragma: no cover - production requires modern torch
        raise RuntimeError(
            "Installed PyTorch lacks weights_only=True; refusing an unsafe "
            "sidecar compatibility fallback"
        ) from exc
    if not isinstance(payload, MutableMapping):
        raise ValueError(f"Model-view sidecar is not a mapping: {path}")
    return payload


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        # A sidecar is not publishable until the same restricted loader used
        # by training can deserialize it.  Validate the temporary file before
        # replacing any prior good artifact.
        _torch_load_sidecar(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _conversion_type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _safe_metadata_key(value: Any, *, path: str) -> str:
    """Convert a metadata mapping key to one deterministic JSON string key."""
    if value is None:
        return "null"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, bool, numbers.Integral, numbers.Real, Decimal, Path)):
        return str(value)
    raise TypeError(
        f"Unsupported metadata mapping key at {path}: {_conversion_type_name(value)}"
    )


def _safe_metadata_value(
    value: Any,
    *,
    path: str,
    conversions: Counter[str],
) -> Any:
    """Recursively restrict retained metadata to weights-only-safe primitives.

    DICOM value representations such as ``pydicom.valuerep.IS`` are subclasses
    of builtin numeric/string types.  Returning such an object unchanged still
    records its Python class in the pickle stream, so every accepted scalar is
    explicitly cast to an exact builtin type.  Binary payloads and unknown
    objects fail closed rather than leaking raw data into a model sidecar.
    """

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        raise TypeError(f"Unexpected nested tensor in retained metadata at {path}")
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"Binary metadata is forbidden in model sidecars at {path}")
    if isinstance(value, np.ndarray):
        conversions[_conversion_type_name(value)] += 1
        return _safe_metadata_value(
            value.tolist(), path=path, conversions=conversions
        )
    if isinstance(value, np.generic):
        conversions[_conversion_type_name(value)] += 1
        return _safe_metadata_value(
            value.item(), path=path, conversions=conversions
        )
    if isinstance(value, bool):
        if type(value) is not bool:
            conversions[_conversion_type_name(value)] += 1
        return bool(value)
    if isinstance(value, numbers.Integral):
        if type(value) is not int:
            conversions[_conversion_type_name(value)] += 1
        return int(value)
    if isinstance(value, numbers.Real):
        if type(value) is not float:
            conversions[_conversion_type_name(value)] += 1
        number = float(value)
        if not math.isfinite(number):
            conversions["nonfinite-real-to-null"] += 1
            return None
        return number
    if isinstance(value, str):
        if type(value) is not str:
            conversions[_conversion_type_name(value)] += 1
        return str(value)
    if isinstance(value, (Decimal, Path, dt.date, dt.time, dt.datetime)):
        conversions[_conversion_type_name(value)] += 1
        return str(value)
    if isinstance(value, ABCMapping):
        output: Dict[str, Any] = {}
        for raw_key, item in value.items():
            key = _safe_metadata_key(raw_key, path=f"{path}.<key>")
            if key in output:
                raise ValueError(
                    f"Metadata key collision after string conversion at {path}: {key!r}"
                )
            output[key] = _safe_metadata_value(
                item, path=f"{path}.{key}", conversions=conversions
            )
        return output
    if isinstance(value, ABCSequence):
        return [
            _safe_metadata_value(
                item, path=f"{path}[{index}]", conversions=conversions
            )
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"Unsupported retained metadata object at {path}: "
        f"{_conversion_type_name(value)}"
    )


def _resolve_artifact_path(value: Any, index_path: Path) -> Path:
    text = _norm(value)
    if not text:
        raise ValueError("retinal index contains an empty artifact_path")
    raw = Path(text)
    if raw.is_absolute():
        return raw.resolve()
    candidates = ((index_path.parent / raw).resolve(), raw.resolve())
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _sidecar_is_current(
    path: Path,
    *,
    model_view_config_hash: str,
    canonical_signature: Mapping[str, Any],
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _torch_load_sidecar(path)
    except Exception:
        return False
    return bool(
        payload.get("model_view_format_version") == FORMAT_VERSION
        and payload.get("model_view_config_hash") == model_view_config_hash
        and payload.get("canonical_artifact_file") == dict(canonical_signature)
        and "model_views" in payload
        and not any("archive" in str(key).lower() for key in payload)
    )


def _validate_and_compact(
    artifact: Mapping[str, Any],
    *,
    expected_person_id: str,
    expected_modality: str,
    expected_artifact_config_hash: str,
    canonical_signature: Mapping[str, Any],
    model_view_config: Mapping[str, Any],
    model_view_config_hash: str,
) -> Dict[str, Any]:
    person_id = _norm(artifact.get("person_id"))
    if person_id != expected_person_id:
        raise ValueError(
            f"person_id mismatch: index={expected_person_id!r}, artifact={person_id!r}"
        )
    observed_hash = _norm(artifact.get("artifact_config_hash"))
    if observed_hash != expected_artifact_config_hash:
        raise ValueError(
            "artifact_config_hash mismatch: "
            f"index={expected_artifact_config_hash!r}, artifact={observed_hash!r}"
        )
    protocol = artifact.get("protocol_metadata")
    if not isinstance(protocol, Mapping):
        raise ValueError("canonical artifact lacks protocol_metadata")
    modality = _norm(protocol.get("modality")).lower()
    if modality != expected_modality:
        raise ValueError(
            f"modality mismatch: index={expected_modality!r}, artifact={modality!r}"
        )

    views = artifact.get("model_views")
    pixel_mask = artifact.get("model_pixel_mask")
    quality_mask = artifact.get("view_quality_mask")
    if not isinstance(views, torch.Tensor) or views.ndim != 4 or views.shape[1] != 3:
        raise ValueError("model_views must be Tensor[n_views,3,H,W]")
    if not isinstance(pixel_mask, torch.Tensor) or tuple(pixel_mask.shape) != (
        int(views.shape[0]), 1, int(views.shape[2]), int(views.shape[3])
    ):
        raise ValueError("model_pixel_mask shape does not match model_views")
    if not isinstance(quality_mask, torch.Tensor) or tuple(quality_mask.shape) != (
        int(views.shape[0]),
    ):
        raise ValueError("view_quality_mask shape does not match model_views")
    if not bool(torch.isfinite(views).all().item()):
        raise ValueError("model_views contains non-finite values")
    if views.numel() and bool(((views < 0) | (views > 1)).any().item()):
        raise ValueError("model_views must lie in [0,1]")

    compact = {key: artifact[key] for key in MODEL_KEYS if key in artifact}
    compact["model_views"] = views.detach().cpu().to(dtype=torch.float16).contiguous()
    compact["model_pixel_mask"] = pixel_mask.detach().cpu().bool().contiguous()
    compact["view_quality_mask"] = quality_mask.detach().cpu().bool().contiguous()
    compact.update(
        {
            "model_view_format_version": FORMAT_VERSION,
            "model_view_config": dict(model_view_config),
            "model_view_config_hash": model_view_config_hash,
            "canonical_artifact_file": dict(canonical_signature),
            "canonical_model_views_dtype": str(views.dtype),
            "model_view_storage_dtype": "float16",
            "model_view_quantization_note": (
                "IEEE float16 storage only; dataset converts to float32 before model input"
            ),
        }
    )
    tensor_keys = {"model_views", "model_pixel_mask", "view_quality_mask"}
    conversions: Counter[str] = Counter()
    safe_compact: Dict[str, Any] = {}
    for key, value in compact.items():
        if key in tensor_keys:
            safe_compact[key] = value
        else:
            safe_compact[key] = _safe_metadata_value(
                value, path=key, conversions=conversions
            )
    safe_compact["metadata_sanitization"] = {
        "format_version": METADATA_SERIALIZATION_VERSION,
        "converted_type_counts": dict(sorted(conversions.items())),
        "unknown_object_policy": "fail_closed",
        "binary_metadata_policy": "forbidden",
        "published_sidecar_validation": "torch.load(weights_only=True)",
    }
    compact = safe_compact
    forbidden = [key for key in compact if "archive" in str(key).lower()]
    if forbidden:
        raise AssertionError(f"archive keys escaped whitelist: {forbidden}")
    return compact


def _process_one(
    record: Mapping[str, Any],
    *,
    index_path_text: str,
    output_root_text: str,
    model_view_config: Mapping[str, Any],
    model_view_config_hash: str,
    cache_policy: str,
) -> Dict[str, Any]:
    index_path = Path(index_path_text)
    person_id = _norm(record.get("person_id"))
    modality = _norm(record.get("modality")).lower()
    asset_key = _norm(record.get("asset_key"))
    canonical_path = _resolve_artifact_path(record.get("artifact_path"), index_path)
    output_path = (
        Path(output_root_text)
        / _safe_token(modality, "unknown")
        / f"{_safe_token(asset_key, 'asset')}.pt"
    )
    try:
        if not canonical_path.is_file():
            raise FileNotFoundError(f"canonical artifact does not exist: {canonical_path}")
        signature = _file_signature(canonical_path)
        if cache_policy == "reuse" and _sidecar_is_current(
            output_path,
            model_view_config_hash=model_view_config_hash,
            canonical_signature=signature,
        ):
            status = "cached"
        else:
            artifact = _torch_load(canonical_path)
            compact = _validate_and_compact(
                artifact,
                expected_person_id=person_id,
                expected_modality=modality,
                expected_artifact_config_hash=_norm(record.get("artifact_config_hash")),
                canonical_signature=signature,
                model_view_config=model_view_config,
                model_view_config_hash=model_view_config_hash,
            )
            _atomic_torch_save(compact, output_path)
            status = "processed"
        result = dict(record)
        result.update(
            {
                "status": status,
                "artifact_path": str(output_path.resolve()),
                "canonical_artifact_path": str(canonical_path),
                "canonical_artifact_size_bytes": int(signature["size_bytes"]),
                "model_view_config_hash": model_view_config_hash,
                "model_view_sidecar_size_bytes": int(output_path.stat().st_size),
            }
        )
        return result
    except Exception as exc:
        return {
            "status": "failed",
            "person_id": person_id,
            "asset_key": asset_key,
            "modality": modality,
            "canonical_artifact_path": str(canonical_path),
            "failure_stage": "sidecar_extraction",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }


def _select_with_caps(frame: pd.DataFrame, caps: Mapping[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sort_columns = [
        column
        for column in (
            "person_id",
            "modality",
            "protocol",
            "laterality",
            "slab_or_wavelength",
            "asset_key",
            "artifact_path",
        )
        if column in frame
    ]
    ordered = frame.sort_values(sort_columns, kind="mergesort").copy()
    ordered["model_view_selection_rank"] = (
        ordered.groupby(["person_id", "modality"], sort=False).cumcount() + 1
    )
    cap_series = ordered["modality"].map(caps)
    selected_mask = ordered["model_view_selection_rank"].le(cap_series)
    return ordered.loc[selected_mask].copy(), ordered.loc[~selected_mask].copy()


def _parse_caps(args: argparse.Namespace) -> Dict[str, int]:
    caps = {
        "photography": int(args.max_photography_assets),
        "oct": int(args.max_oct_assets),
        "octa": int(args.max_octa_assets),
        "flio": int(args.max_flio_assets),
    }
    if any(value <= 0 for value in caps.values()):
        raise ValueError("all retinal asset caps must be positive")
    return caps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retinal-index", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cache-policy", choices=("reuse", "refresh"), default="reuse")
    parser.add_argument("--max-photography-assets", type=int, default=DEFAULT_CAPS["photography"])
    parser.add_argument("--max-oct-assets", type=int, default=DEFAULT_CAPS["oct"])
    parser.add_argument("--max-octa-assets", type=int, default=DEFAULT_CAPS["octa"])
    parser.add_argument("--max-flio-assets", type=int, default=DEFAULT_CAPS["flio"])
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if args.num_workers <= 0:
        raise ValueError("num-workers must be positive")

    index_path = Path(args.retinal_index).resolve()
    output_dir = Path(args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(index_path, dtype={"person_id": "string"}, keep_default_na=False)
    missing = sorted(REQUIRED_INDEX_COLUMNS - set(frame))
    if missing:
        raise ValueError(f"retinal index is missing columns: {missing}")
    status = frame["status"].astype(str).str.strip().str.lower()
    unknown_statuses = sorted(set(status) - SUCCESS_STATUSES - {"failed", "error"})
    if unknown_statuses:
        raise ValueError(f"retinal index contains unknown statuses: {unknown_statuses}")
    frame = frame.loc[status.isin(SUCCESS_STATUSES)].copy()
    frame["person_id"] = frame["person_id"].astype(str).str.strip()
    frame["modality"] = frame["modality"].astype(str).str.strip().str.lower()
    if frame["person_id"].eq("").any() or frame["asset_key"].astype(str).str.strip().eq("").any():
        raise ValueError("successful retinal rows require person_id and asset_key")
    unknown_modalities = sorted(set(frame["modality"]) - set(DEFAULT_CAPS))
    if unknown_modalities:
        raise ValueError(f"retinal index contains unknown modalities: {unknown_modalities}")
    if frame["asset_key"].duplicated().any():
        raise ValueError("retinal index contains duplicate asset_key values")
    hashes = sorted(set(frame["artifact_config_hash"].astype(str).str.strip()))
    if len(hashes) != 1 or not hashes[0]:
        raise ValueError(f"retinal index must contain one artifact_config_hash: {hashes}")

    caps = _parse_caps(args)
    selected, omitted = _select_with_caps(frame, caps)
    model_view_config = {
        "model_view_format_version": FORMAT_VERSION,
        "builder": "build_retinal_model_view_sidecars.py",
        "upstream_artifact_config_hash": hashes[0],
        "storage_dtype": "float16",
        "metadata_serialization": METADATA_SERIALIZATION_VERSION,
        "retained_keys": list(MODEL_KEYS),
        "forbidden_payload": "all unlisted keys, especially lossless archives and raw pixel arrays",
        "selection_caps_per_person_modality": caps,
        "selection_order": [
            "person_id", "modality", "protocol", "laterality",
            "slab_or_wavelength", "asset_key", "artifact_path",
        ],
    }
    model_view_config_hash = canonical_json_hash(model_view_config)
    artifact_root = output_dir / "retinal_assets" / model_view_config_hash[:16]
    records = selected.to_dict(orient="records")
    kwargs = {
        "index_path_text": str(index_path),
        "output_root_text": str(artifact_root),
        "model_view_config": model_view_config,
        "model_view_config_hash": model_view_config_hash,
        "cache_policy": args.cache_policy,
    }
    results = []
    if args.num_workers == 1:
        iterator = records
        if tqdm is not None and not args.no_progress:
            iterator = tqdm(iterator, total=len(records), desc="Retinal model sidecars", unit="asset")
        for record in iterator:
            results.append(_process_one(record, **kwargs))
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(_process_one, record, **kwargs): record for record in records}
            iterator = as_completed(futures)
            if tqdm is not None and not args.no_progress:
                iterator = tqdm(iterator, total=len(futures), desc="Retinal model sidecars", unit="asset")
            for future in iterator:
                try:
                    results.append(future.result())
                except Exception as exc:  # pragma: no cover - worker boundary safety
                    record = futures[future]
                    results.append(
                        {
                            "status": "failed",
                            "person_id": _norm(record.get("person_id")),
                            "asset_key": _norm(record.get("asset_key")),
                            "modality": _norm(record.get("modality")),
                            "failure_stage": "worker_boundary",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )

    failures = [item for item in results if item.get("status") == "failed"]
    successes = [item for item in results if item.get("status") in {"processed", "cached"}]
    successes.sort(key=lambda item: (_norm(item.get("person_id")), _norm(item.get("modality")), _norm(item.get("asset_key"))))
    index = pd.DataFrame(successes)
    if not index.empty:
        index = index[[column for column in selected.columns if column in index] + [
            column for column in (
                "canonical_artifact_path",
                "canonical_artifact_size_bytes",
                "model_view_config_hash",
                "model_view_sidecar_size_bytes",
            ) if column in index
        ]]
    omitted_audit = omitted[[
        column for column in (
            "person_id", "modality", "protocol", "laterality",
            "slab_or_wavelength", "asset_key", "artifact_path",
            "model_view_selection_rank",
        ) if column in omitted
    ]].copy()
    _atomic_text(index.to_csv(index=False), output_dir / "retinal_model_view_index.csv")
    _atomic_text(omitted_audit.to_csv(index=False), output_dir / "retinal_model_view_omitted_by_cap.csv")
    _atomic_text(json.dumps(failures, indent=2, sort_keys=True), output_dir / "retinal_model_view_failures.json")

    canonical_bytes = int(sum(int(item.get("canonical_artifact_size_bytes", 0)) for item in successes))
    sidecar_bytes = int(sum(int(item.get("model_view_sidecar_size_bytes", 0)) for item in successes))
    summary = {
        "format_version": FORMAT_VERSION,
        "retinal_index": str(index_path),
        "out_dir": str(output_dir),
        "model_view_config": model_view_config,
        "model_view_config_hash": model_view_config_hash,
        "upstream_artifact_config_hash": hashes[0],
        "n_successful_input_rows": int(len(frame)),
        "n_selected": int(len(selected)),
        "n_omitted_by_cap": int(len(omitted)),
        "n_processed": int(sum(item.get("status") == "processed" for item in results)),
        "n_cached": int(sum(item.get("status") == "cached" for item in results)),
        "n_failed": int(len(failures)),
        "selected_canonical_bytes": canonical_bytes,
        "sidecar_bytes": sidecar_bytes,
        "size_ratio_sidecar_to_canonical": (
            float(sidecar_bytes / canonical_bytes) if canonical_bytes else None
        ),
        "canonical_artifacts_modified": False,
    }
    _atomic_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        output_dir / "retinal_model_view_summary.json",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        print("first failures:", failures[:5])


if __name__ == "__main__":
    main()
