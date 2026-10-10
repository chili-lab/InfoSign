#!/usr/bin/env python3
"""Preprocess metadata-rich AI-READI retinal assets into auditable V2 artifacts.

This stage consumes either ``retinal_asset_index.csv`` or
``retinal_default_model_view.csv`` from ``build_retinal_asset_index.py``.
It deliberately keeps the long-asset semantics of that index: photography
protocols, OCT/OCTA volumes, OCTA en-face slabs, and FLIO wavelengths remain
separate assets instead of being silently averaged into participant slots.

Important scientific contracts
------------------------------

* DICOM grayscale pixels pass through the modality/rescale transform, optional
  VOI transform, and MONOCHROME1 presentation inversion in that order.
* RGB photography uses one shared joint upper-percentile scale for all three
  channels.  It never subtracts per-channel offsets or stretches channels
  independently, so positive unsaturated RGB ratios are retained.
* Every 2-D model view is resized with its aspect ratio intact and padded.  The
  original geometry, resized geometry, padding, and valid-pixel mask are saved.
* OCT and OCTA flow cubes produce a configurable stack of central slices (and a
  three-slice 2.5-D view) rather than one middle slice.  OCTA en-face assets are
  processed independently as 2-D products.
* FLIO artifacts preserve the full decoded short/long-wavelength decay cube for
  each indexed wavelength by default.  Intensity and photon-weighted mean-bin
  auxiliary images are derived before display transforms.  Mean-bin is a frame
  index, not an IRF-corrected or physical lifetime.
  A typical 1024x256x256 uint16 cube is about 128 MiB before compression, so a
  full-cohort cache can approach one TiB.  Lossless compression is data
  dependent; pilot ``retinal_default_model_view.csv`` with ``--limit`` first.
* There is no participant-specific z-score.  All display normalization choices
  are recorded and hashed.  Downstream learned normalization must be fit on the
  training split only.

Outputs under ``--out-dir``
---------------------------

``retinal_assets/<config-prefix>/<modality>/*.pt``
    One config- and source-aware artifact per input row.
``retinal_preprocessed_index.csv``
    Successful/cached assets and compact interpretation/QC audit fields.
``retinal_preprocess_failures.json``
    Explicit path, dependency, decode, dimension, interpretation, and save
    failures.
``retinal_preprocess_schema.json``
    Tensor, mask, normalization, FLIO, and run-count contract.
``artifact_config.json``
    Canonical preprocessing configuration and SHA-256 hash.

PyTorch and pydicom are imported lazily so importing the module for manifest or
geometry tests does not require the imaging dependencies.  Running the
preprocessor gives an actionable dependency error when either is unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from PIL import Image, __version__ as PILLOW_VERSION
except Exception:  # pragma: no cover - dependency varies by runtime
    Image = None  # type: ignore[assignment]
    PILLOW_VERSION = "missing"

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - progress is optional
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


REQUIRED_COLUMNS = {
    "person_id",
    "modality",
    "protocol",
    "laterality",
    "slab_or_wavelength",
    "filepath",
}
MISSING_TOKENS = {"", "nan", "none", "null", "not reported", "not_provided"}
MONOCHROME_PHOTOMETRICS = {"MONOCHROME1", "MONOCHROME2"}
COLOR_PHOTOMETRIC_PREFIXES = ("RGB", "YBR", "PALETTE", "HSV", "ARGB", "CMYK")
INDEX_COLUMNS = [
    "status", "person_id", "asset_id", "asset_key", "artifact_path", "modality", "protocol",
    "laterality", "slab_or_wavelength", "filepath", "split", "clinical_site",
    "product_kind", "decoded_shape", "model_views_shape", "n_model_views",
    "n_quality_views", "all_views_quality_pass", "monochrome1_inverted",
    "modality_transform_applied", "voi_transform_applied",
    "n_interpretation_warnings", "interpretation_warnings",
    "input_index_config_hash", "input_index_source_fingerprint",
    "artifact_config_hash", "source_fingerprint", "artifact_size_bytes",
]


class RetinalPreprocessError(RuntimeError):
    """A staged error that can carry partial pixel-interpretation audit data."""

    def __init__(self, stage: str, message: str, audit: Optional[Mapping[str, Any]] = None):
        super().__init__(f"{stage}: {message}")
        self.stage = stage
        self.audit = dict(audit or {})


@dataclass(frozen=True)
class ImageQCThresholds:
    """Conservative checks; failed views remain in the artifact but are masked."""

    min_finite_fraction: float = 0.98
    min_robust_range: float = 1e-8
    max_low_saturation_fraction: float = 0.98
    max_high_saturation_fraction: float = 0.98

    def __post_init__(self) -> None:
        for name in (
            "min_finite_fraction",
            "max_low_saturation_fraction",
            "max_high_saturation_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.min_robust_range < 0:
            raise ValueError("min_robust_range must be non-negative")


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_json_scalar(v) for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_scalar(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_scalar(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(_json_scalar(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _require_pillow() -> Any:
    if Image is None:
        raise ImportError(
            "Pillow is required for aspect-preserving retinal resize/pad. "
            "Install pillow in the preprocessing environment."
        )
    return Image


def _import_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency varies by runtime
        raise ImportError(
            "PyTorch is required to write/read V2 retinal .pt artifacts. "
            "Install a compatible torch build before running this script."
        ) from exc
    return torch


def _import_pydicom() -> Tuple[Any, Any, Any]:
    try:
        import pydicom  # type: ignore
        try:
            from pydicom.pixels import apply_modality_lut, apply_voi_lut  # type: ignore
        except ImportError:  # pydicom < 3
            from pydicom.pixel_data_handlers.util import (  # type: ignore
                apply_modality_lut,
                apply_voi_lut,
            )
    except Exception as exc:  # pragma: no cover - dependency varies by runtime
        raise ImportError(
            "pydicom is required to decode AI-READI retinal DICOM files. "
            "Compressed transfer syntaxes may additionally require pylibjpeg, "
            "pylibjpeg-libjpeg, pylibjpeg-openjpeg, or GDCM."
        ) from exc
    return pydicom, apply_modality_lut, apply_voi_lut


def _runtime_versions() -> Dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": str(np.__version__),
        "pandas": str(pd.__version__),
        "pillow": str(PILLOW_VERSION),
    }
    try:
        torch = _import_torch()
        versions["torch"] = str(getattr(torch, "__version__", "unknown"))
    except ImportError:
        versions["torch"] = "missing"
    try:
        pydicom, _, _ = _import_pydicom()
        versions["pydicom"] = str(getattr(pydicom, "__version__", "unknown"))
    except ImportError:
        versions["pydicom"] = "missing"
    return versions


def _valid_text(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() not in MISSING_TOKENS


def _norm(value: Any) -> str:
    return str(value).strip() if _valid_text(value) else ""


def _safe_token(value: Any, fallback: str = "unknown", max_length: int = 48) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", _norm(value)).strip("_.-")
    return (token or fallback)[:max_length]


def _canonical_laterality(value: Any) -> str:
    text = _norm(value).upper()
    return {
        "L": "L", "LEFT": "L", "OS": "L",
        "R": "R", "RIGHT": "R", "OD": "R",
        "B": "B", "BILATERAL": "B", "BOTH": "B",
    }.get(text, text)


def resolve_path(dataset_root: Path, value: Any) -> Path:
    """Resolve absolute-within-dataset manifest paths without aliasing missing paths."""
    text = _norm(value)
    if not text:
        return dataset_root / "__missing_retinal_manifest_path__"
    path = Path(text)
    if path.is_absolute():
        if path.exists():
            return path
        return dataset_root / text.lstrip("/\\")
    return dataset_root / path


def _dicom_value(value: Any) -> Any:
    """Convert a pydicom scalar/MultiValue to strict JSON-compatible metadata."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)):
        return _json_scalar(value)
    try:
        return [_dicom_value(item) for item in value]
    except TypeError:
        return str(value)


def _finite_int(value: Any) -> Optional[int]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return int(round(numeric))


def _geometry_consistency_audit(
    row: Mapping[str, Any], dicom: Mapping[str, Any]
) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    comparisons: Dict[str, Any] = {}
    fields = (
        ("height", "rows"),
        ("width", "columns"),
        ("number_of_frames", "number_of_frames"),
        ("color_channels", "samples_per_pixel"),
    )
    for manifest_key, dicom_key in fields:
        manifest_value = _finite_int(row.get(manifest_key))
        dicom_value = _finite_int(dicom.get(dicom_key))
        matches = (
            None if manifest_value is None or dicom_value is None
            else manifest_value == dicom_value
        )
        comparisons[manifest_key] = {
            "manifest": manifest_value,
            "dicom": dicom_value,
            "matches": matches,
        }
        if matches is False:
            warnings.append(
                f"Manifest {manifest_key}={manifest_value} disagrees with DICOM {dicom_key}={dicom_value}"
            )
    return comparisons, warnings


def _source_fingerprint(files: Iterable[Mapping[str, Any]], row_fingerprint: str) -> str:
    return canonical_json_hash(
        {"source_files": list(files), "input_row_fingerprint": row_fingerprint}
    )


def _row_fingerprint(row: Mapping[str, Any]) -> str:
    excluded = {"asset_key", "input_row_index", "duplicate_occurrence"}
    payload = {
        str(key): _json_scalar(value)
        for key, value in row.items()
        if str(key) not in excluded
    }
    return canonical_json_hash(payload)


def _asset_identity(row: Mapping[str, Any]) -> str:
    fingerprint = _row_fingerprint(row)
    prefix = "_".join(
        [
            _safe_token(row.get("person_id"), "person"),
            _safe_token(row.get("modality"), "retinal"),
            _safe_token(row.get("protocol"), "unknown"),
            _safe_token(row.get("laterality"), "U", max_length=8),
            _safe_token(row.get("slab_or_wavelength"), "na", max_length=32),
        ]
    )
    occurrence = int(row.get("duplicate_occurrence", 0))
    return f"{prefix}_{fingerprint[:16]}_{occurrence:02d}"


def prepare_asset_index(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Retinal asset index missing required columns: {missing}")
    out = frame.copy()
    out["input_row_index"] = np.arange(len(out), dtype=np.int64)
    out["person_id"] = out["person_id"].map(as_person_id)
    if (out["person_id"].str.len() == 0).any():
        bad = out.loc[out["person_id"].str.len() == 0, "input_row_index"].tolist()
        raise ValueError(f"Retinal asset index has empty person_id at rows {bad[:20]}")
    if (~out["filepath"].map(_valid_text)).any():
        bad = out.loc[~out["filepath"].map(_valid_text), "input_row_index"].tolist()
        raise ValueError(f"Retinal asset index has empty filepath at rows {bad[:20]}")

    if "config_hash" in out.columns:
        out["input_index_config_hash"] = out["config_hash"].map(_norm)
        if (~out["input_index_config_hash"].map(_valid_text)).any():
            bad = out.loc[
                ~out["input_index_config_hash"].map(_valid_text), "input_row_index"
            ].tolist()
            raise ValueError(f"Retinal asset index has empty config_hash at rows {bad[:20]}")
        distinct_configs = sorted(out["input_index_config_hash"].unique())
        if len(distinct_configs) != 1:
            raise ValueError(
                "One preprocessing run cannot mix retinal index configurations: "
                f"{distinct_configs[:10]}"
            )
    else:
        # Backward-compatible with the first V2 draft while making the missing
        # upstream provenance explicit in every output row/artifact.
        out["input_index_config_hash"] = ""
    if "source_fingerprint" in out.columns:
        out["input_index_source_fingerprint"] = out["source_fingerprint"].map(_norm)
        if (~out["input_index_source_fingerprint"].map(_valid_text)).any():
            bad = out.loc[
                ~out["input_index_source_fingerprint"].map(_valid_text), "input_row_index"
            ].tolist()
            raise ValueError(f"Retinal asset index has empty source_fingerprint at rows {bad[:20]}")
        distinct_source_fingerprints = sorted(
            out["input_index_source_fingerprint"].unique()
        )
        if len(distinct_source_fingerprints) != 1:
            raise ValueError(
                "One preprocessing run cannot mix retinal index source fingerprints: "
                f"{distinct_source_fingerprints[:10]}"
            )
    else:
        out["input_index_source_fingerprint"] = ""

    identity_columns = [
        "person_id", "modality", "protocol", "laterality", "slab_or_wavelength",
        "filepath", "sop_instance_uid", "source_manifest_row", "source_variant",
    ]
    identity_columns = [column for column in identity_columns if column in out.columns]
    out["duplicate_occurrence"] = out.groupby(
        identity_columns, dropna=False, sort=False
    ).cumcount()
    records = out.to_dict("records")
    out["input_row_fingerprint"] = [_row_fingerprint(row) for row in records]
    if "asset_id" in out.columns:
        if (~out["asset_id"].map(_valid_text)).any():
            bad = out.loc[~out["asset_id"].map(_valid_text), "input_row_index"].tolist()
            raise ValueError(f"Retinal asset index has empty asset_id at rows {bad[:20]}")
        out["asset_id"] = out["asset_id"].map(_norm)
        if out["asset_id"].duplicated().any():
            duplicates = out.loc[out["asset_id"].duplicated(False), "asset_id"].tolist()
            raise ValueError(f"Retinal asset index has duplicate asset_id values: {duplicates[:20]}")
        out["asset_key"] = out["asset_id"].map(
            lambda value: _safe_token(value, "asset", max_length=96)
        )
    else:
        out["asset_key"] = [_asset_identity(row) for row in records]
        out["asset_id"] = out["asset_key"]
    if out["asset_key"].duplicated().any():
        duplicates = out.loc[out["asset_key"].duplicated(False), "asset_key"].tolist()
        raise ValueError(f"Internal asset-key collision: {duplicates[:10]}")
    return out


def estimate_flio_uint16_uncompressed_bytes(assets: pd.DataFrame) -> int:
    """Estimate declared FLIO cube bytes assuming the usual uint16 storage."""
    if "modality" not in assets.columns:
        return 0
    flio = assets[assets["modality"].astype(str).str.lower().eq("flio")]

    def numeric(column: str) -> pd.Series:
        if column not in flio.columns:
            return pd.Series(0.0, index=flio.index, dtype=float)
        return pd.to_numeric(flio[column], errors="coerce").fillna(0.0)

    voxels = (numeric("number_of_frames") * numeric("height") * numeric("width")).sum()
    return int(voxels * 2)


def _tag_present(ds: Any, keyword: str) -> bool:
    try:
        return keyword in ds and getattr(ds, keyword, None) is not None
    except Exception:
        return getattr(ds, keyword, None) is not None


def _padding_mask(raw: np.ndarray, ds: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return spatial/frame validity based on DICOM Pixel Padding tags."""
    try:
        samples_per_pixel = int(getattr(ds, "SamplesPerPixel", 1) or 1)
    except (TypeError, ValueError):
        samples_per_pixel = 1
    is_color = samples_per_pixel > 1 and raw.ndim >= 3 and raw.shape[-1] in (3, 4)
    shape = raw.shape[:-1] if is_color else raw.shape
    valid = np.ones(shape, dtype=bool)
    padding = getattr(ds, "PixelPaddingValue", None)
    limit = getattr(ds, "PixelPaddingRangeLimit", None)
    audit: Dict[str, Any] = {
        "pixel_padding_value": _dicom_value(padding),
        "pixel_padding_range_limit": _dicom_value(limit),
        "padding_pixels_masked": 0,
    }
    if padding is None:
        return valid, audit
    try:
        p0 = float(padding)
        if is_color:
            base = np.all(np.asarray(raw[..., :3], dtype=np.float64) == p0, axis=-1)
        else:
            numeric = np.asarray(raw, dtype=np.float64)
            if limit is None:
                base = numeric == p0
            else:
                p1 = float(limit)
                base = (numeric >= min(p0, p1)) & (numeric <= max(p0, p1))
        valid &= ~base
        audit["padding_pixels_masked"] = int(np.count_nonzero(base))
    except Exception as exc:
        audit["padding_mask_warning"] = f"Could not interpret PixelPaddingValue: {exc}"
    return valid, audit


def _functional_group_pixel_measures(ds: Any) -> Dict[str, Any]:
    """Retain enhanced-DICOM spacing without copying large functional groups."""

    def extract(group: Any) -> Optional[Dict[str, Any]]:
        sequence = getattr(group, "PixelMeasuresSequence", None)
        if not sequence:
            return None
        measures = sequence[0]
        return {
            "pixel_spacing": _dicom_value(getattr(measures, "PixelSpacing", None)),
            "slice_thickness": _dicom_value(getattr(measures, "SliceThickness", None)),
            "spacing_between_slices": _dicom_value(
                getattr(measures, "SpacingBetweenSlices", None)
            ),
        }

    shared: Optional[Dict[str, Any]] = None
    shared_sequence = getattr(ds, "SharedFunctionalGroupsSequence", None)
    if shared_sequence:
        shared = extract(shared_sequence[0])

    per_frame_sequence = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    unique: Dict[str, Dict[str, Any]] = {}
    n_with_measures = 0
    if per_frame_sequence:
        for group in per_frame_sequence:
            measures = extract(group)
            if measures is None:
                continue
            n_with_measures += 1
            key = json.dumps(_json_scalar(measures), sort_keys=True, separators=(",", ":"))
            unique[key] = measures
    return {
        "shared_functional_group_pixel_measures": shared,
        "per_frame_groups_total": int(len(per_frame_sequence or [])),
        "per_frame_groups_with_pixel_measures": int(n_with_measures),
        "per_frame_pixel_measures_unique": list(unique.values()),
        "per_frame_spacing_varies": len(unique) > 1,
    }


def read_dicom(path: Path) -> Tuple[Any, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Read decoded stored values, validity, and non-pixel DICOM metadata."""
    pydicom, _, _ = _import_pydicom()
    try:
        ds = pydicom.dcmread(str(path))
    except Exception as exc:
        raise RetinalPreprocessError("dicom_read", f"Cannot read {path}: {exc}") from exc

    metadata: Dict[str, Any] = {
        "transfer_syntax_uid": str(
            getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", "")
        ),
        "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "")),
        "sop_class_uid": str(getattr(ds, "SOPClassUID", "")),
        "photometric_interpretation": str(
            getattr(ds, "PhotometricInterpretation", "")
        ).upper(),
        "samples_per_pixel": _dicom_value(getattr(ds, "SamplesPerPixel", None)),
        "planar_configuration": _dicom_value(getattr(ds, "PlanarConfiguration", None)),
        "rows": _dicom_value(getattr(ds, "Rows", None)),
        "columns": _dicom_value(getattr(ds, "Columns", None)),
        "number_of_frames": _dicom_value(getattr(ds, "NumberOfFrames", None)),
        "bits_allocated": _dicom_value(getattr(ds, "BitsAllocated", None)),
        "bits_stored": _dicom_value(getattr(ds, "BitsStored", None)),
        "high_bit": _dicom_value(getattr(ds, "HighBit", None)),
        "pixel_representation": _dicom_value(getattr(ds, "PixelRepresentation", None)),
        "rescale_slope": _dicom_value(getattr(ds, "RescaleSlope", None)),
        "rescale_intercept": _dicom_value(getattr(ds, "RescaleIntercept", None)),
        "rescale_type": _dicom_value(getattr(ds, "RescaleType", None)),
        "window_center": _dicom_value(getattr(ds, "WindowCenter", None)),
        "window_width": _dicom_value(getattr(ds, "WindowWidth", None)),
        "voi_lut_function": _dicom_value(getattr(ds, "VOILUTFunction", None)),
        "presentation_lut_shape": _dicom_value(getattr(ds, "PresentationLUTShape", None)),
        "pixel_spacing": _dicom_value(getattr(ds, "PixelSpacing", None)),
        "imager_pixel_spacing": _dicom_value(getattr(ds, "ImagerPixelSpacing", None)),
        "slice_thickness": _dicom_value(getattr(ds, "SliceThickness", None)),
        "spacing_between_slices": _dicom_value(getattr(ds, "SpacingBetweenSlices", None)),
        "frame_time": _dicom_value(getattr(ds, "FrameTime", None)),
        "frame_time_vector": _dicom_value(getattr(ds, "FrameTimeVector", None)),
        "content_date": _dicom_value(getattr(ds, "ContentDate", None)),
        "content_time": _dicom_value(getattr(ds, "ContentTime", None)),
        "image_laterality": _dicom_value(getattr(ds, "ImageLaterality", None)),
        "laterality": _dicom_value(getattr(ds, "Laterality", None)),
        "manufacturer": _dicom_value(getattr(ds, "Manufacturer", None)),
        "manufacturer_model_name": _dicom_value(
            getattr(ds, "ManufacturerModelName", None)
        ),
        "series_description": _dicom_value(getattr(ds, "SeriesDescription", None)),
        "protocol_name": _dicom_value(getattr(ds, "ProtocolName", None)),
        **_functional_group_pixel_measures(ds),
    }
    try:
        # pydicom's default decoded representation includes its standard color
        # conversion for YBR transfer syntaxes.  Do not convert a second time.
        raw = np.asarray(ds.pixel_array)
    except Exception as exc:
        message = (
            f"Cannot decode PixelData in {path}: {exc}. If the transfer syntax is "
            "compressed, install the appropriate pylibjpeg/OpenJPEG or GDCM plugin."
        )
        raise RetinalPreprocessError("pixel_decode", message, metadata) from exc
    if raw.size == 0:
        raise RetinalPreprocessError("pixel_decode", f"Empty pixel array in {path}", metadata)
    valid, padding_audit = _padding_mask(raw, ds)
    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    if photometric == "RGB":
        decoded_color_space = "RGB"
    elif photometric.startswith("YBR"):
        decoded_color_space = "RGB (pydicom default YBR conversion)"
    elif photometric == "PALETTE COLOR":
        decoded_color_space = "palette indices; color LUT is applied in interpretation stage"
    elif photometric in MONOCHROME_PHOTOMETRICS:
        decoded_color_space = "grayscale"
    else:
        decoded_color_space = "uncertain"
    metadata.update(
        {
            "decoded_shape": list(raw.shape),
            "decoded_dtype": str(raw.dtype),
            "decoder_color_policy": (
                "pydicom decoded pixel_array; YBR conversion is delegated exactly once to pydicom"
            ),
            "decoded_color_space": decoded_color_space,
            **padding_audit,
        }
    )
    return ds, raw, valid, metadata


def _declared_modality_transform(ds: Any) -> bool:
    return any(
        _tag_present(ds, keyword)
        for keyword in ("ModalityLUTSequence", "RescaleSlope", "RescaleIntercept")
    )


def _declared_voi_transform(ds: Any) -> bool:
    return any(
        _tag_present(ds, keyword)
        for keyword in ("VOILUTSequence", "WindowCenter", "WindowWidth")
    )


def apply_modality_transform(
    raw: np.ndarray,
    ds: Any,
    *,
    interpretation_policy: str,
) -> Tuple[np.ndarray, Dict[str, Any], List[str]]:
    """Apply DICOM modality LUT/rescale without presentation operations."""
    _, apply_modality_lut, _ = _import_pydicom()
    warnings: List[str] = []
    declared = _declared_modality_transform(ds)
    audit: Dict[str, Any] = {
        "modality_transform_declared": bool(declared),
        "modality_transform_applied": False,
    }
    if not declared:
        return np.asarray(raw), audit, warnings
    try:
        transformed = np.asarray(apply_modality_lut(raw, ds))
        audit["modality_transform_applied"] = True
        audit["modality_output_dtype"] = str(transformed.dtype)
        return transformed, audit, warnings
    except Exception as exc:
        message = f"Declared DICOM modality/rescale transform could not be applied: {exc}"
        audit["modality_transform_error"] = message
        if interpretation_policy == "strict":
            raise RetinalPreprocessError("modality_lut", message, audit) from exc
        warnings.append(message + "; continuing with decoded stored values by explicit audit policy")
        return np.asarray(raw), audit, warnings


def apply_display_transform(
    modality_pixels: np.ndarray,
    ds: Any,
    *,
    voi_policy: str,
    interpretation_policy: str,
) -> Tuple[np.ndarray, Dict[str, Any], List[str]]:
    """Apply optional VOI then MONOCHROME1 inversion to grayscale pixels."""
    _, _, apply_voi_lut = _import_pydicom()
    warnings: List[str] = []
    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    declared_voi = _declared_voi_transform(ds)
    audit: Dict[str, Any] = {
        "voi_transform_declared": bool(declared_voi),
        "voi_transform_policy": voi_policy,
        "voi_transform_applied": False,
        "monochrome1_inversion_applied": False,
        "transform_order": ["decoded_stored_values", "modality_lut_or_rescale"],
    }
    pixels = np.asarray(modality_pixels)
    if declared_voi and voi_policy == "apply_if_present":
        try:
            pixels = np.asarray(apply_voi_lut(pixels, ds, index=0))
            audit["voi_transform_applied"] = True
            audit["voi_output_dtype"] = str(pixels.dtype)
            audit["transform_order"].append("voi_lut_or_first_window")
        except Exception as exc:
            message = f"Declared DICOM VOI transform could not be applied: {exc}"
            audit["voi_transform_error"] = message
            if interpretation_policy == "strict":
                raise RetinalPreprocessError("voi_lut", message, audit) from exc
            warnings.append(message + "; continuing without VOI by explicit audit policy")
    elif declared_voi and voi_policy == "ignore":
        warnings.append("Declared DICOM VOI transform intentionally ignored by configuration")

    if photometric == "MONOCHROME1":
        finite = np.isfinite(pixels)
        if not finite.any():
            raise RetinalPreprocessError(
                "monochrome1_inversion", "MONOCHROME1 array has no finite values", audit
            )
        low = float(np.nanmin(pixels[finite]))
        high = float(np.nanmax(pixels[finite]))
        pixels = (low + high) - np.asarray(pixels, dtype=np.float64)
        audit["monochrome1_inversion_applied"] = True
        audit["monochrome1_inversion_range"] = [low, high]
        audit["transform_order"].append("monochrome1_presentation_inversion")
    elif photometric not in MONOCHROME_PHOTOMETRICS and not photometric.startswith(
        COLOR_PHOTOMETRIC_PREFIXES
    ):
        warnings.append(
            f"Unrecognized/empty PhotometricInterpretation={photometric!r}; pixel semantics uncertain"
        )

    presentation = str(getattr(ds, "PresentationLUTShape", "")).upper()
    if presentation == "INVERSE":
        warnings.append(
            "PresentationLUTShape=INVERSE is recorded but not composed beyond the explicit "
            "MONOCHROME1 rule; verify vendor presentation semantics before clinical display use"
        )
    return np.asarray(pixels), audit, warnings


def interpret_dicom_pixels(
    raw: np.ndarray,
    ds: Any,
    *,
    voi_policy: str,
    interpretation_policy: str,
) -> Tuple[np.ndarray, Dict[str, Any], List[str]]:
    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    color_audit: Dict[str, Any] = {}
    color_warnings: List[str] = []
    if photometric == "PALETTE COLOR":
        try:
            try:
                from pydicom.pixels import apply_color_lut  # type: ignore
            except ImportError:
                from pydicom.pixel_data_handlers.util import apply_color_lut  # type: ignore
            color = np.asarray(apply_color_lut(raw, ds))
        except Exception as exc:
            message = f"PALETTE COLOR LUT could not be applied: {exc}"
            raise RetinalPreprocessError("palette_color_lut", message) from exc
        color_audit.update(
            {
                "palette_color_lut_applied": True,
                "decoded_output_color_space": "RGB",
                "modality_transform_declared": _declared_modality_transform(ds),
                "modality_transform_applied": False,
                "voi_transform_declared": _declared_voi_transform(ds),
                "voi_transform_applied": False,
                "monochrome1_inversion_applied": False,
                "transform_order": ["decoded_palette_indices", "palette_color_lut_to_rgb"],
            }
        )
        if _declared_modality_transform(ds) or _declared_voi_transform(ds):
            color_warnings.append(
                "Grayscale modality/VOI tags on PALETTE COLOR input were recorded but not applied to RGB channels"
            )
        return color, color_audit, color_warnings
    if photometric == "RGB" or photometric.startswith("YBR"):
        color_audit.update(
            {
                "decoded_output_color_space": "RGB",
                "ybr_to_rgb_delegated_to_pydicom": photometric.startswith("YBR"),
                "modality_transform_declared": _declared_modality_transform(ds),
                "modality_transform_applied": False,
                "voi_transform_declared": _declared_voi_transform(ds),
                "voi_transform_applied": False,
                "monochrome1_inversion_applied": False,
                "transform_order": ["pydicom_decoded_rgb"],
            }
        )
        if _declared_modality_transform(ds) or _declared_voi_transform(ds):
            color_warnings.append(
                "Grayscale modality/VOI tags on RGB/YBR input were recorded but not applied per DICOM color semantics"
            )
        return np.asarray(raw), color_audit, color_warnings
    if photometric in {"HSV", "ARGB", "CMYK"}:
        raise RetinalPreprocessError(
            "color_space",
            f"Unsupported PhotometricInterpretation={photometric!r}; refusing uncertain channel semantics",
        )
    modality, modality_audit, modality_warnings = apply_modality_transform(
        raw, ds, interpretation_policy=interpretation_policy
    )
    display, display_audit, display_warnings = apply_display_transform(
        modality,
        ds,
        voi_policy=voi_policy,
        interpretation_policy=interpretation_policy,
    )
    return display, {**modality_audit, **display_audit}, modality_warnings + display_warnings


def _spatial_valid_for_array(valid: np.ndarray, expected_shape: Sequence[int]) -> np.ndarray:
    mask = np.asarray(valid, dtype=bool)
    if tuple(mask.shape) == tuple(expected_shape):
        return mask
    if mask.size == int(np.prod(expected_shape)):
        return mask.reshape(tuple(expected_shape))
    raise RetinalPreprocessError(
        "validity_shape",
        f"Decoded validity mask {mask.shape} cannot be aligned to pixel shape {tuple(expected_shape)}",
    )


def classify_pixel_array(
    pixels: np.ndarray,
    validity: np.ndarray,
    *,
    modality: str,
    protocol: str,
    color_hint: Optional[bool] = None,
) -> Tuple[str, np.ndarray, np.ndarray, List[str]]:
    """Map decoded DICOM axes to explicit 2-D/color/volume semantics."""
    arr = np.asarray(pixels)
    modality = modality.lower()
    protocol = protocol.lower()
    warnings: List[str] = []
    product = (
        "flio_decay"
        if modality == "flio"
        else "volume"
        if modality == "oct" or (modality == "octa" and protocol == "flow_cube")
        else "two_dimensional"
        if modality == "photography" or (modality == "octa" and protocol == "enface")
        else "unknown"
    )
    if product == "unknown":
        raise RetinalPreprocessError(
            "asset_semantics",
            f"Unsupported modality/protocol combination {modality!r}/{protocol!r}",
        )

    if product in {"volume", "flio_decay"}:
        if arr.ndim == 2:
            if product == "flio_decay":
                raise RetinalPreprocessError(
                    "dimension_interpretation",
                    f"FLIO decay cube is 2-D {arr.shape}; temporal frame axis is absent",
                )
            warnings.append("Declared volume has one decoded 2-D plane; retaining it as a one-slice volume")
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise RetinalPreprocessError(
                "dimension_interpretation",
                f"Expected frames x height x width for {modality}/{protocol}, got {arr.shape}",
            )
        mask = _spatial_valid_for_array(validity, arr.shape)
        mask &= np.isfinite(arr)
        return product, arr, mask, warnings

    # A 2-D product may be grayscale, RGB, or multi-frame grayscale/RGB.  Keep
    # every frame; never choose one without recording/controlling it.
    if arr.ndim == 2:
        arr = arr[None, ..., None]
    elif arr.ndim == 3 and (
        color_hint is True or (color_hint is None and arr.shape[-1] in (3, 4))
    ):
        if color_hint is None:
            warnings.append(
                "Color channel axis inferred from array shape because DICOM photometric semantics were unavailable"
            )
        if arr.shape[-1] == 4:
            warnings.append("Four-channel color input: alpha channel dropped after recording uncertainty")
            arr = arr[..., :3]
        arr = arr[None, ...]
    elif arr.ndim == 3:
        arr = arr[..., None]  # frames, H, W, 1
        warnings.append(f"Multi-frame 2-D product retained as {arr.shape[0]} independent model views")
    elif arr.ndim == 4 and (
        color_hint is True or (color_hint is None and arr.shape[-1] in (3, 4))
    ):
        if color_hint is None:
            warnings.append(
                "Multi-frame color channel axis inferred from shape because photometric semantics were unavailable"
            )
        if arr.shape[-1] == 4:
            warnings.append("Four-channel multi-frame input: alpha channel dropped")
            arr = arr[..., :3]
        warnings.append(f"Multi-frame color product retained as {arr.shape[0]} independent model views")
    else:
        raise RetinalPreprocessError(
            "dimension_interpretation",
            f"Expected HxW, HxWx3, FxHxW, or FxHxWx3 for 2-D product, got {arr.shape}",
        )
    if arr.shape[-1] not in (1, 3):
        raise RetinalPreprocessError(
            "channel_interpretation", f"Expected 1 or 3 channels after decoding, got {arr.shape}"
        )
    spatial_shape = arr.shape[:-1]
    mask = _spatial_valid_for_array(validity, spatial_shape)
    mask &= np.all(np.isfinite(arr), axis=-1)
    return product, arr, mask, warnings


def central_slice_indices(n_frames: int, count: int, stride: int) -> List[int]:
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if count <= 0 or count % 2 == 0:
        raise ValueError("center-slices must be a positive odd integer")
    if stride <= 0:
        raise ValueError("center-slice-stride must be positive")
    center = (n_frames - 1) / 2.0
    offsets = np.arange(-(count // 2), count // 2 + 1, dtype=np.float64) * stride
    requested = np.rint(center + offsets).astype(int)
    clipped = np.clip(requested, 0, n_frames - 1)
    # Preserve requested view count (including duplicated boundary slices for a
    # very shallow cube) and audit duplicates in metadata.
    return [int(value) for value in clipped]


def image_qc(
    image: np.ndarray,
    validity: np.ndarray,
    thresholds: ImageQCThresholds,
) -> Dict[str, Any]:
    arr = np.asarray(image, dtype=np.float64)
    spatial_valid = np.asarray(validity, dtype=bool)
    if arr.ndim == 3:
        finite_pixel = np.all(np.isfinite(arr), axis=-1)
        values = arr[spatial_valid & finite_pixel]
    else:
        finite_pixel = np.isfinite(arr)
        values = arr[spatial_valid & finite_pixel]
    denominator = max(int(spatial_valid.sum()), 1)
    finite_fraction = float(np.count_nonzero(spatial_valid & finite_pixel) / denominator)
    if values.size:
        q01, q50, q99 = np.percentile(values, [1.0, 50.0, 99.0])
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        scale = max(maximum - minimum, np.finfo(np.float64).eps)
        low_sat = float(np.mean(values <= minimum + 1e-6 * scale))
        high_sat = float(np.mean(values >= maximum - 1e-6 * scale))
    else:
        q01 = q50 = q99 = minimum = maximum = 0.0
        low_sat = high_sat = 1.0
    robust_range = float(q99 - q01)
    reasons: List[str] = []
    if finite_fraction < thresholds.min_finite_fraction:
        reasons.append("low_finite_fraction")
    if robust_range <= thresholds.min_robust_range:
        reasons.append("flat_or_near_constant")
    if low_sat > thresholds.max_low_saturation_fraction:
        reasons.append("mostly_at_minimum")
    if high_sat > thresholds.max_high_saturation_fraction:
        reasons.append("mostly_at_maximum")
    return {
        "finite_fraction": finite_fraction,
        "minimum": minimum,
        "q01": float(q01),
        "median": float(q50),
        "q99": float(q99),
        "maximum": maximum,
        "robust_range": robust_range,
        "low_saturation_fraction": low_sat,
        "high_saturation_fraction": high_sat,
        "quality_pass": len(reasons) == 0,
        "failure_reasons": reasons,
    }


def _percentile_limits(
    values: np.ndarray,
    validity: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> Tuple[float, float, List[str]]:
    arr = np.asarray(values, dtype=np.float64)
    valid = np.asarray(validity, dtype=bool) & np.isfinite(arr)
    warnings: List[str] = []
    observed = arr[valid]
    if observed.size == 0:
        raise RetinalPreprocessError("normalization", "No finite non-padding pixels")
    low, high = np.percentile(observed, [low_percentile, high_percentile])
    if not np.isfinite(low) or not np.isfinite(high):
        raise RetinalPreprocessError("normalization", "Non-finite percentile limits")
    if high <= low:
        low, high = float(observed.min()), float(observed.max())
        warnings.append("Requested percentile range collapsed; used finite min/max")
    if high <= low:
        # Keep a deterministic zero view but let QC mask it.
        high = low + 1.0
        warnings.append("Image is constant; normalized view is zero and quality mask is false")
    return float(low), float(high), warnings


def normalize_grayscale_stack(
    images: np.ndarray,
    validity: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> Tuple[np.ndarray, Dict[str, Any], List[str]]:
    arr = np.asarray(images, dtype=np.float64)
    valid = np.asarray(validity, dtype=bool)
    low, high, warnings = _percentile_limits(
        arr, valid, low_percentile, high_percentile
    )
    normalized = (arr - low) / (high - low)
    normalized = np.where(np.isfinite(normalized), normalized, 0.0)
    normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)
    normalized = np.where(valid, normalized, 0.0).astype(np.float32)
    return normalized, {
        "method": "joint_grayscale_percentile_affine",
        "low_percentile": low_percentile,
        "high_percentile": high_percentile,
        "low_value": low,
        "high_value": high,
        "scope": "all retained views/slices in this asset jointly",
        "participant_zscore": False,
    }, warnings


def normalize_rgb_stack(
    images: np.ndarray,
    validity: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> Tuple[np.ndarray, Dict[str, Any], List[str]]:
    arr = np.asarray(images, dtype=np.float64)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"normalize_rgb_stack expects FxHxWx3, got {arr.shape}")
    valid = np.asarray(validity, dtype=bool)
    finite_pixel = valid & np.all(np.isfinite(arr), axis=-1)
    observed = arr[finite_pixel]
    if observed.size == 0:
        raise RetinalPreprocessError("normalization", "No finite non-padding RGB pixels")
    low, high = np.percentile(observed.reshape(-1), [low_percentile, high_percentile])
    warnings: List[str] = []
    if not np.isfinite(high) or high <= 0:
        high = float(np.nanmax(observed))
        warnings.append("RGB upper percentile was non-positive; used finite joint maximum")
    if not np.isfinite(high) or high <= 0:
        raise RetinalPreprocessError(
            "normalization", "RGB image has no positive scale; quality is not interpretable"
        )
    negative_fraction = float(np.mean(observed < 0))
    upper_clip_fraction = float(np.mean(observed > high))
    if negative_fraction:
        warnings.append(
            f"Clipped {negative_fraction:.6%} negative RGB values to zero; positive-value ratios remain shared-scale"
        )
    # Pure multiplication by one scalar preserves positive RGB ratios exactly
    # until values reach the shared [0, 1] clipping ceiling.  A black-level
    # subtraction, even if shared, would alter channel ratios.
    normalized = arr / high
    normalized = np.where(np.isfinite(normalized), normalized, 0.0)
    normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)
    normalized = np.where(valid[..., None], normalized, 0.0).astype(np.float32)
    return normalized, {
        "method": "shared_rgb_joint_upper_percentile_scale",
        "low_percentile": low_percentile,
        "high_percentile": high_percentile,
        "observed_low_percentile_value_for_audit": float(low),
        "shared_scale_value": float(high),
        "shared_offset": 0.0,
        "same_scale_for_all_rgb_channels": True,
        "per_channel_stretch": False,
        "rgb_ratios_preserved_before_shared_ceiling_clip": True,
        "negative_fraction_clipped": negative_fraction,
        "upper_fraction_clipped_by_shared_scale": upper_clip_fraction,
        "scope": "all retained frames in this asset jointly",
        "participant_zscore": False,
    }, warnings


def resize_and_pad(
    image: np.ndarray,
    validity: np.ndarray,
    target_size: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Aspect-preserving resize/pad, returning CHW image and 1xHxW mask."""
    image_module = _require_pillow()
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3):
        raise ValueError(f"Expected HxWx1 or HxWx3, got {arr.shape}")
    height, width, channels = arr.shape
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image size {height}x{width}")
    scale = min(target_size / height, target_size / width)
    resized_h = max(1, min(target_size, int(round(height * scale))))
    resized_w = max(1, min(target_size, int(round(width * scale))))
    top = (target_size - resized_h) // 2
    bottom = target_size - resized_h - top
    left = (target_size - resized_w) // 2
    right = target_size - resized_w - left

    output = np.zeros((target_size, target_size, channels), dtype=np.float32)
    preclip_min = float("inf")
    preclip_max = float("-inf")
    n_below_zero = 0
    n_above_one = 0
    resized_value_count = 0
    for channel in range(channels):
        pil_channel = image_module.fromarray(arr[..., channel], mode="F")
        resized = pil_channel.resize((resized_w, resized_h), resample=image_module.Resampling.BICUBIC)
        resized_array = np.asarray(resized, dtype=np.float32)
        finite = np.isfinite(resized_array)
        if finite.any():
            preclip_min = min(preclip_min, float(resized_array[finite].min()))
            preclip_max = max(preclip_max, float(resized_array[finite].max()))
        n_below_zero += int(np.count_nonzero(resized_array < 0.0))
        n_above_one += int(np.count_nonzero(resized_array > 1.0))
        resized_value_count += int(resized_array.size)
        # Bicubic interpolation of bounded floats can overshoot at sharp edges.
        # Make the range contract explicit and audit how often clipping occurs.
        resized_array = np.nan_to_num(
            resized_array, nan=0.0, posinf=1.0, neginf=0.0
        )
        resized_array = np.clip(resized_array, 0.0, 1.0)
        output[top : top + resized_h, left : left + resized_w, channel] = resized_array

    source_mask = np.asarray(validity, dtype=bool)
    if source_mask.shape != (height, width):
        raise ValueError(
            f"Validity mask {source_mask.shape} does not match image {(height, width)}"
        )
    mask_pil = image_module.fromarray(source_mask.astype(np.uint8) * 255, mode="L")
    mask_resized = mask_pil.resize(
        (resized_w, resized_h), resample=image_module.Resampling.NEAREST
    )
    output_mask = np.zeros((target_size, target_size), dtype=bool)
    output_mask[top : top + resized_h, left : left + resized_w] = (
        np.asarray(mask_resized) > 0
    )
    geometry = {
        "original_height": int(height),
        "original_width": int(width),
        "resized_height": int(resized_h),
        "resized_width": int(resized_w),
        "target_height": int(target_size),
        "target_width": int(target_size),
        "scale": float(scale),
        "padding": {"top": top, "bottom": bottom, "left": left, "right": right},
        "aspect_ratio_preserved": True,
        "image_interpolation": "bicubic",
        "mask_interpolation": "nearest",
        "post_resize_range_clip": {
            "range": [0.0, 1.0],
            "preclip_min": preclip_min if np.isfinite(preclip_min) else None,
            "preclip_max": preclip_max if np.isfinite(preclip_max) else None,
            "values_below_zero": int(n_below_zero),
            "values_above_one": int(n_above_one),
            "resized_value_count": int(resized_value_count),
            "clipped_fraction": float(
                (n_below_zero + n_above_one) / max(resized_value_count, 1)
            ),
            "reason": "Pillow bicubic float interpolation can overshoot bounded inputs",
        },
    }
    return np.moveaxis(output, -1, 0), output_mask[None, ...], geometry


def _repeat_gray_to_rgb(image_chw: np.ndarray) -> np.ndarray:
    if image_chw.shape[0] == 3:
        return image_chw
    if image_chw.shape[0] != 1:
        raise ValueError(f"Expected 1 or 3 channels, got {image_chw.shape}")
    return np.repeat(image_chw, 3, axis=0)


def _resize_views(
    images: np.ndarray,
    validity: np.ndarray,
    target_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    views: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    geometries: List[Dict[str, Any]] = []
    for index in range(images.shape[0]):
        resized, mask, geometry = resize_and_pad(images[index], validity[index], target_size)
        views.append(_repeat_gray_to_rgb(resized))
        masks.append(mask)
        geometries.append(geometry)
    return np.stack(views), np.stack(masks), geometries


def _two_point_five_d(views: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Compose a conventional three-adjacent-slice 2.5-D RGB representation."""
    if views.ndim != 4 or views.shape[1] != 3:
        raise ValueError(f"Expected Vx3xHxW model views, got {views.shape}")
    count = views.shape[0]
    center = count // 2
    positions = [max(0, center - 1), center, min(count - 1, center + 1)]
    gray = views[:, 0, ...]
    return np.stack([gray[position] for position in positions], axis=0), positions


def _flio_auxiliaries(
    modality_cube: np.ndarray,
    validity: np.ndarray,
    *,
    chunk_frames: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], List[str]]:
    """Derive count intensity and photon-weighted *frame index* in chunks."""
    cube = np.asarray(modality_cube)
    if cube.ndim != 3:
        raise ValueError(f"FLIO cube must be FxHxW, got {cube.shape}")
    n_frames, height, width = cube.shape
    valid = np.asarray(validity, dtype=bool)
    warnings: List[str] = []
    intensity = np.zeros((height, width), dtype=np.float64)
    weighted = np.zeros((height, width), dtype=np.float64)
    negative_count = 0
    nonfinite_count = 0
    total = int(cube.size)
    for start in range(0, n_frames, chunk_frames):
        stop = min(n_frames, start + chunk_frames)
        block = np.asarray(cube[start:stop], dtype=np.float64)
        block_valid = valid[start:stop] & np.isfinite(block)
        nonfinite_count += int(np.count_nonzero(valid[start:stop] & ~np.isfinite(block)))
        negative_count += int(np.count_nonzero(block_valid & (block < 0)))
        # Negative baseline-corrected values cannot be photon weights.  Clip for
        # this auxiliary only; preserve the original cube and audit the fraction.
        weights = np.where(block_valid, np.maximum(block, 0.0), 0.0)
        frame_index = np.arange(start, stop, dtype=np.float64)[:, None, None]
        intensity += weights.sum(axis=0, dtype=np.float64)
        weighted += (weights * frame_index).sum(axis=0, dtype=np.float64)
    photon_valid = intensity > 0
    mean_bin = np.full((height, width), np.nan, dtype=np.float32)
    mean_bin[photon_valid] = (weighted[photon_valid] / intensity[photon_valid]).astype(
        np.float32
    )
    if negative_count:
        warnings.append(
            f"Clipped {negative_count} negative modality values to zero for photon-weighted auxiliaries only"
        )
    if nonfinite_count:
        warnings.append(
            f"Excluded {nonfinite_count} non-finite modality values from FLIO auxiliaries"
        )
    metadata = {
        "n_frames": int(n_frames),
        "frame_index_origin": 0,
        "intensity_definition": "sum of non-negative modality values across decay frames",
        "mean_bin_definition": "sum(frame_index * photon_weight) / sum(photon_weight)",
        "mean_bin_units": "frame_index (dimensionless bin number)",
        "physical_lifetime": False,
        "irf_corrected": False,
        "bin_width_or_time_calibration_applied": False,
        "negative_values_clipped_for_auxiliary_only": int(negative_count),
        "nonfinite_values_excluded": int(nonfinite_count),
        "valid_photon_pixels": int(photon_valid.sum()),
        "total_values": total,
    }
    return intensity.astype(np.float32), mean_bin, photon_valid, metadata, warnings


def _zlib_compress_array(array: np.ndarray, level: int) -> Tuple[bytes, str]:
    """Losslessly compress a contiguous array while hashing uncompressed bytes."""
    contiguous = np.ascontiguousarray(array)
    view = memoryview(contiguous).cast("B")
    compressor = zlib.compressobj(level)
    digest = hashlib.sha256()
    pieces: List[bytes] = []
    chunk_bytes = 16 * 1024 * 1024
    for start in range(0, len(view), chunk_bytes):
        chunk = view[start : start + chunk_bytes]
        digest.update(chunk)
        encoded = compressor.compress(chunk)
        if encoded:
            pieces.append(encoded)
    pieces.append(compressor.flush())
    return b"".join(pieces), digest.hexdigest()


def encode_flio_decay_cube(
    decoded_stored_values: np.ndarray,
    validity: np.ndarray,
    *,
    compression_level: int,
) -> Dict[str, Any]:
    """Encode an exact full FLIO cube and bit-packed validity mask.

    The archive is a regular Python mapping so it can live inside the same
    atomic ``.pt`` artifact.  It is lossless: dtype, byte order, shape, every
    frame, and a SHA-256 of the uncompressed cube are retained.
    """
    cube = np.ascontiguousarray(decoded_stored_values)
    mask = np.asarray(validity, dtype=bool)
    if cube.shape != mask.shape:
        raise ValueError(f"FLIO cube/mask shape mismatch: {cube.shape} vs {mask.shape}")
    compressed_cube, cube_sha256 = _zlib_compress_array(cube, compression_level)
    packed_mask = np.packbits(mask.reshape(-1), bitorder="little")
    compressed_mask, mask_sha256 = _zlib_compress_array(packed_mask, compression_level)
    uncompressed_bytes = int(cube.nbytes)
    compressed_bytes = int(len(compressed_cube))
    return {
        "format": "aireadi_flio_decay_cube_lossless_v1",
        "codec": "zlib",
        "compression_level": int(compression_level),
        "shape": list(cube.shape),
        "dtype": cube.dtype.str,
        "order": "C",
        "all_frames_preserved": True,
        "uncompressed_nbytes": uncompressed_bytes,
        "compressed_nbytes": compressed_bytes,
        "compression_ratio": (
            float(uncompressed_bytes / compressed_bytes) if compressed_bytes else None
        ),
        "decoded_stored_values_sha256": cube_sha256,
        "decoded_stored_values_zlib": compressed_cube,
        "validity_mask": {
            "encoding": "numpy.packbits(bitorder=little), then zlib",
            "n_values": int(mask.size),
            "packed_dtype": packed_mask.dtype.str,
            "packed_sha256": mask_sha256,
            "compressed_nbytes": int(len(compressed_mask)),
            "packed_zlib": compressed_mask,
        },
    }


def decode_flio_decay_cube(archive: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Decode and integrity-check an archive produced by :func:`encode_flio_decay_cube`."""
    if archive.get("format") != "aireadi_flio_decay_cube_lossless_v1":
        raise ValueError(f"Unknown FLIO decay archive format: {archive.get('format')!r}")
    raw_bytes = zlib.decompress(archive["decoded_stored_values_zlib"])
    digest = hashlib.sha256(raw_bytes).hexdigest()
    if digest != archive.get("decoded_stored_values_sha256"):
        raise ValueError("FLIO decay cube SHA-256 mismatch")
    shape = tuple(int(value) for value in archive["shape"])
    dtype = np.dtype(str(archive["dtype"]))
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(raw_bytes) != expected_bytes:
        raise ValueError(
            f"FLIO decay byte length mismatch: {len(raw_bytes)} vs expected {expected_bytes}"
        )
    cube = np.frombuffer(raw_bytes, dtype=dtype).copy().reshape(shape)

    mask_archive = archive["validity_mask"]
    packed_bytes = zlib.decompress(mask_archive["packed_zlib"])
    packed_digest = hashlib.sha256(packed_bytes).hexdigest()
    if packed_digest != mask_archive.get("packed_sha256"):
        raise ValueError("FLIO decay validity-mask SHA-256 mismatch")
    packed = np.frombuffer(packed_bytes, dtype=np.dtype(mask_archive["packed_dtype"]))
    n_values = int(mask_archive["n_values"])
    mask = np.unpackbits(packed, bitorder="little", count=n_values).astype(bool).reshape(shape)
    return cube, mask


def _manifest_metadata(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {str(key): _json_scalar(value) for key, value in row.items()}


def _spacing_metadata(row: Mapping[str, Any], dicom: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "manifest_pixel_spacing": _json_scalar(row.get("pixel_spacing")),
        "manifest_slice_thickness": _json_scalar(row.get("slice_thickness")),
        "dicom_pixel_spacing": dicom.get("pixel_spacing"),
        "dicom_imager_pixel_spacing": dicom.get("imager_pixel_spacing"),
        "dicom_slice_thickness": dicom.get("slice_thickness"),
        "dicom_spacing_between_slices": dicom.get("spacing_between_slices"),
        "dicom_frame_time": dicom.get("frame_time"),
        "dicom_frame_time_vector": dicom.get("frame_time_vector"),
        "shared_functional_group_pixel_measures": dicom.get(
            "shared_functional_group_pixel_measures"
        ),
        "per_frame_pixel_measures_unique": dicom.get(
            "per_frame_pixel_measures_unique"
        ),
        "per_frame_spacing_varies": dicom.get("per_frame_spacing_varies"),
        "note": "Values are retained as metadata; resize does not silently claim physical isotropy.",
    }


def _torch_load(path: Path) -> Dict[str, Any]:
    torch = _import_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch versions
        return torch.load(path, map_location="cpu")


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    torch = _import_torch()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        # os.replace exposes either the previous complete artifact or the new
        # complete artifact.  Avoid a read-back here: a preserved FLIO cube can
        # be hundreds of MiB and loading it a second time would double peak RAM.
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _cache_status(
    output_path: Path,
    config_hash: str,
    source_fingerprint: str,
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    if not output_path.exists():
        return False, None, "missing"
    try:
        artifact = _torch_load(output_path)
    except Exception as exc:
        return False, None, f"unreadable:{exc!r}"
    if artifact.get("artifact_config_hash") != config_hash:
        return False, artifact, "config_hash_mismatch"
    if artifact.get("source_fingerprint") != source_fingerprint:
        return False, artifact, "source_fingerprint_mismatch"
    return True, artifact, "valid"


def _summary_from_artifact(
    artifact: Mapping[str, Any],
    output_path: Path,
    status: str,
) -> Dict[str, Any]:
    metadata = artifact.get("asset_metadata", {})
    audit = artifact.get("pixel_interpretation_audit", {})
    warnings = list(artifact.get("interpretation_warnings", []))
    quality_mask = np.asarray(artifact.get("view_quality_mask", []), dtype=bool)
    model_views = np.asarray(artifact.get("model_views", []))
    return {
        "status": status,
        "person_id": str(artifact.get("person_id", "")),
        "asset_id": str(artifact.get("asset_id", artifact.get("asset_key", ""))),
        "asset_key": str(artifact.get("asset_key", "")),
        "artifact_path": str(output_path),
        "modality": metadata.get("modality"),
        "protocol": metadata.get("protocol"),
        "laterality": metadata.get("laterality"),
        "slab_or_wavelength": metadata.get("slab_or_wavelength"),
        "filepath": metadata.get("filepath"),
        "split": metadata.get("split"),
        "clinical_site": metadata.get("clinical_site"),
        "product_kind": artifact.get("product_kind"),
        "decoded_shape": json.dumps(audit.get("decoded_shape", []), separators=(",", ":")),
        "model_views_shape": json.dumps(list(model_views.shape), separators=(",", ":")),
        "n_model_views": int(model_views.shape[0]) if model_views.ndim else 0,
        "n_quality_views": int(quality_mask.sum()),
        "all_views_quality_pass": bool(quality_mask.all()) if quality_mask.size else False,
        "monochrome1_inverted": bool(audit.get("monochrome1_inversion_applied", False)),
        "modality_transform_applied": bool(audit.get("modality_transform_applied", False)),
        "voi_transform_applied": bool(audit.get("voi_transform_applied", False)),
        "n_interpretation_warnings": int(len(warnings)),
        "interpretation_warnings": json.dumps(warnings, ensure_ascii=False, separators=(",", ":")),
        "input_index_config_hash": artifact.get("input_index_config_hash", ""),
        "input_index_source_fingerprint": artifact.get(
            "input_index_source_fingerprint", ""
        ),
        "artifact_config_hash": artifact.get("artifact_config_hash"),
        "source_fingerprint": artifact.get("source_fingerprint"),
        "artifact_size_bytes": int(output_path.stat().st_size) if output_path.is_file() else None,
    }


def _to_torch_artifact_arrays(artifact: MutableMapping[str, Any]) -> None:
    torch = _import_torch()
    for key in (
        "model_views",
        "model_pixel_mask",
        "view_quality_mask",
        "two_point_five_d_view",
        "flio_intensity",
        "flio_photon_weighted_mean_bin",
        "flio_photon_valid_mask",
    ):
        value = artifact.get(key)
        if isinstance(value, np.ndarray):
            artifact[key] = torch.from_numpy(np.ascontiguousarray(value))
    # The exact full FLIO cube remains a lossless compressed byte archive rather
    # than a tensor: this preserves unsigned DICOM dtypes without a lossy cast
    # and avoids roughly terabyte-scale uncompressed cohort caches.


def process_asset(
    row: Dict[str, Any],
    *,
    dataset_root_text: str,
    input_index_text: str,
    artifact_root_text: str,
    run_config: Dict[str, Any],
    qc_payload: Dict[str, Any],
    cache_policy: str,
) -> Dict[str, Any]:
    person_id = as_person_id(row.get("person_id"))
    asset_id = str(row.get("asset_id", row["asset_key"]))
    asset_key = str(row["asset_key"])
    modality = _norm(row.get("modality")).lower()
    protocol = _norm(row.get("protocol")).lower()
    slab_or_wavelength = _norm(row.get("slab_or_wavelength"))
    output_path = (
        Path(artifact_root_text) / _safe_token(modality, "unknown") / f"{asset_key}.pt"
    )
    dataset_root = Path(dataset_root_text)
    input_index = Path(input_index_text)
    source_path = resolve_path(dataset_root, row.get("filepath"))
    row_fingerprint = str(row["input_row_fingerprint"])
    # Cache validity is asset-local: the exact input row (including upstream
    # config/source hashes) plus raw-file size/mtime.  An unrelated row added to
    # the all-assets CSV should not invalidate this asset, and the identical row
    # in all-assets versus default-view CSV should be reusable.
    sources = source_file_metadata([source_path])
    input_index_file_metadata = source_file_metadata([input_index])
    source_fingerprint = _source_fingerprint(sources, row_fingerprint)
    config_hash = str(run_config["config_hash"])
    partial_audit: Dict[str, Any] = {}
    warnings: List[str] = []

    try:
        _require_pillow()
        _import_torch()
        if not source_path.is_file():
            raise RetinalPreprocessError(
                "path_resolution",
                f"Resolved source is not a file: manifest={row.get('filepath')!r}, resolved={source_path}",
            )
        if cache_policy != "rebuild":
            valid_cache, cached, cache_reason = _cache_status(
                output_path, config_hash, source_fingerprint
            )
            if valid_cache and cached is not None:
                return _summary_from_artifact(cached, output_path, "cached")
            if output_path.exists() and cache_policy == "error_on_stale":
                raise RetinalPreprocessError(
                    "cache", f"Stale retinal cache {output_path}: {cache_reason}"
                )

        ds, raw, raw_validity, dicom_metadata = read_dicom(source_path)
        partial_audit.update(dicom_metadata)
        qc = ImageQCThresholds(**qc_payload)
        geometry_audit, geometry_warnings = _geometry_consistency_audit(
            row, dicom_metadata
        )
        partial_audit["manifest_dicom_geometry"] = geometry_audit
        warnings.extend(geometry_warnings)
        manifest_laterality = _canonical_laterality(row.get("laterality"))
        dicom_laterality = _canonical_laterality(
            dicom_metadata.get("image_laterality") or dicom_metadata.get("laterality")
        )
        if manifest_laterality and dicom_laterality and manifest_laterality != dicom_laterality:
            warnings.append(
                f"Manifest laterality {manifest_laterality!r} disagrees with DICOM laterality {dicom_laterality!r}"
            )
            partial_audit["laterality_mismatch"] = True
        else:
            partial_audit["laterality_mismatch"] = False

        product_expected = (
            "flio_decay"
            if modality == "flio"
            else "volume"
            if modality == "oct" or (modality == "octa" and protocol == "flow_cube")
            else "two_dimensional"
        )
        if product_expected == "flio_decay":
            # Preserve decoded stored values exactly.  For photon auxiliaries,
            # apply only modality/rescale calibration: VOI and MONOCHROME1 are
            # presentation operations and must not become photon weights.
            modality_cube, modality_audit, modality_warnings = apply_modality_transform(
                raw,
                ds,
                interpretation_policy=str(run_config["interpretation_policy"]),
            )
            partial_audit.update(modality_audit)
            partial_audit.update(
                {
                    "voi_transform_declared": _declared_voi_transform(ds),
                    "voi_transform_applied": False,
                    "voi_transform_policy": "not_applied_to_quantitative_flio_counts",
                    "monochrome1_inversion_applied": False,
                    "flio_count_pipeline": [
                        "decoded_stored_values",
                        "modality_lut_or_rescale_if_declared",
                        "no_voi",
                        "no_photometric_inversion",
                    ],
                }
            )
            warnings.extend(modality_warnings)
            if _declared_voi_transform(ds):
                warnings.append(
                    "DICOM VOI is display-only and was not applied to FLIO photon counts"
                )
            if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
                warnings.append(
                    "MONOCHROME1 presentation inversion was not applied to FLIO photon weights; "
                    "stored decay counts are preserved and the exception is explicit"
                )
            product_kind, cube, cube_validity, dimension_warnings = classify_pixel_array(
                modality_cube,
                raw_validity,
                modality=modality,
                protocol=protocol,
                color_hint=False,
            )
            warnings.extend(dimension_warnings)
            intensity, mean_bin, photon_valid, flio_meta, flio_warnings = _flio_auxiliaries(
                cube,
                cube_validity,
                chunk_frames=int(run_config["flio_chunk_frames"]),
            )
            warnings.extend(flio_warnings)

            intensity_valid = np.any(cube_validity, axis=0) & np.isfinite(intensity)
            mean_valid = intensity_valid & photon_valid & np.isfinite(mean_bin)
            # These auxiliaries have different units and radically different
            # scales.  Normalize them separately; a joint percentile would make
            # mean-bin nearly constant relative to photon intensity.  log1p is
            # display compression only and the uncompressed intensity is saved.
            intensity_preview = np.log1p(np.maximum(intensity, 0.0)).astype(np.float32)
            mean_preview = np.where(mean_valid, mean_bin, 0.0).astype(np.float32)
            intensity_normalized, intensity_norm, intensity_norm_warnings = normalize_grayscale_stack(
                intensity_preview[None, ...],
                intensity_valid[None, ...],
                float(run_config["low_percentile"]),
                float(run_config["high_percentile"]),
            )
            mean_normalized, mean_norm, mean_norm_warnings = normalize_grayscale_stack(
                mean_preview[None, ...],
                mean_valid[None, ...],
                float(run_config["low_percentile"]),
                float(run_config["high_percentile"]),
            )
            warnings.extend(intensity_norm_warnings)
            warnings.extend(mean_norm_warnings)
            normalized = np.concatenate([intensity_normalized, mean_normalized], axis=0)[..., None]
            preview_stack = np.stack([intensity, mean_preview], axis=0)
            preview_valid = np.stack([intensity_valid, mean_valid], axis=0)
            normalization = {
                "method": "separate_semantic_auxiliary_normalization",
                "flio_intensity": {
                    "pretransform": "log1p(nonnegative intensity)",
                    **intensity_norm,
                },
                "flio_photon_weighted_mean_bin": {
                    "pretransform": "none",
                    **mean_norm,
                },
                "participant_zscore": False,
            }
            model_views, model_masks, geometry = _resize_views(
                normalized, preview_valid, int(run_config["target_size"])
            )
            quality_records = [
                image_qc(preview_stack[index], preview_valid[index], qc)
                for index in range(2)
            ]
            quality_mask = np.asarray(
                [record["quality_pass"] for record in quality_records], dtype=bool
            )
            artifact: Dict[str, Any] = {
                "person_id": person_id,
                "asset_id": asset_id,
                "asset_key": asset_key,
                "product_kind": product_kind,
                "model_view_names": ["flio_intensity", "flio_photon_weighted_mean_bin"],
                "model_views": model_views.astype(np.float32),
                "model_pixel_mask": model_masks.astype(bool),
                "view_quality_mask": quality_mask,
                "view_quality_metrics": quality_records,
                "view_geometry": geometry,
                "normalization": normalization,
                "flio_intensity": intensity.astype(np.float32),
                "flio_photon_weighted_mean_bin": mean_bin.astype(np.float32),
                "flio_photon_valid_mask": photon_valid.astype(bool),
                "flio_auxiliary_metadata": flio_meta,
                "flio_decay_cube_storage": str(run_config["flio_cube_policy"]),
            }
            if str(run_config["flio_cube_policy"]) == "preserve":
                artifact["flio_decay_cube_lossless_archive"] = encode_flio_decay_cube(
                    raw,
                    raw_validity,
                    compression_level=int(run_config["flio_compression_level"]),
                )
            else:
                warnings.append(
                    "Full FLIO decay cube omitted by explicit aux_only configuration; this is not the default"
                )
        else:
            interpreted, transform_audit, transform_warnings = interpret_dicom_pixels(
                raw,
                ds,
                voi_policy=str(run_config["voi_policy"]),
                interpretation_policy=str(run_config["interpretation_policy"]),
            )
            partial_audit.update(transform_audit)
            warnings.extend(transform_warnings)
            photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
            if photometric in MONOCHROME_PHOTOMETRICS:
                color_hint: Optional[bool] = False
            elif photometric == "RGB" or photometric.startswith("YBR") or photometric == "PALETTE COLOR":
                color_hint = True
            else:
                color_hint = None
            product_kind, structured, structured_validity, dimension_warnings = classify_pixel_array(
                interpreted,
                raw_validity,
                modality=modality,
                protocol=protocol,
                color_hint=color_hint,
            )
            warnings.extend(dimension_warnings)

            if product_kind == "volume":
                indices = central_slice_indices(
                    structured.shape[0],
                    int(run_config["center_slices"]),
                    int(run_config["center_slice_stride"]),
                )
                selected = np.asarray(structured[indices])
                selected_validity = np.asarray(structured_validity[indices], dtype=bool)
                if len(set(indices)) < len(indices):
                    warnings.append(
                        "Cube is shallower than requested central stack; boundary slices were duplicated"
                    )
                normalized, normalization, normalize_warnings = normalize_grayscale_stack(
                    selected,
                    selected_validity,
                    float(run_config["low_percentile"]),
                    float(run_config["high_percentile"]),
                )
                warnings.extend(normalize_warnings)
                model_views, model_masks, geometry = _resize_views(
                    normalized[..., None],
                    selected_validity,
                    int(run_config["target_size"]),
                )
                quality_records = [
                    image_qc(selected[index], selected_validity[index], qc)
                    for index in range(selected.shape[0])
                ]
                quality_mask = np.asarray(
                    [record["quality_pass"] for record in quality_records], dtype=bool
                )
                view_25d, positions_25d = _two_point_five_d(model_views)
                artifact = {
                    "person_id": person_id,
                    "asset_id": asset_id,
                    "asset_key": asset_key,
                    "product_kind": product_kind,
                    "selected_source_frame_indices": indices,
                    "model_view_names": [f"central_slice_{index}" for index in indices],
                    "model_views": model_views.astype(np.float32),
                    "model_pixel_mask": model_masks.astype(bool),
                    "view_quality_mask": quality_mask,
                    "view_quality_metrics": quality_records,
                    "view_geometry": geometry,
                    "normalization": normalization,
                    "two_point_five_d_view": view_25d.astype(np.float32),
                    "two_point_five_d_positions_in_model_views": positions_25d,
                    "volume_metadata": {
                        "decoded_frame_count": int(structured.shape[0]),
                        "selected_frame_count": int(len(indices)),
                        "selection_policy": "symmetric central slices with configurable stride",
                        "full_volume_saved": False,
                        "note": "The configurable multi-slice stack and 2.5-D view replace legacy single-middle-slice collapse.",
                    },
                }
            else:
                is_rgb = structured.shape[-1] == 3
                if is_rgb:
                    normalized, normalization, normalize_warnings = normalize_rgb_stack(
                        structured,
                        structured_validity,
                        float(run_config["low_percentile"]),
                        float(run_config["high_percentile"]),
                    )
                else:
                    normalized_gray, normalization, normalize_warnings = normalize_grayscale_stack(
                        structured[..., 0],
                        structured_validity,
                        float(run_config["low_percentile"]),
                        float(run_config["high_percentile"]),
                    )
                    normalized = normalized_gray[..., None]
                warnings.extend(normalize_warnings)
                model_views, model_masks, geometry = _resize_views(
                    normalized,
                    structured_validity,
                    int(run_config["target_size"]),
                )
                quality_records = [
                    image_qc(structured[index], structured_validity[index], qc)
                    for index in range(structured.shape[0])
                ]
                quality_mask = np.asarray(
                    [record["quality_pass"] for record in quality_records], dtype=bool
                )
                artifact = {
                    "person_id": person_id,
                    "asset_id": asset_id,
                    "asset_key": asset_key,
                    "product_kind": product_kind,
                    "model_view_names": [f"frame_{index}" for index in range(structured.shape[0])],
                    "model_views": model_views.astype(np.float32),
                    "model_pixel_mask": model_masks.astype(bool),
                    "view_quality_mask": quality_mask,
                    "view_quality_metrics": quality_records,
                    "view_geometry": geometry,
                    "normalization": normalization,
                    "two_dimensional_metadata": {
                        "decoded_frame_count": int(structured.shape[0]),
                        "decoded_channels": int(structured.shape[-1]),
                        "all_frames_retained": True,
                        "enface_processed_independently": modality == "octa" and protocol == "enface",
                    },
                }

        partial_audit["decoded_shape"] = list(raw.shape)
        partial_audit["decoded_dtype"] = str(raw.dtype)
        partial_audit["product_kind"] = artifact["product_kind"]
        artifact.update(
            {
                "asset_metadata": _manifest_metadata(row),
                "input_index_config_hash": _norm(row.get("input_index_config_hash")),
                "input_index_source_fingerprint": _norm(
                    row.get("input_index_source_fingerprint")
                ),
                "protocol_metadata": {
                    "modality": modality,
                    "protocol": protocol,
                    "laterality": _canonical_laterality(row.get("laterality")),
                    "slab_or_wavelength": slab_or_wavelength,
                    "is_projection_removed": _json_scalar(row.get("is_projection_removed")),
                    "source_variant": _json_scalar(row.get("source_variant")),
                    "segmentation_surface_1": _json_scalar(row.get("segmentation_surface_1")),
                    "segmentation_surface_2": _json_scalar(row.get("segmentation_surface_2")),
                    "manufacturer": _json_scalar(row.get("manufacturer")),
                    "device_model": _json_scalar(row.get("device_model")),
                    "anatomic_region": _json_scalar(row.get("anatomic_region")),
                },
                "spacing_metadata": _spacing_metadata(row, dicom_metadata),
                "dicom_metadata": dicom_metadata,
                "pixel_interpretation_audit": partial_audit,
                "interpretation_warnings": sorted(set(warnings)),
                "artifact_config": run_config,
                "artifact_config_hash": config_hash,
                "source_files": sources,
                "artifact_build_input_index_file": input_index_file_metadata,
                "input_row_fingerprint": row_fingerprint,
                "source_fingerprint": source_fingerprint,
                "normalization_scope_note": (
                    "No participant-specific z-score; learned normalization must be fit on training assets only."
                ),
            }
        )
        _to_torch_artifact_arrays(artifact)
        _atomic_torch_save(artifact, output_path)
        return _summary_from_artifact(artifact, output_path, "processed")
    except Exception as exc:
        if isinstance(exc, RetinalPreprocessError):
            stage = exc.stage
            partial_audit.update(exc.audit)
        else:
            stage = "unexpected"
        return {
            "status": "failed",
            "person_id": person_id,
            "asset_id": asset_id,
            "asset_key": asset_key,
            "modality": modality,
            "protocol": protocol,
            "laterality": _canonical_laterality(row.get("laterality")),
            "slab_or_wavelength": slab_or_wavelength,
            "filepath": _norm(row.get("filepath")),
            "resolved_path": str(source_path),
            "failure_stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "pixel_interpretation_audit": partial_audit,
            "interpretation_warnings": sorted(set(warnings)),
            "traceback": traceback.format_exc(limit=10),
            "input_index_config_hash": _norm(row.get("input_index_config_hash")),
            "input_index_source_fingerprint": _norm(
                row.get("input_index_source_fingerprint")
            ),
            "artifact_config_hash": config_hash,
            "source_fingerprint": source_fingerprint,
        }


def _build_run_config(args: argparse.Namespace, qc: ImageQCThresholds) -> Dict[str, Any]:
    return artifact_config(
        modality="retinal_assets",
        extra={
            "artifact_schema_version": "retinal-assets-v2.0",
            "target_size": int(args.target_size),
            "low_percentile": float(args.low_percentile),
            "high_percentile": float(args.high_percentile),
            "rgb_normalization": (
                "single joint upper-percentile multiplicative scale shared by RGB channels; zero offset"
            ),
            "grayscale_normalization": "joint per-asset retained-view percentile affine",
            "participant_specific_zscore": False,
            "resize": "aspect_preserving_bicubic_then_symmetric_zero_pad",
            "augmentation": "none; deterministic raw-to-canonical preprocessing only",
            "center_slices": int(args.center_slices),
            "center_slice_stride": int(args.center_slice_stride),
            "volume_model_views": "all selected central slices plus 3-channel 2.5-D composition",
            "voi_policy": str(args.voi_policy),
            "interpretation_policy": str(args.interpretation_policy),
            "monochrome1_policy": "invert_after_modality_and_voi_for_display products",
            "flio_cube_policy": str(args.flio_cube_policy),
            "flio_chunk_frames": int(args.flio_chunk_frames),
            "flio_compression_level": int(args.flio_compression_level),
            "flio_lossless_storage": (
                "full native-dtype decoded stored-value cube, zlib compressed with shape and SHA-256"
            ),
            "flio_auxiliary_semantics": (
                "intensity plus photon-weighted zero-based mean frame index; not a physical or IRF-corrected lifetime"
            ),
            "image_qc_thresholds": asdict(qc),
            "runtime_versions": _runtime_versions(),
        },
    )


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-index",
        "--asset_index",
        "--manifest",
        dest="asset_index",
        required=True,
        help="retinal_asset_index.csv or retinal_default_model_view.csv",
    )
    parser.add_argument("--dataset-root", "--dataset_root", dest="dataset_root", required=True)
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir", required=True)
    parser.add_argument("--target-size", "--target_size", dest="target_size", type=int, default=224)
    parser.add_argument("--low-percentile", type=float, default=1.0)
    parser.add_argument("--high-percentile", type=float, default=99.0)
    parser.add_argument("--center-slices", type=int, default=5)
    parser.add_argument("--center-slice-stride", type=int, default=1)
    parser.add_argument(
        "--voi-policy",
        choices=("apply_if_present", "ignore"),
        default="apply_if_present",
        help="VOI is applied after modality LUT for non-FLIO display products",
    )
    parser.add_argument(
        "--interpretation-policy",
        choices=("strict", "audit"),
        default="strict",
        help="strict fails if a declared LUT cannot be applied; audit records and continues",
    )
    parser.add_argument(
        "--flio-cube-policy",
        choices=("preserve", "aux_only"),
        default="preserve",
        help=(
            "preserve stores the complete decoded decay cube (scientific default). "
            "A typical 1024x256x256 uint16 cube is ~128 MiB before data-dependent "
            "lossless compression; pilot default-view with --limit before a cohort run"
        ),
    )
    parser.add_argument("--flio-chunk-frames", type=int, default=64)
    parser.add_argument(
        "--flio-compression-level",
        type=int,
        default=6,
        help="zlib level 0-9 for exact full-cube storage; does not change pixel values",
    )
    parser.add_argument(
        "--cache-policy",
        choices=("reuse", "rebuild", "error_on_stale"),
        default="reuse",
    )
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N indexed assets; recommended for storage/runtime pilots",
    )
    parser.add_argument("--no-progress", "--no_progress", dest="no_progress", action="store_true")
    parser.add_argument("--qc-min-finite-fraction", type=float, default=0.98)
    parser.add_argument("--qc-min-robust-range", type=float, default=1e-8)
    parser.add_argument("--qc-max-low-saturation-fraction", type=float, default=0.98)
    parser.add_argument("--qc-max-high-saturation-fraction", type=float, default=0.98)
    return parser


def main() -> None:
    args = _arg_parser().parse_args()
    if args.target_size <= 0:
        raise ValueError("target-size must be positive")
    if not (0.0 <= args.low_percentile < args.high_percentile <= 100.0):
        raise ValueError("Require 0 <= low-percentile < high-percentile <= 100")
    if args.center_slices <= 0 or args.center_slices % 2 == 0:
        raise ValueError("center-slices must be a positive odd integer")
    if args.center_slice_stride <= 0:
        raise ValueError("center-slice-stride must be positive")
    if args.flio_chunk_frames <= 0:
        raise ValueError("flio-chunk-frames must be positive")
    if not 0 <= args.flio_compression_level <= 9:
        raise ValueError("flio-compression-level must be in [0, 9]")
    if args.num_workers <= 0:
        raise ValueError("num-workers must be >= 1")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")

    # Fail once, clearly, before creating a partial run when core dependencies
    # are absent.  Decoder plugins can still fail per asset and are audited.
    _require_pillow()
    _import_torch()
    _import_pydicom()

    qc = ImageQCThresholds(
        min_finite_fraction=args.qc_min_finite_fraction,
        min_robust_range=args.qc_min_robust_range,
        max_low_saturation_fraction=args.qc_max_low_saturation_fraction,
        max_high_saturation_fraction=args.qc_max_high_saturation_fraction,
    )
    run_config = _build_run_config(args, qc)
    input_index = Path(args.asset_index).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    artifact_root = out_dir / "retinal_assets" / str(run_config["config_hash"])[:16]
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "artifact_config.json", run_config)

    source_frame = pd.read_csv(
        input_index,
        low_memory=False,
        dtype={
            "person_id": "string",
            "asset_id": "string",
            "config_hash": "string",
            "source_fingerprint": "string",
            "sop_instance_uid": "string",
            "filepath": "string",
        },
    )
    assets = prepare_asset_index(source_frame)
    if args.limit is not None:
        assets = assets.head(args.limit).copy()
    estimated_flio_uint16_bytes = estimate_flio_uint16_uncompressed_bytes(assets)
    if args.flio_cube_policy == "preserve" and estimated_flio_uint16_bytes > 0:
        print(
            "WARNING: selected FLIO cubes declare approximately "
            f"{estimated_flio_uint16_bytes / (1024 ** 3):.2f} GiB as uint16 before "
            "lossless compression. Compression ratio is data dependent. Pilot "
            "retinal_default_model_view.csv with --limit and inspect artifact_size_bytes.",
            file=sys.stderr,
            flush=True,
        )
    records = assets.to_dict("records")
    kwargs = {
        "dataset_root_text": str(dataset_root),
        "input_index_text": str(input_index),
        "artifact_root_text": str(artifact_root),
        "run_config": run_config,
        "qc_payload": asdict(qc),
        "cache_policy": args.cache_policy,
    }
    results: List[Dict[str, Any]] = []

    if args.num_workers == 1:
        iterator: Iterable[Any] = records
        if tqdm is not None and not args.no_progress:
            iterator = tqdm(records, total=len(records), desc="Retinal V2", unit="asset", dynamic_ncols=True)
        for row in iterator:
            results.append(process_asset(row, **kwargs))
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_row = {
                executor.submit(process_asset, row, **kwargs): row for row in records
            }
            iterator = as_completed(future_to_row)
            if tqdm is not None and not args.no_progress:
                iterator = tqdm(
                    iterator,
                    total=len(future_to_row),
                    desc="Retinal V2",
                    unit="asset",
                    dynamic_ncols=True,
                )
            for future in iterator:
                row = future_to_row[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # process_asset normally catches its own errors
                    results.append(
                        {
                            "status": "failed",
                            "person_id": str(row.get("person_id", "")),
                            "asset_id": str(row.get("asset_id", row.get("asset_key", ""))),
                            "asset_key": str(row.get("asset_key", "")),
                            "modality": str(row.get("modality", "")),
                            "protocol": str(row.get("protocol", "")),
                            "filepath": str(row.get("filepath", "")),
                            "failure_stage": "worker_boundary",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(limit=10),
                            "input_index_config_hash": str(
                                row.get("input_index_config_hash", "")
                            ),
                            "input_index_source_fingerprint": str(
                                row.get("input_index_source_fingerprint", "")
                            ),
                            "artifact_config_hash": run_config["config_hash"],
                        }
                    )

    results.sort(
        key=lambda item: (
            str(item.get("person_id", "")),
            str(item.get("modality", "")),
            str(item.get("protocol", "")),
            str(item.get("asset_key", "")),
        )
    )
    failures = [item for item in results if item.get("status") == "failed"]
    successes = [item for item in results if item.get("status") in {"processed", "cached"}]
    index = pd.DataFrame(successes, columns=INDEX_COLUMNS)
    if not index.empty:
        index = index.sort_values(
            ["person_id", "modality", "protocol", "asset_key"], kind="mergesort"
        )
    _atomic_write_csv(out_dir / "retinal_preprocessed_index.csv", index)
    _write_json(out_dir / "retinal_preprocess_failures.json", failures)

    n_processed = sum(item.get("status") == "processed" for item in results)
    n_cached = sum(item.get("status") == "cached" for item in results)
    successful_artifact_bytes = int(
        sum(int(item.get("artifact_size_bytes") or 0) for item in successes)
    )
    schema = {
        "artifact_config": run_config,
        "input_index": str(input_index),
        "input_kind": input_index.name,
        "input_index_config_hashes": sorted(
            set(assets["input_index_config_hash"].astype(str))
        ),
        "input_index_source_fingerprints": sorted(
            set(assets["input_index_source_fingerprint"].astype(str))
        ),
        "artifact_config_hash": run_config["config_hash"],
        "artifact_root": str(artifact_root),
        "model_views": {
            "key": "model_views",
            "dtype": "float32",
            "shape": ["n_views", 3, args.target_size, args.target_size],
            "range": [0.0, 1.0],
            "padding": "symmetric zero padding after aspect-preserving bicubic resize",
            "post_resize_clip": (
                "clip Pillow bicubic float overshoot to [0,1]; preclip min/max and counts are in each view_geometry"
            ),
            "normalization": (
                "recorded per-asset percentile transform; shared across RGB channels; no participant z-score"
            ),
        },
        "masks": {
            "model_pixel_mask": "valid decoded/non-padding support after nearest-neighbor mask resize",
            "view_quality_mask": "view passes conservative finite/range/saturation QC",
            "flio_photon_valid_mask": "positive summed photon weight exists for mean-bin auxiliary",
        },
        "products": {
            "photography": "one or all decoded 2-D frames; RGB color balance retained with shared transform",
            "oct_volume": "configurable central slice stack plus 3-channel adjacent-slice 2.5-D view",
            "octa_flow_cube": "same volume policy as OCT",
            "octa_enface": "independent 2-D slab asset; projection/slab metadata retained",
            "flio_decay_cube": {
                "default_storage": (
                    "full decoded stored-value cube plus validity mask in a lossless, checksummed zlib archive"
                ),
                "archive_key": "flio_decay_cube_lossless_archive",
                "decoder": "decode_flio_decay_cube() in this module",
                "wavelengths": "short/long remain separate indexed artifacts and retain slab_or_wavelength",
                "auxiliaries": ["intensity", "photon_weighted_mean_bin"],
                "mean_bin_warning": (
                    "Mean-bin is a zero-based frame index. It is not time calibrated, IRF corrected, fitted, or a physical lifetime."
                ),
                "storage_warning": (
                    "A typical 1024x256x256 uint16 cube is ~128 MiB uncompressed. "
                    "All indexed FLIO cubes can approach one TiB before compression; "
                    "zlib ratio is data dependent. Run retinal_default_model_view.csv "
                    "with --limit first and inspect artifact_size_bytes."
                ),
                "estimated_selected_cube_bytes_if_uint16_uncompressed": estimated_flio_uint16_bytes,
                "estimated_selected_cube_gib_if_uint16_uncompressed": float(
                    estimated_flio_uint16_bytes / (1024 ** 3)
                ),
            },
        },
        "dicom_pipeline": {
            "display_products": [
                "decode stored values",
                "modality LUT/rescale if declared",
                "VOI LUT/first window if declared and configured",
                "MONOCHROME1 inversion",
            ],
            "flio_quantitative_auxiliaries": [
                "decode and preserve stored decay values",
                "modality LUT/rescale if declared",
                "exclude VOI and photometric presentation inversion from photon weights",
            ],
            "uncertainty": "all warnings are saved in artifact and compact index; failures include partial audit",
        },
        "spacing_and_protocol": (
            "manifest and DICOM spacing, laterality, protocol, slab/wavelength, device, and surfaces are retained as metadata"
        ),
        "cache": (
            "artifact directory includes config hash; reuse additionally requires source size/mtime and input-row fingerprint"
        ),
        "atomic_writes": True,
        "n_input_rows": int(len(source_frame)),
        "n_assets_selected_for_run": int(len(assets)),
        "n_processed": int(n_processed),
        "n_cached": int(n_cached),
        "n_failed": int(len(failures)),
        "successful_artifact_bytes_written_or_cached": successful_artifact_bytes,
        "successful_artifact_gib_written_or_cached": float(
            successful_artifact_bytes / (1024 ** 3)
        ),
        "output_files": {
            "index": str(out_dir / "retinal_preprocessed_index.csv"),
            "failures": str(out_dir / "retinal_preprocess_failures.json"),
            "schema": str(out_dir / "retinal_preprocess_schema.json"),
            "artifact_config": str(out_dir / "artifact_config.json"),
        },
        "notes": [
            "MoCA/CVD labels are not read or used by this task-independent stage.",
            "Do not fit learned intensity standardization before the participant-level train split.",
            "Full OCT/OCTA volumes are not copied; the selected multi-center stack is explicit and configurable.",
            "Full FLIO decay cubes are preserved by default because a display projection is not a substitute for decay data.",
        ],
    }
    _write_json(out_dir / "retinal_preprocess_schema.json", schema)

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "artifact_config_hash": run_config["config_hash"],
                "n_input_rows": len(source_frame),
                "n_assets_selected_for_run": len(assets),
                "n_processed": n_processed,
                "n_cached": n_cached,
                "n_failed": len(failures),
                "flio_cube_policy": args.flio_cube_policy,
                "estimated_flio_uint16_uncompressed_gib": (
                    estimated_flio_uint16_bytes / (1024 ** 3)
                ),
                "successful_artifact_gib": successful_artifact_bytes / (1024 ** 3),
            },
            indent=2,
            allow_nan=False,
        )
    )
    if failures[:5]:
        print(
            "first failures:",
            [
                {
                    key: item.get(key)
                    for key in ("person_id", "asset_key", "failure_stage", "error")
                }
                for item in failures[:5]
            ],
        )


if __name__ == "__main__":
    main()
