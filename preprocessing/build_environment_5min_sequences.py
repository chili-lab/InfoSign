#!/usr/bin/env python3
"""Build canonical 5-minute environmental-sensor sequences.

This is the V2, task-independent replacement for
``build_environment_hourly_sequences.py``.  It keeps the 5-minute resolution
used by LANISTR, but aligns every longitudinal modality to the same absolute
participant grid defined in :mod:`preprocessing.preprocessing_utils` instead of anchoring
each source independently at its first observation.

The default per-participant artifact contains 10 days x 288 five-minute slots
per day (2,880 slots total):

* ``environment``: raw-scale mean/std/missing-rate features for 21 channels,
  raw row count, and three local/window-time features;
* ``environment_channel_observed_mask``: one mask per slot and source channel;
* ``environment_channel_rejected_mask``: a slot/channel audit mask for
  non-missing source values rejected before aggregation;
* ``environment_channel_saturation_mask``: retained right-censored sensor
  saturation events (currently the PM maximum reportable code);
* ``environment_channel_outside_specified_mask``: retained values outside a
  manufacturer's accuracy-specified range;
* ``environment_slot_mask``: true only when at least one finite,
  float32-representable, hard-domain-valid primary sensor value is present (a
  timestamped row or the device-internal ``inttemp`` alone does not make a
  slot observed);
* ``environment_row_count``: number of rows with valid timestamps in each slot.

No participant-level or whole-dataset normalization is performed here.  Any
imputation/scaling parameters must be fitted on a downstream training split.
No value is clipped. Documented impossible/sentinel values are treated as
missing; documented but merely out-of-specification values are retained and
flagged. Per-channel raw and aggregation-usable ranges are recorded.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
try:
    import torch
except ImportError:  # Pure aggregation helpers/tests do not require serialization.
    torch = None  # type: ignore

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None

try:
    from .preprocessing_utils import (
        ParticipantAnchor,
        TimeGridSpec,
        artifact_config,
        as_person_id,
        canonical_json_hash,
        in_window,
        load_participant_anchors,
        local_time_features,
        read_anchor_overrides,
        slot_indices,
        source_file_metadata,
    )
except ImportError:  # Allow ``python preprocessing/build_environment_5min_sequences.py``.
    from preprocessing_utils import (  # type: ignore
        ParticipantAnchor,
        TimeGridSpec,
        artifact_config,
        as_person_id,
        canonical_json_hash,
        in_window,
        load_participant_anchors,
        local_time_features,
        read_anchor_overrides,
        slot_indices,
        source_file_metadata,
    )


SENSOR_COLUMNS: Tuple[str, ...] = (
    "lch0",
    "lch1",
    "lch2",
    "lch3",
    "lch6",
    "lch7",
    "lch8",
    "lch9",
    "lch10",
    "lch11",
    "pm1",
    "pm2.5",
    "pm4",
    "pm10",
    "hum",
    "temp",
    "voc",
    "nox",
    "screen",
    "ff",
    "inttemp",
)
TECHNICAL_SENSOR_COLUMNS: Tuple[str, ...] = ("inttemp",)
PRIMARY_SENSOR_COLUMNS: Tuple[str, ...] = tuple(
    column for column in SENSOR_COLUMNS if column not in TECHNICAL_SENSOR_COLUMNS
)
BUILDER_VERSION = "environment-5min-4"

# Task-independent sensor validity rules from the AI-READI v3 environment
# data dictionary and the Sensirion SEN55 interface contract.  ``hard_*``
# bounds define values that may participate in aggregation.  ``specified_*``
# bounds are audit-only: a value can be a genuine sensor output outside the
# manufacturer's accuracy-specified range (notably high PM) and must not be
# silently clipped or discarded.  PM 6553.4 ug/m3 is the largest reportable
# value (0xFFFE / 10) and is retained as a right-censored saturation event;
# 6553.5 (0xFFFF / 10, "unknown") falls outside the hard maximum and is
# treated as missing.
PM_COLUMNS: Tuple[str, ...] = ("pm1", "pm2.5", "pm4", "pm10")
LIGHT_COLUMNS: Tuple[str, ...] = (
    "lch0",
    "lch1",
    "lch2",
    "lch3",
    "lch6",
    "lch7",
    "lch8",
    "lch9",
    "lch10",
    "lch11",
)

SENSOR_VALUE_RULES: Dict[str, Dict[str, Any]] = {
    **{
        column: {
            "hard_min": 0.0,
            "hard_max": 1.0,
            "specified_min": 0.0,
            "specified_max": 1.0,
            "units": "relative_intensity",
        }
        for column in LIGHT_COLUMNS
    },
    **{
        column: {
            "hard_min": 0.0,
            "hard_max": 6553.4,
            "specified_min": 0.0,
            "specified_max": 1000.0,
            "sentinel_values": [6553.5],
            "sentinel_atol": 1e-6,
            "saturation_values": [6553.4],
            "saturation_atol": 1e-6,
            "units": "ug/m3",
        }
        for column in PM_COLUMNS
    },
    "hum": {
        "hard_min": 0.0,
        "hard_max": 100.0,
        "specified_min": 0.0,
        "specified_max": 100.0,
        "units": "percent_relative_humidity",
    },
    "temp": {
        "hard_min": -10.0,
        "hard_max": 50.0,
        "specified_min": -10.0,
        "specified_max": 50.0,
        "units": "degrees_C",
    },
    # AI-READI retains startup zero/NaN values even though the SEN55's
    # stable index specification is commonly written as 1..500.
    "voc": {
        "hard_min": 0.0,
        "hard_max": 500.0,
        "specified_min": 0.0,
        "specified_max": 500.0,
        "units": "VOC_index_points",
    },
    "nox": {
        "hard_min": 0.0,
        "hard_max": 500.0,
        "specified_min": 0.0,
        "specified_max": 500.0,
        "units": "NOx_index_points",
    },
    "screen": {
        "allowed_values": [0.0, 1.0],
        "specified_values": [0.0, 1.0],
        "units": "boolean_device_screen_state",
    },
    "ff": {
        "hard_min": 0.0,
        "hard_max": 2000.0,
        "specified_min": 0.0,
        "specified_max": 2000.0,
        "units": "Hz",
    },
    # The AI-READI v3 table has a malformed upper-bound entry for inttemp.
    # Keep finite float32 values but expose this device-internal channel only
    # as a technical/QC variable downstream, not a primary exposure feature.
    "inttemp": {
        "hard_min": None,
        "hard_max": None,
        "specified_min": None,
        "specified_max": None,
        "units": "degrees_C_device_internal",
    },
}

TIME_FEATURE_NAMES: Tuple[str, ...] = (
    "time__day_sin",
    "time__day_cos",
    "time__day_index_norm",
)

MANIFEST_METADATA_COLUMNS: Tuple[str, ...] = (
    "modality",
    "sensor_location",
    "number_of_observations",
    "sensor_sampling_extent_in_days",
    "sensor_id",
    "manufacturer",
    "manufacturers_model_name",
)

INDEX_COLUMNS: Tuple[str, ...] = (
    "person_id",
    "artifact_path",
    "metadata_path",
    "config_hash",
    "anchor_utc",
    "timezone",
    "shared_grid_hash",
    "n_raw_rows",
    "n_valid_timestamp_rows",
    "n_invalid_timestamp_rows",
    "n_rows_before_window",
    "n_rows_in_window",
    "n_rows_after_window",
    "n_rows_in_window_with_sensor_values",
    "n_observed_slots",
    "slot_coverage_fraction",
    "n_observed_channel_slots",
    "n_missing_source_channels",
    "n_rejected_sensor_values_in_window",
    "n_hard_invalid_sensor_values_in_window",
    "n_saturated_sensor_values_in_window",
    "n_outside_specified_sensor_values_in_window",
    "qc_flags",
    "status",
)


def build_feature_names(sensor_columns: Sequence[str] = SENSOR_COLUMNS) -> List[str]:
    """Return the stable column order of the primary environment tensor."""
    names: List[str] = []
    for column in sensor_columns:
        safe = str(column).replace(".", "_")
        names.extend(
            [
                f"env__{safe}__mean",
                f"env__{safe}__std",
                f"env__{safe}__missing_rate",
            ]
        )
    names.append("env__row_count")
    names.extend(TIME_FEATURE_NAMES)
    return names


def default_model_excluded_feature_names(
    technical_sensor_columns: Sequence[str] = TECHNICAL_SENSOR_COLUMNS,
) -> List[str]:
    """Technical/device-health features retained for QC but excluded by default."""
    names: List[str] = []
    for column in technical_sensor_columns:
        safe = str(column).replace(".", "_")
        names.extend(
            [
                f"env__{safe}__mean",
                f"env__{safe}__std",
                f"env__{safe}__missing_rate",
            ]
        )
    return names


def resolve_path(dataset_root: str | Path, manifest_path: Any) -> Path:
    """Resolve AI-READI's slash-prefixed, dataset-root-relative paths."""
    root = Path(dataset_root)
    text = str(manifest_path).strip()
    if not text or text.lower() == "nan":
        raise ValueError("Empty env_sensor_filepath in environment manifest")
    candidate = Path(text)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    return root / text.lstrip("/\\")


def _find_csv_header(path: Path) -> Tuple[int, List[str]]:
    """Find the ENV table header, including when metadata comments it out."""
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for line_number, line in enumerate(handle):
            cleaned = line.lstrip().lstrip("#").strip().lstrip("\ufeff")
            if not cleaned.lower().startswith("ts,"):
                continue
            columns = [str(value).strip() for value in next(csv.reader([cleaned]))]
            if not columns or columns[0].lower() != "ts":
                continue
            columns[0] = "ts"
            if len(columns) != len(set(columns)):
                raise ValueError(f"Duplicate ENV CSV columns in {path}: {columns}")
            return line_number, columns
    raise ValueError(f"Could not find a 'ts,...' table header in {path}")


def read_env_csv(path: str | Path) -> pd.DataFrame:
    """Read one ENV CSV without assuming whether its header is commented."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    header_line, columns = _find_csv_header(path)
    frame = pd.read_csv(
        path,
        skiprows=header_line + 1,
        names=columns,
        header=None,
        comment="#",
        skipinitialspace=True,
        low_memory=False,
        encoding_errors="replace",
    )
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame[
        [column for column in frame.columns if column and not column.lower().startswith("unnamed")]
    ]
    if "ts" not in frame.columns:
        raise ValueError(f"No 'ts' column found in {path}")
    return frame


def _parse_utc_series(values: pd.Series) -> pd.Series:
    """Parse a source timestamp vector as timezone-aware UTC."""
    return pd.to_datetime(values, errors="coerce", utc=True)


def _finite_range(values: np.ndarray) -> Tuple[float | None, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    return float(np.min(finite)), float(np.max(finite))


def _numeric_array(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.astype("float64").to_numpy()


def _matches_rule_values(
    numeric: np.ndarray,
    values: Sequence[float] | None,
    *,
    atol: float = 0.0,
) -> np.ndarray:
    """Return a vectorized exact/tolerance match for documented codes."""
    matched = np.zeros(numeric.shape, dtype=bool)
    for value in values or ():
        matched |= np.isclose(
            numeric,
            float(value),
            rtol=0.0,
            atol=float(atol),
            equal_nan=False,
        )
    return matched


def _sensor_value_masks(
    column: str,
    numeric: np.ndarray,
    float32_representable: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Classify numeric values using the documented per-channel policy.

    ``aggregation_valid`` is mutually exclusive with ``sentinel`` and
    ``hard_domain_invalid``.  ``saturation`` and ``outside_specified`` are
    audit overlays on aggregation-valid values and therefore do not remove or
    clip observations.
    """
    rule = SENSOR_VALUE_RULES.get(str(column), {})
    sentinel = float32_representable & _matches_rule_values(
        numeric,
        rule.get("sentinel_values"),
        atol=float(rule.get("sentinel_atol", 0.0)),
    )
    candidate = float32_representable & (~sentinel)

    hard_valid = candidate.copy()
    allowed_values = rule.get("allowed_values")
    if allowed_values is not None:
        hard_valid &= _matches_rule_values(
            numeric,
            allowed_values,
            atol=float(rule.get("allowed_values_atol", 1e-6)),
        )
    hard_min = rule.get("hard_min")
    hard_max = rule.get("hard_max")
    if hard_min is not None:
        hard_valid &= numeric >= float(hard_min)
    if hard_max is not None:
        hard_valid &= numeric <= float(hard_max)
    hard_domain_invalid = candidate & (~hard_valid)

    specified_valid = hard_valid.copy()
    specified_values = rule.get("specified_values")
    if specified_values is not None:
        specified_valid &= _matches_rule_values(
            numeric,
            specified_values,
            atol=float(rule.get("specified_values_atol", 1e-6)),
        )
    specified_min = rule.get("specified_min")
    specified_max = rule.get("specified_max")
    if specified_min is not None:
        specified_valid &= numeric >= float(specified_min)
    if specified_max is not None:
        specified_valid &= numeric <= float(specified_max)

    saturation = hard_valid & _matches_rule_values(
        numeric,
        rule.get("saturation_values"),
        atol=float(rule.get("saturation_atol", 0.0)),
    )
    return {
        "aggregation_valid": hard_valid,
        "sentinel": sentinel,
        "hard_domain_invalid": hard_domain_invalid,
        "saturation": saturation,
        "outside_specified": hard_valid & (~specified_valid),
    }


def aggregate_environment_5min(
    frame: pd.DataFrame,
    anchor: ParticipantAnchor,
    spec: TimeGridSpec,
    sensor_columns: Sequence[str] = SENSOR_COLUMNS,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, np.ndarray],
    Dict[str, Any],
]:
    """Aggregate a raw ENV dataframe onto a participant's shared UTC grid.

    Missing-rate is defined within each slot as ``1 - valid_count/row_count``,
    where valid means finite, float32-representable, non-sentinel, and inside
    the documented hard domain. Empty slots and source-absent channels have
    missing-rate 1. The standard deviation is population std (``ddof=0``); a
    singleton has std 0. No clipping, normalization, or interpolation occurs.
    Values outside a merely accuracy-specified range remain in the aggregate
    and are surfaced through QC masks and metadata.
    """
    if "ts" not in frame.columns:
        raise ValueError("Environment dataframe is missing required 'ts' column")

    n_rows = int(len(frame))
    timestamps = _parse_utc_series(frame["ts"])
    valid_timestamp = timestamps.notna()
    slot_index = slot_indices(timestamps, anchor.utc_timestamp(), spec)
    window_rows = in_window(slot_index, spec)

    valid_timestamp_array = valid_timestamp.to_numpy(dtype=bool)
    window_rows_array = window_rows.to_numpy(dtype=bool)
    slots = slot_index.fillna(-1).to_numpy(dtype=np.int64)
    before_rows = valid_timestamp & slot_index.notna() & (slot_index < 0)
    after_rows = valid_timestamp & slot_index.notna() & (slot_index >= spec.seq_len)

    row_count = np.bincount(
        slots[window_rows_array], minlength=spec.seq_len
    )[: spec.seq_len].astype(np.int64, copy=False)

    n_channels = len(sensor_columns)
    n_features = n_channels * 3 + 1 + len(TIME_FEATURE_NAMES)
    values_out = np.zeros((spec.seq_len, n_features), dtype=np.float64)
    channel_observed = np.zeros((spec.seq_len, n_channels), dtype=bool)
    channel_rejected = np.zeros((spec.seq_len, n_channels), dtype=bool)
    channel_saturation = np.zeros((spec.seq_len, n_channels), dtype=bool)
    channel_outside_specified = np.zeros((spec.seq_len, n_channels), dtype=bool)
    valid_count_matrix = np.zeros((spec.seq_len, n_channels), dtype=np.int64)
    values_out[:, 2 : n_channels * 3 : 3] = 1.0

    missing_source_channels: List[str] = []
    channels_without_window_values: List[str] = []
    constant_channels_in_window: List[str] = []
    channels_with_invalid_numeric_values: List[str] = []
    channels_with_sentinel_values: List[str] = []
    channels_with_hard_invalid_values: List[str] = []
    channels_with_saturation_values: List[str] = []
    channels_with_outside_specified_values: List[str] = []
    aggregation_nonfinite_channels: List[str] = []
    sensor_ranges: Dict[str, Dict[str, Any]] = {}
    any_primary_aggregation_valid_sensor_row = np.zeros(n_rows, dtype=bool)

    for channel_number, column in enumerate(sensor_columns):
        base = channel_number * 3
        if column not in frame.columns:
            missing_source_channels.append(str(column))
            channels_without_window_values.append(str(column))
            sensor_ranges[str(column)] = {
                "source_column_present": False,
                "value_rule": SENSOR_VALUE_RULES.get(str(column), {}),
                "n_finite_numeric_all_rows": 0,
                "n_valid_float32_all_rows": 0,
                "n_valid_float32_in_window": 0,
                "n_source_missing_all_rows": n_rows,
                "n_source_missing_in_window": int(np.count_nonzero(window_rows_array)),
                "n_nonfinite_nonnumeric_or_float32_unrepresentable_all_rows": 0,
                "n_nonfinite_nonnumeric_or_float32_unrepresentable_in_window": 0,
                "n_sentinel_all_rows": 0,
                "n_hard_domain_invalid_all_rows": 0,
                "n_hard_invalid_all_rows": 0,
                "n_aggregation_valid_all_rows": 0,
                "n_sentinel_in_window": 0,
                "n_hard_domain_invalid_in_window": 0,
                "n_hard_invalid_in_window": 0,
                "n_aggregation_valid_in_window": 0,
                "n_rejected_all_rows": 0,
                "n_rejected_in_window": 0,
                "n_saturated_all_rows": 0,
                "n_saturated_in_window": 0,
                "n_outside_specified_all_rows": 0,
                "n_outside_specified_in_window": 0,
                "partition_count_all_rows": n_rows,
                "partition_invariant_holds": True,
                "partition_count_in_window": int(np.count_nonzero(window_rows_array)),
                "partition_invariant_in_window_holds": True,
                "all_rows_min": None,
                "all_rows_max": None,
                "raw_in_window_min": None,
                "raw_in_window_max": None,
                "aggregation_valid_all_rows_min": None,
                "aggregation_valid_all_rows_max": None,
                "in_window_min": None,
                "in_window_max": None,
                "constant_in_window": False,
            }
            continue

        source_missing = frame[column].isna().to_numpy(dtype=bool)
        numeric = _numeric_array(frame[column])
        finite_numeric = np.isfinite(numeric)
        float32_representable = finite_numeric & (
            np.abs(numeric) <= np.finfo(np.float32).max
        )
        masks = _sensor_value_masks(str(column), numeric, float32_representable)
        aggregation_valid = masks["aggregation_valid"]
        sentinel = masks["sentinel"]
        hard_domain_invalid = masks["hard_domain_invalid"]
        hard_invalid = sentinel | hard_domain_invalid
        if str(column) not in TECHNICAL_SENSOR_COLUMNS:
            any_primary_aggregation_valid_sensor_row |= aggregation_valid
        valid_in_window = aggregation_valid & window_rows_array
        channel_slots = slots[valid_in_window]
        channel_values = numeric[valid_in_window]

        invalid_numeric = (~float32_representable) & (~source_missing)
        rejected = invalid_numeric | hard_invalid
        if np.count_nonzero(invalid_numeric) > 0:
            channels_with_invalid_numeric_values.append(str(column))
        if np.count_nonzero(sentinel) > 0:
            channels_with_sentinel_values.append(str(column))
        if np.count_nonzero(hard_domain_invalid) > 0:
            channels_with_hard_invalid_values.append(str(column))
        if np.count_nonzero(masks["saturation"]) > 0:
            channels_with_saturation_values.append(str(column))
        if np.count_nonzero(masks["outside_specified"]) > 0:
            channels_with_outside_specified_values.append(str(column))

        valid_counts = np.bincount(
            channel_slots, minlength=spec.seq_len
        )[: spec.seq_len].astype(np.int64, copy=False)
        valid_count_matrix[:, channel_number] = valid_counts
        observed = valid_counts > 0
        channel_observed[:, channel_number] = observed

        for source_mask, destination in (
            (rejected, channel_rejected),
            (masks["saturation"], channel_saturation),
            (masks["outside_specified"], channel_outside_specified),
        ):
            source_in_window = source_mask & window_rows_array
            if np.any(source_in_window):
                destination[:, channel_number] = (
                    np.bincount(
                        slots[source_in_window], minlength=spec.seq_len
                    )[: spec.seq_len]
                    > 0
                )

        if channel_values.size:
            sums = np.bincount(
                channel_slots, weights=channel_values, minlength=spec.seq_len
            )[: spec.seq_len]
            with np.errstate(over="ignore", invalid="ignore"):
                square_sums = np.bincount(
                    channel_slots,
                    weights=channel_values * channel_values,
                    minlength=spec.seq_len,
                )[: spec.seq_len]
                means = np.divide(
                    sums,
                    valid_counts,
                    out=np.zeros(spec.seq_len, dtype=np.float64),
                    where=observed,
                )
                second_moments = np.divide(
                    square_sums,
                    valid_counts,
                    out=np.zeros(spec.seq_len, dtype=np.float64),
                    where=observed,
                )
                variances = np.maximum(second_moments - means * means, 0.0)
                stds = np.sqrt(variances)

            invalid_aggregates = observed & (~np.isfinite(means) | ~np.isfinite(stds))
            if np.any(invalid_aggregates):
                aggregation_nonfinite_channels.append(str(column))
                means[invalid_aggregates] = 0.0
                stds[invalid_aggregates] = 0.0
            values_out[:, base] = means
            values_out[:, base + 1] = stds
        else:
            channels_without_window_values.append(str(column))

        rows_present = row_count > 0
        values_out[rows_present, base + 2] = (
            1.0 - valid_counts[rows_present] / row_count[rows_present]
        )

        all_min, all_max = _finite_range(numeric)
        raw_window_min, raw_window_max = _finite_range(numeric[window_rows_array])
        usable_all_min, usable_all_max = _finite_range(numeric[aggregation_valid])
        window_min, window_max = _finite_range(channel_values)
        constant = bool(
            channel_values.size > 1
            and window_min is not None
            and window_max is not None
            and window_min == window_max
        )
        if constant:
            constant_channels_in_window.append(str(column))
        partition_count = int(
            np.count_nonzero(source_missing)
            + np.count_nonzero(invalid_numeric)
            + np.count_nonzero(sentinel)
            + np.count_nonzero(hard_domain_invalid)
            + np.count_nonzero(aggregation_valid)
        )
        partition_count_in_window = int(
            np.count_nonzero(source_missing & window_rows_array)
            + np.count_nonzero(invalid_numeric & window_rows_array)
            + np.count_nonzero(sentinel & window_rows_array)
            + np.count_nonzero(hard_domain_invalid & window_rows_array)
            + np.count_nonzero(aggregation_valid & window_rows_array)
        )
        sensor_ranges[str(column)] = {
            "source_column_present": True,
            "value_rule": SENSOR_VALUE_RULES.get(str(column), {}),
            "n_finite_numeric_all_rows": int(np.count_nonzero(finite_numeric)),
            "n_valid_float32_all_rows": int(np.count_nonzero(float32_representable)),
            "n_valid_float32_in_window": int(
                np.count_nonzero(float32_representable & window_rows_array)
            ),
            "n_source_missing_all_rows": int(np.count_nonzero(source_missing)),
            "n_source_missing_in_window": int(
                np.count_nonzero(source_missing & window_rows_array)
            ),
            "n_nonfinite_nonnumeric_or_float32_unrepresentable_all_rows": int(
                np.count_nonzero(invalid_numeric)
            ),
            "n_nonfinite_nonnumeric_or_float32_unrepresentable_in_window": int(
                np.count_nonzero(invalid_numeric & window_rows_array)
            ),
            "n_sentinel_all_rows": int(np.count_nonzero(sentinel)),
            "n_hard_domain_invalid_all_rows": int(
                np.count_nonzero(hard_domain_invalid)
            ),
            "n_hard_invalid_all_rows": int(np.count_nonzero(hard_invalid)),
            "n_aggregation_valid_all_rows": int(
                np.count_nonzero(aggregation_valid)
            ),
            "n_sentinel_in_window": int(
                np.count_nonzero(sentinel & window_rows_array)
            ),
            "n_hard_domain_invalid_in_window": int(
                np.count_nonzero(hard_domain_invalid & window_rows_array)
            ),
            "n_hard_invalid_in_window": int(
                np.count_nonzero(hard_invalid & window_rows_array)
            ),
            "n_aggregation_valid_in_window": int(channel_values.size),
            "n_rejected_all_rows": int(np.count_nonzero(rejected)),
            "n_rejected_in_window": int(
                np.count_nonzero(rejected & window_rows_array)
            ),
            "n_saturated_all_rows": int(np.count_nonzero(masks["saturation"])),
            "n_saturated_in_window": int(
                np.count_nonzero(masks["saturation"] & window_rows_array)
            ),
            "n_outside_specified_all_rows": int(
                np.count_nonzero(masks["outside_specified"])
            ),
            "n_outside_specified_in_window": int(
                np.count_nonzero(masks["outside_specified"] & window_rows_array)
            ),
            "partition_count_all_rows": partition_count,
            "partition_invariant_holds": partition_count == n_rows,
            "partition_count_in_window": partition_count_in_window,
            "partition_invariant_in_window_holds": (
                partition_count_in_window == int(np.count_nonzero(window_rows_array))
            ),
            "all_rows_min": all_min,
            "all_rows_max": all_max,
            "raw_in_window_min": raw_window_min,
            "raw_in_window_max": raw_window_max,
            "aggregation_valid_all_rows_min": usable_all_min,
            "aggregation_valid_all_rows_max": usable_all_max,
            "in_window_min": window_min,
            "in_window_max": window_max,
            "constant_in_window": constant,
        }

    sensor_value_qc_counts = {
        key: int(sum(int(details[key]) for details in sensor_ranges.values()))
        for key in (
            "n_nonfinite_nonnumeric_or_float32_unrepresentable_all_rows",
            "n_nonfinite_nonnumeric_or_float32_unrepresentable_in_window",
            "n_sentinel_all_rows",
            "n_sentinel_in_window",
            "n_hard_domain_invalid_all_rows",
            "n_hard_domain_invalid_in_window",
            "n_hard_invalid_all_rows",
            "n_hard_invalid_in_window",
            "n_rejected_all_rows",
            "n_rejected_in_window",
            "n_saturated_all_rows",
            "n_saturated_in_window",
            "n_outside_specified_all_rows",
            "n_outside_specified_in_window",
        )
    }
    channels_with_rejected_values_in_window = [
        str(column)
        for column, details in sensor_ranges.items()
        if int(details["n_rejected_in_window"]) > 0
    ]
    channels_with_saturation_values_in_window = [
        str(column)
        for column, details in sensor_ranges.items()
        if int(details["n_saturated_in_window"]) > 0
    ]
    channels_with_outside_specified_values_in_window = [
        str(column)
        for column, details in sensor_ranges.items()
        if int(details["n_outside_specified_in_window"]) > 0
    ]
    primary_channel_indices = [
        index
        for index, column in enumerate(sensor_columns)
        if str(column) not in TECHNICAL_SENSOR_COLUMNS
    ]
    slot_mask = (
        channel_observed[:, primary_channel_indices].any(axis=1)
        if primary_channel_indices
        else np.zeros(spec.seq_len, dtype=bool)
    )
    values_out[:, n_channels * 3] = row_count.astype(np.float64)
    time_values = local_time_features(anchor, spec)
    for offset, feature_name in enumerate(TIME_FEATURE_NAMES, start=n_channels * 3 + 1):
        values_out[:, offset] = time_values[feature_name]
    utc_offsets = time_values["time__utc_offset_hours"]
    offset_transition_slots = (
        np.flatnonzero(np.diff(utc_offsets) != 0).astype(int) + 1
    ).tolist()

    with np.errstate(over="ignore", invalid="ignore"):
        tensor = values_out.astype(np.float32)
    nonfinite_tensor = ~np.isfinite(tensor)
    n_nonfinite_tensor_values = int(np.count_nonzero(nonfinite_tensor))
    if n_nonfinite_tensor_values:
        tensor[nonfinite_tensor] = 0.0

    valid_timestamps_only = timestamps[valid_timestamp]
    n_duplicate_excess = int(valid_timestamps_only.duplicated().sum())
    timestamp_not_monotonic = bool(
        len(valid_timestamps_only) > 1 and not valid_timestamps_only.is_monotonic_increasing
    )
    observed_row_counts = row_count[row_count > 0]
    qc_flags = {
        "has_invalid_timestamps": bool(np.any(~valid_timestamp_array)),
        "has_rows_before_window": bool(before_rows.any()),
        "has_rows_after_window": bool(after_rows.any()),
        "has_duplicate_timestamps": n_duplicate_excess > 0,
        "timestamps_not_monotonic": timestamp_not_monotonic,
        "has_missing_source_channels": bool(missing_source_channels),
        "has_channels_without_valid_window_values": bool(channels_without_window_values),
        "has_nonfinite_nonnumeric_or_float32_unrepresentable_sensor_values": bool(
            channels_with_invalid_numeric_values
        ),
        "has_documented_sentinel_sensor_values_any_source_row": bool(
            channels_with_sentinel_values
        ),
        "has_hard_domain_invalid_sensor_values_any_source_row": bool(
            channels_with_hard_invalid_values
        ),
        "has_sensor_saturation_values_any_source_row": bool(
            channels_with_saturation_values
        ),
        "has_values_outside_accuracy_specified_range_any_source_row": bool(
            channels_with_outside_specified_values
        ),
        "has_rejected_sensor_values_in_window": bool(
            channels_with_rejected_values_in_window
        ),
        "has_sensor_saturation_values_in_window": bool(
            channels_with_saturation_values_in_window
        ),
        "has_values_outside_accuracy_specified_range_in_window": bool(
            channels_with_outside_specified_values_in_window
        ),
        "has_constant_channels_in_window": bool(constant_channels_in_window),
        "has_nonfinite_aggregates_replaced": bool(
            aggregation_nonfinite_channels or n_nonfinite_tensor_values
        ),
        "no_valid_sensor_values_in_window": not bool(slot_mask.any()),
    }

    metadata: Dict[str, Any] = {
        "timestamp_semantics": "parsed_as_UTC",
        "anchor_utc": anchor.anchor_utc,
        "anchor_local": anchor.anchor_local,
        "anchor_strategy": anchor.strategy,
        "timezone": anchor.timezone,
        "local_time_audit": {
            "utc_offset_hours_observed": sorted(
                {float(value) for value in utc_offsets.tolist()}
            ),
            "utc_offset_transition_slots": offset_transition_slots,
            "utc_offset_in_model_tensor": False,
        },
        "grid_minutes": int(spec.grid_minutes),
        "window_days": int(spec.window_days),
        "seq_len": int(spec.seq_len),
        "window_end_utc_exclusive": (
            anchor.utc_timestamp() + pd.Timedelta(days=spec.window_days)
        ).isoformat(),
        "n_raw_rows": n_rows,
        "n_valid_timestamp_rows": int(valid_timestamp.sum()),
        "n_invalid_timestamp_rows": int((~valid_timestamp).sum()),
        "n_rows_before_window": int(before_rows.sum()),
        "n_rows_in_window": int(window_rows.sum()),
        "n_rows_after_window": int(after_rows.sum()),
        "n_rows_before_window_with_sensor_values": int(
            np.count_nonzero(
                before_rows.to_numpy(dtype=bool)
                & any_primary_aggregation_valid_sensor_row
            )
        ),
        "n_rows_in_window_with_sensor_values": int(
            np.count_nonzero(
                window_rows_array & any_primary_aggregation_valid_sensor_row
            )
        ),
        "n_rows_after_window_with_sensor_values": int(
            np.count_nonzero(
                after_rows.to_numpy(dtype=bool)
                & any_primary_aggregation_valid_sensor_row
            )
        ),
        "n_duplicate_timestamp_excess_rows": n_duplicate_excess,
        "source_timestamp_start_utc": (
            valid_timestamps_only.min().isoformat() if len(valid_timestamps_only) else None
        ),
        "source_timestamp_end_utc": (
            valid_timestamps_only.max().isoformat() if len(valid_timestamps_only) else None
        ),
        "n_observed_slots": int(slot_mask.sum()),
        "slot_coverage_fraction": float(slot_mask.mean()) if spec.seq_len else 0.0,
        "n_slots_with_timestamped_rows": int(np.count_nonzero(row_count)),
        "n_observed_channel_slots": int(channel_observed.sum()),
        "n_valid_values_by_channel": {
            str(column): int(valid_count_matrix[:, i].sum())
            for i, column in enumerate(sensor_columns)
        },
        "row_count_min_observed_slot": (
            int(observed_row_counts.min()) if observed_row_counts.size else 0
        ),
        "row_count_max_observed_slot": (
            int(observed_row_counts.max()) if observed_row_counts.size else 0
        ),
        "missing_source_channels": missing_source_channels,
        "channels_without_valid_window_values": channels_without_window_values,
        "channels_with_invalid_numeric_values_all_rows": (
            channels_with_invalid_numeric_values
        ),
        "channels_with_sentinel_values_all_rows": channels_with_sentinel_values,
        "channels_with_hard_domain_invalid_values_all_rows": (
            channels_with_hard_invalid_values
        ),
        "channels_with_saturation_values_all_rows": channels_with_saturation_values,
        "channels_with_outside_specified_values_all_rows": (
            channels_with_outside_specified_values
        ),
        "channels_with_rejected_values_in_window": (
            channels_with_rejected_values_in_window
        ),
        "channels_with_saturation_values_in_window": (
            channels_with_saturation_values_in_window
        ),
        "channels_with_outside_specified_values_in_window": (
            channels_with_outside_specified_values_in_window
        ),
        "constant_channels_in_window": constant_channels_in_window,
        "aggregation_nonfinite_channels": aggregation_nonfinite_channels,
        "n_nonfinite_tensor_values_replaced": n_nonfinite_tensor_values,
        "sensor_observed_ranges": sensor_ranges,
        "sensor_value_qc_counts": sensor_value_qc_counts,
        "qc_flags": qc_flags,
        "qc_policy": {
            "source_missing_or_nonnumeric_action": "exclude_as_missing",
            "documented_sentinel_action": "exclude_as_missing",
            "hard_domain_invalid_action": "exclude_as_missing",
            "saturation_action": "retain_and_flag_as_right_censored",
            "outside_accuracy_specified_range_action": "retain_and_flag",
            "clipping": "none",
            "interpolation": "none",
            "normalization": "none_at_artifact_build_stage",
        },
    }
    qc_masks = {
        "rejected": channel_rejected,
        "saturation": channel_saturation,
        "outside_specified": channel_outside_specified,
    }
    return tensor, slot_mask, channel_observed, row_count, qc_masks, metadata


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
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


def _atomic_json_dump(payload: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_matching_sidecar(
    artifact_path: Path,
    metadata_path: Path,
    config_hash: str,
    requested_source_metadata: Mapping[str, Any],
    requested_anchor_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    if not metadata_path.exists():
        raise RuntimeError(
            f"--skip_existing refused {artifact_path}: metadata sidecar is missing"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"--skip_existing refused {artifact_path}: cannot read metadata sidecar"
        ) from exc
    observed_hash = metadata.get("config_hash")
    if observed_hash != config_hash:
        raise RuntimeError(
            f"--skip_existing refused {artifact_path}: config_hash mismatch "
            f"(existing={observed_hash!r}, requested={config_hash!r}); "
            "rerun without --skip_existing to atomically rebuild it"
        )
    if metadata.get("source_metadata") != requested_source_metadata:
        raise RuntimeError(
            f"--skip_existing refused {artifact_path}: source metadata changed; "
            "rerun without --skip_existing to atomically rebuild it"
        )
    if metadata.get("anchor_metadata") != _json_safe(requested_anchor_metadata):
        raise RuntimeError(
            f"--skip_existing refused {artifact_path}: participant anchor changed; "
            "rerun without --skip_existing to atomically rebuild it"
        )
    index_record = metadata.get("index_record")
    if not isinstance(index_record, dict):
        raise RuntimeError(
            f"--skip_existing refused {artifact_path}: sidecar lacks index_record"
        )
    return index_record


def _manifest_metadata(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        column: _json_safe(record.get(column)) for column in MANIFEST_METADATA_COLUMNS
    }


def _add_manifest_qc(
    aggregation_metadata: Dict[str, Any], record: Mapping[str, Any]
) -> None:
    expected = pd.to_numeric(record.get("number_of_observations"), errors="coerce")
    if pd.isna(expected):
        aggregation_metadata["manifest_number_of_observations"] = None
        aggregation_metadata["manifest_observation_count_difference"] = None
        mismatch = False
    else:
        expected_int = int(expected)
        difference = int(aggregation_metadata["n_raw_rows"]) - expected_int
        aggregation_metadata["manifest_number_of_observations"] = expected_int
        aggregation_metadata["manifest_observation_count_difference"] = difference
        mismatch = difference != 0
    aggregation_metadata["qc_flags"]["manifest_observation_count_mismatch"] = mismatch


def process_record(
    record: Mapping[str, Any],
    anchor: ParticipantAnchor,
    dataset_root: str,
    out_dir: str,
    spec: TimeGridSpec,
    config: Mapping[str, Any],
    skip_existing: bool,
) -> Dict[str, Any]:
    """Build or safely reuse one participant artifact."""
    if torch is None:
        raise ImportError("PyTorch is required to serialize environment .pt artifacts")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    person_id = as_person_id(record.get("person_id"))
    if not person_id:
        raise ValueError("Environment manifest row has an empty person_id")
    output_directory = Path(out_dir)
    artifact_path = output_directory / f"{person_id}.pt"
    metadata_path = output_directory / f"{person_id}.metadata.json"
    config_hash = str(config["config_hash"])

    raw_path = resolve_path(dataset_root, record.get("env_sensor_filepath"))
    source_metadata = {
        "files": source_file_metadata([raw_path]),
        "manifest_row": _manifest_metadata(record),
    }
    anchor_metadata = _json_safe(anchor.__dict__)
    shared_grid_hash = canonical_json_hash(
        {
            "anchor": anchor_metadata,
            "grid_minutes": int(spec.grid_minutes),
            "window_days": int(spec.window_days),
            "seq_len": int(spec.seq_len),
        }
    )

    if skip_existing and artifact_path.exists():
        index_record = dict(
            _load_matching_sidecar(
                artifact_path,
                metadata_path,
                config_hash,
                source_metadata,
                anchor_metadata,
            )
        )
        index_record["status"] = "skipped"
        return index_record

    frame = read_env_csv(raw_path)
    tensor, slot_mask, channel_mask, row_count, qc_masks, aggregation_metadata = (
        aggregate_environment_5min(frame, anchor, spec)
    )
    _add_manifest_qc(aggregation_metadata, record)
    qc_flags = [
        name
        for name, is_set in aggregation_metadata["qc_flags"].items()
        if bool(is_set)
    ]
    index_record: Dict[str, Any] = {
        "person_id": person_id,
        "artifact_path": str(artifact_path),
        "metadata_path": str(metadata_path),
        "config_hash": config_hash,
        "anchor_utc": anchor.anchor_utc,
        "timezone": anchor.timezone,
        "shared_grid_hash": shared_grid_hash,
        "n_raw_rows": int(aggregation_metadata["n_raw_rows"]),
        "n_valid_timestamp_rows": int(
            aggregation_metadata["n_valid_timestamp_rows"]
        ),
        "n_invalid_timestamp_rows": int(
            aggregation_metadata["n_invalid_timestamp_rows"]
        ),
        "n_rows_before_window": int(aggregation_metadata["n_rows_before_window"]),
        "n_rows_in_window": int(aggregation_metadata["n_rows_in_window"]),
        "n_rows_after_window": int(aggregation_metadata["n_rows_after_window"]),
        "n_rows_in_window_with_sensor_values": int(
            aggregation_metadata["n_rows_in_window_with_sensor_values"]
        ),
        "n_observed_slots": int(aggregation_metadata["n_observed_slots"]),
        "slot_coverage_fraction": float(
            aggregation_metadata["slot_coverage_fraction"]
        ),
        "n_observed_channel_slots": int(
            aggregation_metadata["n_observed_channel_slots"]
        ),
        "n_missing_source_channels": len(
            aggregation_metadata["missing_source_channels"]
        ),
        "n_rejected_sensor_values_in_window": int(
            aggregation_metadata["sensor_value_qc_counts"][
                "n_rejected_in_window"
            ]
        ),
        "n_hard_invalid_sensor_values_in_window": int(
            aggregation_metadata["sensor_value_qc_counts"][
                "n_hard_invalid_in_window"
            ]
        ),
        "n_saturated_sensor_values_in_window": int(
            aggregation_metadata["sensor_value_qc_counts"][
                "n_saturated_in_window"
            ]
        ),
        "n_outside_specified_sensor_values_in_window": int(
            aggregation_metadata["sensor_value_qc_counts"][
                "n_outside_specified_in_window"
            ]
        ),
        "qc_flags": json.dumps(qc_flags, separators=(",", ":")),
        "status": "processed",
    }

    artifact = {
        "person_id": person_id,
        "environment": torch.from_numpy(tensor),
        "environment_slot_mask": torch.from_numpy(slot_mask),
        "environment_channel_observed_mask": torch.from_numpy(channel_mask),
        "environment_channel_rejected_mask": torch.from_numpy(
            qc_masks["rejected"]
        ),
        "environment_channel_saturation_mask": torch.from_numpy(
            qc_masks["saturation"]
        ),
        "environment_channel_outside_specified_mask": torch.from_numpy(
            qc_masks["outside_specified"]
        ),
        "environment_row_count": torch.from_numpy(row_count),
        "feature_names": build_feature_names(),
        "sensor_columns": list(SENSOR_COLUMNS),
        "primary_sensor_columns": list(PRIMARY_SENSOR_COLUMNS),
        "technical_sensor_columns": list(TECHNICAL_SENSOR_COLUMNS),
        "default_model_excluded_feature_names": (
            default_model_excluded_feature_names()
        ),
        "anchor_metadata": anchor_metadata,
        "shared_grid_hash": shared_grid_hash,
        "aggregation_metadata": aggregation_metadata,
        "source_metadata": source_metadata,
        "preprocessing_config": dict(config),
        "config_hash": config_hash,
    }
    sidecar = {
        "person_id": person_id,
        "artifact_path": str(artifact_path),
        "config_hash": config_hash,
        "preprocessing_config": dict(config),
        "anchor_metadata": anchor_metadata,
        "shared_grid_hash": shared_grid_hash,
        "aggregation_metadata": aggregation_metadata,
        "source_metadata": source_metadata,
        "index_record": index_record,
    }

    _atomic_torch_save(artifact, artifact_path)
    _atomic_json_dump(sidecar, metadata_path)
    return index_record


def _failure_record(record: Mapping[str, Any], exc: BaseException) -> Dict[str, Any]:
    return {
        "person_id": as_person_id(record.get("person_id")),
        "env_sensor_filepath": str(record.get("env_sensor_filepath", "")),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def run_records(
    records: Sequence[Mapping[str, Any]],
    anchors: Mapping[str, ParticipantAnchor],
    dataset_root: Path,
    out_dir: Path,
    spec: TimeGridSpec,
    config: Mapping[str, Any],
    num_workers: int,
    skip_existing: bool,
    show_progress: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    def anchor_for(record: Mapping[str, Any]) -> ParticipantAnchor:
        person_id = as_person_id(record.get("person_id"))
        if person_id not in anchors:
            raise KeyError(
                f"person_id={person_id!r} from environment manifest is absent "
                "from --participants"
            )
        return anchors[person_id]

    if num_workers <= 1:
        iterator: Iterable[Mapping[str, Any]] = records
        if tqdm is not None and show_progress:
            iterator = tqdm(
                records,
                total=len(records),
                desc="Environment 5-minute aggregation",
                unit="person",
                dynamic_ncols=True,
            )
        for record in iterator:
            try:
                results.append(
                    process_record(
                        record,
                        anchor_for(record),
                        str(dataset_root),
                        str(out_dir),
                        spec,
                        config,
                        skip_existing,
                    )
                )
            except Exception as exc:
                failures.append(_failure_record(record, exc))
            if tqdm is not None and show_progress:
                iterator.set_postfix(  # type: ignore[attr-defined]
                    processed=sum(r["status"] == "processed" for r in results),
                    skipped=sum(r["status"] == "skipped" for r in results),
                    failed=len(failures),
                )
        return results, failures

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures: Dict[Any, Mapping[str, Any]] = {}
        for record in records:
            try:
                anchor = anchor_for(record)
            except Exception as exc:
                failures.append(_failure_record(record, exc))
                continue
            future = executor.submit(
                process_record,
                record,
                anchor,
                str(dataset_root),
                str(out_dir),
                spec,
                config,
                skip_existing,
            )
            futures[future] = record

        completed: Iterable[Any] = as_completed(futures)
        if tqdm is not None and show_progress:
            completed = tqdm(
                completed,
                total=len(futures),
                desc="Environment 5-minute aggregation",
                unit="person",
                dynamic_ncols=True,
            )
        for future in completed:
            record = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append(_failure_record(record, exc))
            if tqdm is not None and show_progress:
                completed.set_postfix(  # type: ignore[attr-defined]
                    processed=sum(r["status"] == "processed" for r in results),
                    skipped=sum(r["status"] == "skipped" for r in results),
                    failed=len(failures),
                )
    return results, failures


def _validate_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    required = {"person_id", "env_sensor_filepath"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Environment manifest missing required columns: {missing}")
    manifest = manifest.copy()
    manifest["person_id"] = manifest["person_id"].map(as_person_id)
    if (manifest["person_id"] == "").any():
        raise ValueError("Environment manifest contains an empty person_id")
    duplicates = sorted(
        manifest.loc[
            manifest["person_id"].duplicated(keep=False), "person_id"
        ].unique()
    )
    if duplicates:
        preview = duplicates[:10]
        raise ValueError(
            "Environment manifest has duplicate person_id rows; refusing unsafe "
            f"artifact overwrites. First duplicates: {preview}"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build shared-anchor, raw-scale 5-minute environment tensors."
    )
    parser.add_argument(
        "--manifest", required=True, help="Path to environment/manifest.tsv"
    )
    parser.add_argument(
        "--participants", required=True, help="Path to participants.tsv"
    )
    parser.add_argument(
        "--anchor_overrides",
        "--anchor-overrides",
        dest="anchor_overrides",
        default=None,
        help="Optional shared CSV/TSV with person_id,anchor_utc",
    )
    parser.add_argument(
        "--dataset_root",
        "--dataset-root",
        dest="dataset_root",
        required=True,
        help="Dataset root used to resolve env_sensor_filepath",
    )
    parser.add_argument("--out_dir", "--out-dir", dest="out_dir", required=True)
    parser.add_argument("--grid_minutes", type=int, default=5)
    parser.add_argument("--window_days", type=int, default=10)
    parser.add_argument("--anchor_hour_local", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--skip_existing", "--skip-existing", action="store_true")
    parser.add_argument("--no_progress", "--no-progress", action="store_true")
    args = parser.parse_args()
    if torch is None:
        parser.error("PyTorch is required to serialize environment .pt artifacts")

    spec = TimeGridSpec(
        grid_minutes=args.grid_minutes,
        window_days=args.window_days,
        anchor_hour_local=args.anchor_hour_local,
    )
    anchor_overrides = read_anchor_overrides(args.anchor_overrides)
    config = artifact_config(
        modality="environment_5min",
        spec=spec,
        extra={
            "artifact_format_version": 3,
            "builder_version": BUILDER_VERSION,
            "sensor_columns": list(SENSOR_COLUMNS),
            "primary_sensor_columns": list(PRIMARY_SENSOR_COLUMNS),
            "technical_sensor_columns": list(TECHNICAL_SENSOR_COLUMNS),
            "default_model_excluded_feature_names": (
                default_model_excluded_feature_names()
            ),
            "sensor_value_policy_version": "aireadi-environment-documented-v1",
            "sensor_value_rules": SENSOR_VALUE_RULES,
            "sensor_value_policy_provenance": {
                "aireadi_environment_v3": (
                    "https://docs.aireadi.org/docs/3/dataset/environment/"
                ),
                "sensirion_sen5x_datasheet": (
                    "https://sensirion.com/media/documents/6791EFA0/62A1F68F/"
                    "Sensirion_Datasheet_Environmental_Node_SEN5x.pdf"
                ),
                "sensirion_official_python_driver": (
                    "https://github.com/Sensirion/python-i2c-sen5x/blob/master/"
                    "sensirion_i2c_sen5x/response_types.py"
                ),
            },
            "statistics": ["mean", "std_ddof0", "missing_rate"],
            "missing_rate_denominator": "timestamped_rows_in_slot",
            "aggregation_valid_value_definition": (
                "finite_float32_representable_non_sentinel_and_hard_domain_valid"
            ),
            "slot_mask_definition": (
                "any_aggregation_valid_primary_sensor_value_excluding_technical_channels"
            ),
            "timestamp_semantics": "parse_as_UTC",
            "value_transform": "none_raw_scale",
            "hard_invalid_action": "exclude_as_missing_without_clipping",
            "saturation_action": "retain_and_flag_as_right_censored",
            "outside_specified_action": "retain_and_flag_without_clipping",
            "empty_numeric_fill": 0.0,
            "empty_missing_rate_fill": 1.0,
            "time_feature_names": list(TIME_FEATURE_NAMES),
            "anchor_policy": (
                "explicit_shared_utc_override"
                if anchor_overrides
                else "study_visit_date_local_civil_anchor_hour"
            ),
            "anchor_overrides_hash": (
                canonical_json_hash({"anchors": anchor_overrides})
                if anchor_overrides
                else None
            ),
        },
    )

    manifest_path = Path(args.manifest).resolve()
    participants_path = Path(args.participants).resolve()
    manifest = _validate_manifest(
        pd.read_csv(manifest_path, sep="\t", dtype={"person_id": "string"})
    )
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        manifest = manifest.head(args.limit).copy()

    anchors = load_participant_anchors(
        participants_path,
        spec,
        anchor_overrides_utc=anchor_overrides,
    )
    unknown_manifest_people = sorted(set(manifest["person_id"]) - set(anchors))
    if unknown_manifest_people:
        raise ValueError(
            "Environment manifest contains people absent from participants.tsv: "
            f"{unknown_manifest_people[:20]}"
        )
    output_directory = Path(args.out_dir).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    records = manifest.to_dict("records")
    results, failures = run_records(
        records,
        anchors,
        Path(args.dataset_root).resolve(),
        output_directory,
        spec,
        config,
        max(int(args.num_workers), 1),
        bool(args.skip_existing),
        not bool(args.no_progress),
    )

    index = pd.DataFrame(results)
    if index.empty:
        index = pd.DataFrame(columns=INDEX_COLUMNS)
    else:
        index = index.reindex(columns=INDEX_COLUMNS).sort_values("person_id")
    index.to_csv(output_directory / "environment_5min_index.csv", index=False)
    _atomic_json_dump(
        failures,
        output_directory / "environment_5min_failures.json",
    )

    feature_names = build_feature_names()
    n_processed = sum(result["status"] == "processed" for result in results)
    n_skipped = sum(result["status"] == "skipped" for result in results)
    schema = {
        "raw_modality": "environmental_sensor_time_series",
        "model_representation": "shared_anchor_5min_sequence",
        "artifact_glob": "*.pt",
        "sidecar_glob": "*.metadata.json",
        "tensor_key": "environment",
        "slot_mask_key": "environment_slot_mask",
        "channel_observed_mask_key": "environment_channel_observed_mask",
        "channel_rejected_mask_key": "environment_channel_rejected_mask",
        "channel_saturation_mask_key": "environment_channel_saturation_mask",
        "channel_outside_specified_mask_key": (
            "environment_channel_outside_specified_mask"
        ),
        "row_count_key": "environment_row_count",
        "shape": [spec.seq_len, len(feature_names)],
        "channel_mask_shape": [spec.seq_len, len(SENSOR_COLUMNS)],
        "channel_qc_mask_shape": [spec.seq_len, len(SENSOR_COLUMNS)],
        "row_count_shape": [spec.seq_len],
        "sensor_columns": list(SENSOR_COLUMNS),
        "primary_sensor_columns": list(PRIMARY_SENSOR_COLUMNS),
        "technical_sensor_columns": list(TECHNICAL_SENSOR_COLUMNS),
        "default_model_excluded_feature_names": (
            default_model_excluded_feature_names()
        ),
        "sensor_value_rules": SENSOR_VALUE_RULES,
        "feature_names": feature_names,
        "time_feature_names": list(TIME_FEATURE_NAMES),
        "aggregation": {
            "mean": (
                "finite float32-representable, non-sentinel, hard-domain-valid "
                "values in slot"
            ),
            "std": (
                "population std (ddof=0) of aggregation-valid values in slot"
            ),
            "missing_rate": (
                "1 - aggregation-valid channel values / timestamped rows in slot"
            ),
            "row_count": "rows with a valid UTC timestamp in slot",
            "slot_mask": (
                "any primary (non-technical) channel has at least one valid value in slot"
            ),
            "rejected_mask": (
                "slot/channel has a non-missing numeric-policy-rejected value"
            ),
            "saturation_mask": (
                "slot/channel has a retained maximum-code right-censored value"
            ),
            "outside_specified_mask": (
                "slot/channel has a retained value outside an accuracy-specified range"
            ),
        },
        "preprocessing_config": config,
        "config_hash": config["config_hash"],
        "source_files": source_file_metadata(
            [manifest_path, participants_path, args.anchor_overrides]
        ),
        "n_manifest_rows": int(len(manifest)),
        "n_index_rows": int(len(results)),
        "n_processed": int(n_processed),
        "n_skipped": int(n_skipped),
        "n_failed": int(len(failures)),
        "output_files": {
            "index": "environment_5min_index.csv",
            "failures": "environment_5min_failures.json",
            "schema": "environment_5min_schema.json",
        },
        "notes": [
            "The default grid is 5 minutes x 10 days (2880 slots).",
            "The participant anchor is shared with CGM/activity via preprocessing_utils.py.",
            "An exact shared anchor override table can replace the provisional visit-date anchor.",
            "Raw timestamps are parsed as UTC; circadian features use participant local time.",
            "UTC offset is retained in anchor metadata, not exposed as a model feature, to reduce site shortcuts.",
            "Sensor means/stds retain raw scale and missing values are zero-filled with masks.",
            "Documented sentinel/impossible values are excluded as missing; values are never clipped.",
            "PM 6553.4 ug/m3 is retained as right-censored saturation; PM 6553.5 is the unavailable sentinel.",
            "PM above 1000 ug/m3 is retained but flagged outside the manufacturer's accuracy-specified range.",
            "Temperature outside the documented -10 to 50 C domain is excluded as missing.",
            "Device-internal inttemp is retained for QC but default-excluded from model features and slot validity.",
            "No normalization or interpolation is applied.",
            "Fit all value scaling/imputation parameters on the downstream training split only.",
            "--skip_existing requires matching config, source metadata, and participant anchor metadata.",
            "Value-policy provenance: AI-READI v3 Environment documentation and Sensirion SEN5x interface/datasheet.",
        ],
    }
    _atomic_json_dump(schema, output_directory / "environment_5min_schema.json")

    print(
        json.dumps(
            {
                "out_dir": str(output_directory),
                "config_hash": config["config_hash"],
                "n_manifest_rows": int(len(manifest)),
                "n_processed": int(n_processed),
                "n_skipped": int(n_skipped),
                "n_failed": int(len(failures)),
                "shape_per_person": [spec.seq_len, len(feature_names)],
                "channel_mask_shape": [spec.seq_len, len(SENSOR_COLUMNS)],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
