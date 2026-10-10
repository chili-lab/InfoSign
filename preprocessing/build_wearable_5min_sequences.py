#!/usr/bin/env python3
"""Build canonical 5-minute CGM and activity artifacts for AI-READI.

This is the V2 replacement for ``build_wearable_hourly_sequences.py``.  It
keeps CGM and Garmin activity as separate tensors on one participant-level
absolute UTC grid.  The shared grid is defined by ``preprocessing_utils`` and is
anchored at the study-visit date in local civil time (5 minutes x 10 days by
default).

Important properties of the canonical representation:

* all participants in ``participants.tsv`` are represented, including people
  for whom one or both wearable manifests have no row;
* CGM values are never clipped or interpolated; censored Low/High records,
  duplicate readings, numeric validity, and plausibility are explicit;
* Garmin point streams use observed counts rather than a guessed expected
  cadence or count fraction;
* steps and calories are allocated to slots in proportion to interval overlap;
* sleep stages use exact deduplication and deterministic interval-union/conflict
  resolution before minutes are allocated, with observed/stage/conflict masks;
* deterministic local-clock features are computed from the shared UTC grid;
* every artifact contains its preprocessing config hash, input provenance,
  anchor, and audit metadata.  ``--skip_existing`` only skips a current cache.

No task labels (including MoCA or CVD outcomes) are read by this script.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
        load_participant_anchors,
        local_time_features,
        parse_timestamp_utc,
        read_anchor_overrides,
        slot_indices,
        source_file_metadata,
    )
except ImportError:  # Allow direct script execution.
    from preprocessing_utils import (  # type: ignore
        ParticipantAnchor,
        TimeGridSpec,
        artifact_config,
        as_person_id,
        canonical_json_hash,
        load_participant_anchors,
        local_time_features,
        parse_timestamp_utc,
        read_anchor_overrides,
        slot_indices,
        source_file_metadata,
    )


ARTIFACT_FORMAT = "aireadi_ot2_wearable_5min_v2"
BUILDER_VERSION = "wearable-5min-5"

CGM_FEATURE_NAMES = [
    "cgm__glucose_mg_dl",
    "cgm__record_count",
    "cgm__numeric_count",
    "cgm__duplicate_count",
    "cgm__censored_low",
    "cgm__censored_high",
    "cgm__below_plausible_range",
    "cgm__above_plausible_range",
]
CGM_FEATURE_INDEX = {name: i for i, name in enumerate(CGM_FEATURE_NAMES)}

# source name -> (JSON body key, nested measurement path)
POINT_SIGNAL_SPECS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "heart_rate": ("heart_rate", ("heart_rate", "value")),
    "oxygen_saturation": ("breathing", ("oxygen_saturation", "value")),
    "stress": ("stress", ("stress", "value")),
    "respiratory_rate": ("breathing", ("respiratory_rate", "value")),
}
POINT_SIGNAL_NAMES = list(POINT_SIGNAL_SPECS)

INTERVAL_SIGNAL_NAMES = ["steps", "calories"]
SLEEP_STAGE_NAMES = ["awake", "light", "deep", "rem", "unknown"]
LOCAL_TIME_FEATURE_NAMES = [
    "time__day_sin",
    "time__day_cos",
    "time__day_index_norm",
]


def _activity_feature_names() -> List[str]:
    names: List[str] = []
    for signal in POINT_SIGNAL_NAMES:
        names.extend(
            [
                f"activity__{signal}__mean",
                f"activity__{signal}__valid_count",
                f"activity__{signal}__offline_count",
            ]
        )
    for signal in INTERVAL_SIGNAL_NAMES:
        names.extend(
            [
                f"activity__{signal}__allocated_value",
                f"activity__{signal}__event_count",
            ]
        )
    names.append("activity__sleep__any_minutes")
    names.extend(f"activity__sleep__{stage}_minutes" for stage in SLEEP_STAGE_NAMES)
    names.extend(LOCAL_TIME_FEATURE_NAMES)
    return names


ACTIVITY_FEATURE_NAMES = _activity_feature_names()
ACTIVITY_FEATURE_INDEX = {name: i for i, name in enumerate(ACTIVITY_FEATURE_NAMES)}

ACTIVITY_PATH_COLUMNS = {
    "heart_rate": "heartrate_filepath",
    "oxygen_saturation": "oxygen_saturation_filepath",
    "stress": "stress_level_filepath",
    "sleep": "sleep_filepath",
    "respiratory_rate": "respiratory_rate_filepath",
    "steps": "physical_activity_filepath",
    "calories": "active_calories_filepath",
}


def _valid_manifest_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    text = str(value).strip()
    return bool(text) and text.lower() not in {
        "none",
        "nan",
        "not reported",
        "not_provided",
        "not provided",
        "null",
    }


def resolve_path(dataset_root: Path, value: Any) -> Optional[Path]:
    """Resolve AI-READI manifest paths relative to the supplied dataset root."""
    if not _valid_manifest_value(value):
        return None
    text = str(value).strip()
    # AI-READI manifests commonly use dataset-root-relative paths beginning
    # with '/'.  Preserve the established OT2 interpretation of those paths.
    if text.startswith("/"):
        text = text[1:]
    return dataset_root / text


def _body_records(obj: Optional[dict], key: str) -> List[dict]:
    if not isinstance(obj, dict):
        return []
    body = obj.get("body", {})
    if not isinstance(body, dict):
        return []
    records = body.get(key, [])
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def _nested_value(record: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = record
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _timeframe_raw(record: Mapping[str, Any]) -> Tuple[Any, Any]:
    frame = record.get("effective_time_frame", {})
    if not isinstance(frame, Mapping):
        return None, None
    if frame.get("date_time") is not None:
        value = frame.get("date_time")
        return value, value
    interval = frame.get("time_interval", {})
    if not isinstance(interval, Mapping):
        return None, None
    start = interval.get("start_date_time") or interval.get("end_date_time")
    end = interval.get("end_date_time") or start
    return start, end


def _timestamp_raw(record: Mapping[str, Any]) -> Any:
    return _timeframe_raw(record)[0]


def _finite_float(value: Any) -> Optional[float]:
    if isinstance(value, Mapping):
        value = value.get("value")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(result):
        return None
    # Artifacts use float32.  Treat an unrepresentable value as invalid and
    # report it in the audit rather than silently overflowing to infinity.
    if abs(result) > np.finfo(np.float32).max:
        return None
    return result


def _parsed_timestamps(raw_values: Sequence[Any]) -> pd.Series:
    if not raw_values:
        return pd.Series([], dtype="datetime64[ns, UTC]")
    return pd.to_datetime(pd.Series(list(raw_values)), errors="coerce", utc=True)


def _slot_array(
    raw_timestamps: Sequence[Any], anchor_utc: pd.Timestamp, spec: TimeGridSpec
) -> Tuple[np.ndarray, np.ndarray]:
    parsed = _parsed_timestamps(raw_timestamps)
    slots = slot_indices(parsed, anchor_utc, spec)
    valid_timestamp = slots.notna().to_numpy(dtype=bool)
    # Use a sentinel only after preserving the valid-timestamp mask.
    slot_values = slots.fillna(np.iinfo(np.int64).min).to_numpy(dtype=np.int64)
    return slot_values, valid_timestamp


def _gap_audit(mask: np.ndarray) -> Dict[str, Any]:
    observed = np.flatnonzero(np.asarray(mask, dtype=bool))
    n = int(len(mask))
    if observed.size == 0:
        return {
            "first_slot": None,
            "last_slot": None,
            "leading_gap_slots": n,
            "trailing_gap_slots": n,
            "n_internal_gap_runs": 0,
            "n_internal_gap_slots": 0,
            "max_internal_gap_slots": 0,
        }
    differences = np.diff(observed)
    gap_lengths = differences[differences > 1] - 1
    return {
        "first_slot": int(observed[0]),
        "last_slot": int(observed[-1]),
        "leading_gap_slots": int(observed[0]),
        "trailing_gap_slots": int(n - 1 - observed[-1]),
        "n_internal_gap_runs": int(gap_lengths.size),
        "n_internal_gap_slots": int(gap_lengths.sum()) if gap_lengths.size else 0,
        "max_internal_gap_slots": int(gap_lengths.max()) if gap_lengths.size else 0,
    }


def _classify_cgm_value(value: Any) -> Tuple[str, Optional[float]]:
    if isinstance(value, Mapping):
        value = value.get("value")
    if value is None:
        return "invalid", None
    if isinstance(value, str):
        text = value.strip()
        normalized = re.sub(r"\s+", "", text).lower()
        if normalized in {"low", "lo"} or normalized.startswith("low") or normalized.startswith("<"):
            return "censored_low", None
        if normalized in {"high", "hi"} or normalized.startswith("high") or normalized.startswith(">"):
            return "censored_high", None
    numeric = _finite_float(value)
    if numeric is None:
        return "invalid", None
    return "numeric", numeric


def aggregate_cgm(
    obj: Optional[dict],
    anchor: ParticipantAnchor,
    spec: TimeGridSpec,
    plausible_min: float,
    plausible_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Aggregate raw CGM records without interpolation, clipping, or censor imputation."""
    n = spec.seq_len
    x = np.zeros((n, len(CGM_FEATURE_NAMES)), dtype=np.float32)
    feature_mask = np.zeros_like(x, dtype=bool)
    observed_mask = np.zeros(n, dtype=bool)
    valid_mask = np.zeros(n, dtype=bool)

    records = _body_records(obj, "cgm")
    raw_timestamps: List[Any] = []
    classes: List[str] = []
    numeric_values = np.full(len(records), np.nan, dtype=np.float64)
    for i, record in enumerate(records):
        raw_timestamps.append(_timestamp_raw(record))
        category, numeric = _classify_cgm_value(record.get("blood_glucose"))
        classes.append(category)
        if numeric is not None:
            numeric_values[i] = numeric

    slots, valid_timestamp = _slot_array(raw_timestamps, anchor.utc_timestamp(), spec)
    in_window = valid_timestamp & (slots >= 0) & (slots < n)
    before_window = valid_timestamp & (slots < 0)
    after_window = valid_timestamp & (slots >= n)

    record_count = np.bincount(slots[in_window], minlength=n).astype(np.float32)
    observed_mask = record_count > 0
    x[:, CGM_FEATURE_INDEX["cgm__record_count"]] = record_count
    x[:, CGM_FEATURE_INDEX["cgm__duplicate_count"]] = np.maximum(record_count - 1.0, 0.0)

    class_array = np.asarray(classes, dtype=object)
    numeric = class_array == "numeric"
    numeric_in = in_window & numeric
    numeric_count = np.bincount(slots[numeric_in], minlength=n).astype(np.float32)
    numeric_sum = np.bincount(
        slots[numeric_in], weights=numeric_values[numeric_in], minlength=n
    )
    valid_mask = numeric_count > 0
    if valid_mask.any():
        x[valid_mask, CGM_FEATURE_INDEX["cgm__glucose_mg_dl"]] = (
            numeric_sum[valid_mask] / numeric_count[valid_mask]
        ).astype(np.float32)
    x[:, CGM_FEATURE_INDEX["cgm__numeric_count"]] = numeric_count

    low = class_array == "censored_low"
    high = class_array == "censored_high"
    below = numeric & (numeric_values < plausible_min)
    above = numeric & (numeric_values > plausible_max)
    for flag, feature in [
        (low, "cgm__censored_low"),
        (high, "cgm__censored_high"),
        (below, "cgm__below_plausible_range"),
        (above, "cgm__above_plausible_range"),
    ]:
        selected = in_window & flag
        if selected.any():
            flagged_slots = np.unique(slots[selected])
            x[flagged_slots, CGM_FEATURE_INDEX[feature]] = 1.0

    value_index = CGM_FEATURE_INDEX["cgm__glucose_mg_dl"]
    feature_mask[:, value_index] = valid_mask
    # Counts, duplicates, and censor flags are meaningful when at least one
    # timestamped raw record was observed in the slot.
    for feature in [
        "cgm__record_count",
        "cgm__numeric_count",
        "cgm__duplicate_count",
        "cgm__censored_low",
        "cgm__censored_high",
    ]:
        feature_mask[:, CGM_FEATURE_INDEX[feature]] = observed_mask
    # Plausibility flags are meaningful only for slots containing a numeric value.
    feature_mask[:, CGM_FEATURE_INDEX["cgm__below_plausible_range"]] = valid_mask
    feature_mask[:, CGM_FEATURE_INDEX["cgm__above_plausible_range"]] = valid_mask

    duplicate_slots = record_count > 1
    audit: Dict[str, Any] = {
        "n_raw_records": int(len(records)),
        "n_invalid_timestamps": int((~valid_timestamp).sum()),
        "n_records_before_window": int(before_window.sum()),
        "n_records_after_window": int(after_window.sum()),
        "n_records_in_window": int(in_window.sum()),
        "n_observed_slots": int(observed_mask.sum()),
        "n_valid_numeric_slots": int(valid_mask.sum()),
        "n_numeric_records": int(numeric.sum()),
        "n_invalid_value_records": int((class_array == "invalid").sum()),
        "n_censored_low_records": int(low.sum()),
        "n_censored_high_records": int(high.sum()),
        "n_below_plausible_records": int(below.sum()),
        "n_above_plausible_records": int(above.sum()),
        "n_duplicate_slots": int(duplicate_slots.sum()),
        "n_duplicate_extra_records": int(np.maximum(record_count - 1, 0).sum()),
        "plausible_min_mg_dl": float(plausible_min),
        "plausible_max_mg_dl": float(plausible_max),
        "observed_gap_audit": _gap_audit(observed_mask),
        "valid_numeric_gap_audit": _gap_audit(valid_mask),
    }
    return x, observed_mask, valid_mask, feature_mask, audit


def _extract_point_records(
    obj: Optional[dict], body_key: str, value_path: Sequence[str]
) -> Tuple[List[Any], np.ndarray]:
    records = _body_records(obj, body_key)
    timestamps: List[Any] = []
    values = np.full(len(records), np.nan, dtype=np.float64)
    for i, record in enumerate(records):
        timestamps.append(_timestamp_raw(record))
        value = _finite_float(_nested_value(record, value_path))
        if value is not None:
            values[i] = value
    return timestamps, values


def aggregate_point_signal(
    obj: Optional[dict],
    body_key: str,
    value_path: Sequence[str],
    anchor: ParticipantAnchor,
    spec: TimeGridSpec,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return slot mean, valid count, offline count, and point masks.

    Negative finite values are retained as explicit offline counts and excluded
    from the physiological mean.  No expected sampling cadence is assumed.
    """
    n = spec.seq_len
    timestamps, values = _extract_point_records(obj, body_key, value_path)
    slots, valid_timestamp = _slot_array(timestamps, anchor.utc_timestamp(), spec)
    in_window = valid_timestamp & (slots >= 0) & (slots < n)
    before_window = valid_timestamp & (slots < 0)
    after_window = valid_timestamp & (slots >= n)

    finite = np.isfinite(values)
    offline = finite & (values < 0)
    valid = finite & ~offline

    raw_count = np.bincount(slots[in_window], minlength=n)
    valid_in = in_window & valid
    offline_in = in_window & offline
    valid_count = np.bincount(slots[valid_in], minlength=n).astype(np.float32)
    offline_count = np.bincount(slots[offline_in], minlength=n).astype(np.float32)
    sums = np.bincount(slots[valid_in], weights=values[valid_in], minlength=n)

    means = np.zeros(n, dtype=np.float32)
    valid_mask = valid_count > 0
    means[valid_mask] = (sums[valid_mask] / valid_count[valid_mask]).astype(np.float32)
    observed_mask = raw_count > 0
    audit = {
        "n_raw_records": int(len(values)),
        "n_invalid_timestamps": int((~valid_timestamp).sum()),
        "n_invalid_values": int((~finite).sum()),
        "n_offline_records": int(offline.sum()),
        "n_records_before_window": int(before_window.sum()),
        "n_records_after_window": int(after_window.sum()),
        "n_records_in_window": int(in_window.sum()),
        "n_observed_slots": int(observed_mask.sum()),
        "n_valid_slots": int(valid_mask.sum()),
        "observed_gap_audit": _gap_audit(observed_mask),
    }
    return means, valid_count, offline_count, observed_mask, valid_mask, audit


def _quantity_events(
    obj: Optional[dict], body_key: str, candidate_fields: Sequence[str]
) -> List[Tuple[Any, Any, Any]]:
    events: List[Tuple[Any, Any, Any]] = []
    for record in _body_records(obj, body_key):
        value: Any = None
        for field in candidate_fields:
            if field in record:
                value = record.get(field)
                break
        start, end = _timeframe_raw(record)
        events.append((start, end, value))
    return events


def _point_slot(timestamp: pd.Timestamp, anchor_utc: pd.Timestamp, spec: TimeGridSpec) -> int:
    return int(math.floor((timestamp - anchor_utc).total_seconds() / spec.slot_seconds))


def allocate_quantity_events(
    events: Sequence[Tuple[Any, Any, Any]],
    anchor: ParticipantAnchor,
    spec: TimeGridSpec,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Allocate interval totals proportionally to overlapping grid slots."""
    n = spec.seq_len
    values = np.zeros(n, dtype=np.float32)
    event_counts = np.zeros(n, dtype=np.float32)
    mask = np.zeros(n, dtype=bool)
    anchor_utc = anchor.utc_timestamp()
    window_end = anchor_utc + pd.Timedelta(seconds=spec.seq_len * spec.slot_seconds)

    audit: Dict[str, Any] = {
        "n_raw_events": int(len(events)),
        "n_invalid_timestamp_events": 0,
        "n_invalid_value_events": 0,
        "n_negative_value_events": 0,
        "n_events_before_window": 0,
        "n_events_after_window": 0,
        "n_events_partially_in_window": 0,
        "n_events_in_window": 0,
        "raw_value_total": 0.0,
        "allocated_value_in_window": 0.0,
        "value_outside_window": 0.0,
    }

    for raw_start, raw_end, raw_value in events:
        start = parse_timestamp_utc(raw_start)
        end = parse_timestamp_utc(raw_end)
        value = _finite_float(raw_value)
        if pd.isna(start):
            audit["n_invalid_timestamp_events"] += 1
            continue
        if value is None:
            audit["n_invalid_value_events"] += 1
            continue
        if value < 0:
            audit["n_negative_value_events"] += 1
            continue

        audit["raw_value_total"] += float(value)
        if pd.isna(end) or end <= start:
            slot = _point_slot(start, anchor_utc, spec)
            if slot < 0:
                audit["n_events_before_window"] += 1
                audit["value_outside_window"] += float(value)
            elif slot >= n:
                audit["n_events_after_window"] += 1
                audit["value_outside_window"] += float(value)
            else:
                values[slot] += float(value)
                event_counts[slot] += 1.0
                mask[slot] = True
                audit["n_events_in_window"] += 1
                audit["allocated_value_in_window"] += float(value)
            continue

        total_seconds = float((end - start).total_seconds())
        clipped_start = max(start, anchor_utc)
        clipped_end = min(end, window_end)
        if clipped_end <= clipped_start:
            if end <= anchor_utc:
                audit["n_events_before_window"] += 1
            else:
                audit["n_events_after_window"] += 1
            audit["value_outside_window"] += float(value)
            continue

        if start < anchor_utc or end > window_end:
            audit["n_events_partially_in_window"] += 1
        audit["n_events_in_window"] += 1

        first_slot = max(0, _point_slot(clipped_start, anchor_utc, spec))
        # Subtract a tiny epsilon so an interval ending on a boundary does not
        # create a zero-overlap extra slot.
        last_slot = min(
            n - 1,
            int(
                math.floor(
                    ((clipped_end - anchor_utc).total_seconds() - 1e-9)
                    / spec.slot_seconds
                )
            ),
        )
        allocated = 0.0
        for slot in range(first_slot, last_slot + 1):
            slot_start = anchor_utc + pd.Timedelta(seconds=slot * spec.slot_seconds)
            slot_end = slot_start + pd.Timedelta(seconds=spec.slot_seconds)
            overlap = max(
                0.0,
                float((min(end, slot_end) - max(start, slot_start)).total_seconds()),
            )
            if overlap <= 0:
                continue
            amount = float(value) * overlap / total_seconds
            values[slot] += amount
            event_counts[slot] += 1.0
            mask[slot] = True
            allocated += amount
        audit["allocated_value_in_window"] += allocated
        audit["value_outside_window"] += max(0.0, float(value) - allocated)

    # Recompute the in-window sum after float32 accumulation for traceability.
    audit["allocated_value_tensor_sum"] = float(values.astype(np.float64).sum())
    audit["n_observed_slots"] = int(mask.sum())
    return values, event_counts, mask, audit


def _duration_unit_multiplier_minutes(unit: Any) -> Optional[float]:
    text = str(unit).strip().lower()
    if text in {"m", "min", "mins", "minute", "minutes"}:
        return 1.0
    if text in {"ms", "msec", "msecs", "millisecond", "milliseconds"}:
        return 1.0 / 60000.0
    if text in {"s", "sec", "secs", "second", "seconds"}:
        return 1.0 / 60.0
    if text in {"h", "hr", "hrs", "hour", "hours"}:
        return 60.0
    if text in {"d", "day", "days"}:
        return 1440.0
    return None


def _duration_minutes(value: Any) -> Optional[float]:
    if value is None:
        return None
    unit = "min"
    raw = value
    if isinstance(value, Mapping):
        raw = value.get("value")
        unit = value.get("unit", "min")
        # Validate an explicit unit before either numeric or ISO-like parsing;
        # otherwise ``{"value": "PT5M", "unit": "fortnight"}`` would
        # accidentally bypass the unit policy through pandas' ISO parser.
        if _duration_unit_multiplier_minutes(unit) is None:
            return None
    try:
        number = float(raw)
    except (TypeError, ValueError, OverflowError):
        # Support ISO-8601-like strings accepted by pandas (for example PT5M).
        try:
            delta = pd.to_timedelta(raw)
            if pd.isna(delta):
                return None
            return float(delta.total_seconds() / 60.0)
        except Exception:
            return None
    if not np.isfinite(number) or number <= 0:
        return None
    multiplier = _duration_unit_multiplier_minutes(unit)
    if multiplier is None:
        return None
    return float(number * multiplier)


def _sleep_interval(record: Mapping[str, Any]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    frame: Any = (
        record.get("sleep_stage_time_frame")
        or record.get("effective_time_frame")
        or record.get("time_frame")
        or {}
    )
    if not isinstance(frame, Mapping):
        return pd.NaT, pd.NaT
    interval = frame.get("time_interval", {})
    if not isinstance(interval, Mapping):
        point = parse_timestamp_utc(frame.get("date_time"))
        return point, point
    start = parse_timestamp_utc(interval.get("start_date_time"))
    end = parse_timestamp_utc(interval.get("end_date_time"))
    if pd.notna(start) and pd.isna(end):
        duration = _duration_minutes(interval.get("duration"))
        if duration is not None:
            end = start + pd.Timedelta(minutes=duration)
    return start, end


def _normalize_sleep_stage(value: Any) -> str:
    text = str(value if value is not None else "unknown").strip().lower()
    if "rem" in text:
        return "rem"
    if "deep" in text:
        return "deep"
    if "light" in text:
        return "light"
    if "awake" in text or "wake" in text:
        return "awake"
    return "unknown"


SleepEvent = Tuple[pd.Timestamp, pd.Timestamp, str]


def _sleep_raw_records(
    obj: Optional[dict],
) -> Tuple[List[Any], Dict[str, Any]]:
    """Return sleep records without silently changing a singleton into no data."""

    audit: Dict[str, Any] = {
        "sleep_node_type": "missing",
        "sleep_record_source": "none",
        "n_raw_sleep_records": 0,
        "n_nonmapping_sleep_records": 0,
    }
    if not isinstance(obj, dict):
        return [], audit
    body = obj.get("body", obj)
    if not isinstance(body, Mapping):
        audit["sleep_node_type"] = type(body).__name__
        return [], audit

    raw_node = body.get("sleep")
    if isinstance(raw_node, list):
        records: List[Any] = list(raw_node)
        audit["sleep_node_type"] = "list"
        audit["sleep_record_source"] = "body.sleep"
    elif isinstance(raw_node, Mapping):
        records = [raw_node]
        audit["sleep_node_type"] = "mapping"
        audit["sleep_record_source"] = "body.sleep"
    else:
        # Restrict legacy fallback to sleep-specific aliases.  Flattening every
        # list under ``body`` can silently ingest an unrelated staged record.
        records = []
        fallback_key = None
        for key in ("sleep_stages", "sleep_records"):
            value = body.get(key)
            if isinstance(value, list):
                records = list(value)
                fallback_key = key
                break
        audit["sleep_node_type"] = (
            "missing" if raw_node is None else type(raw_node).__name__
        )
        audit["sleep_record_source"] = (
            f"body.{fallback_key}" if fallback_key is not None else "none"
        )

    audit["n_raw_sleep_records"] = int(len(records))
    audit["n_nonmapping_sleep_records"] = int(
        sum(not isinstance(record, Mapping) for record in records)
    )
    return records, audit


def _sleep_event_candidates(
    obj: Optional[dict],
) -> Tuple[List[SleepEvent], Dict[str, Any]]:
    """Parse direct/summary records with one mutually exclusive source policy.

    A complete stage summary is more specific than its outer direct record, so
    it replaces that record.  An incomplete summary is never mixed with the
    direct event: the direct event is used as a fallback when available.
    """

    raw_records, audit = _sleep_raw_records(obj)
    counters: Dict[str, int] = {
        "n_records_with_direct_stage": 0,
        "n_records_with_valid_direct_event": 0,
        "n_records_with_invalid_direct_event": 0,
        "n_records_with_summary": 0,
        "n_records_with_direct_and_summary": 0,
        "n_records_using_direct": 0,
        "n_records_using_summary": 0,
        "n_records_summary_fallback_to_direct": 0,
        "n_records_dropped_incomplete_summary_without_direct": 0,
        "n_records_dropped_incomplete_summary_invalid_direct": 0,
        "n_records_dropped_invalid_direct": 0,
        "n_direct_events_suppressed_by_summary": 0,
        "n_invalid_direct_records_ignored_because_summary_used": 0,
        "n_direct_events_selected": 0,
        "n_summary_events_selected": 0,
        "n_summary_items_with_unrecognized_duration_unit": 0,
        "n_summary_outer_end_compared": 0,
        "n_summary_outer_end_mismatch": 0,
    }
    events: List[SleepEvent] = []
    summary_outer_end_differences: List[float] = []

    for record in raw_records:
        if not isinstance(record, Mapping):
            continue
        direct_stage = (
            record.get("sleep_stage_state")
            or record.get("sleep_stage")
            or record.get("stage")
        )
        has_direct = direct_stage is not None
        direct_start, direct_end = pd.NaT, pd.NaT
        direct_valid = False
        if has_direct:
            counters["n_records_with_direct_stage"] += 1
            direct_start, direct_end = _sleep_interval(record)
            direct_valid = bool(
                pd.notna(direct_start)
                and pd.notna(direct_end)
                and direct_end > direct_start
            )
            if direct_valid:
                counters["n_records_with_valid_direct_event"] += 1
            else:
                counters["n_records_with_invalid_direct_event"] += 1

        summary = record.get("sleep_stage_summary") or record.get("stage_summary")
        has_summary = isinstance(summary, list) and bool(summary)
        if has_summary:
            counters["n_records_with_summary"] += 1
        if has_direct and has_summary:
            counters["n_records_with_direct_and_summary"] += 1

        summary_events: List[SleepEvent] = []
        summary_complete = has_summary
        outer_end = pd.NaT
        if has_summary:
            outer_start, outer_end = _sleep_interval(record)
            cursor = outer_start
            if pd.isna(cursor):
                summary_complete = False
            else:
                assert isinstance(summary, list)
                for item in summary:
                    if not isinstance(item, Mapping):
                        summary_complete = False
                        break
                    nested_stage = (
                        item.get("sleep_stage_state")
                        or item.get("sleep_stage")
                        or item.get("stage")
                    )
                    raw_duration = item.get("duration")
                    if (
                        isinstance(raw_duration, Mapping)
                        and "unit" in raw_duration
                        and _duration_unit_multiplier_minutes(raw_duration.get("unit"))
                        is None
                    ):
                        counters[
                            "n_summary_items_with_unrecognized_duration_unit"
                        ] += 1
                    duration = _duration_minutes(raw_duration)
                    if nested_stage is None or duration is None:
                        summary_complete = False
                        break
                    end = cursor + pd.Timedelta(minutes=duration)
                    summary_events.append(
                        (cursor, end, _normalize_sleep_stage(nested_stage))
                    )
                    cursor = end

        if summary_complete and pd.notna(outer_end):
            difference_minutes = float((cursor - outer_end).total_seconds() / 60.0)
            summary_outer_end_differences.append(difference_minutes)
            counters["n_summary_outer_end_compared"] += 1
            if abs(difference_minutes) > (1.0 / 60.0):
                counters["n_summary_outer_end_mismatch"] += 1

        if summary_complete:
            events.extend(summary_events)
            counters["n_records_using_summary"] += 1
            counters["n_summary_events_selected"] += len(summary_events)
            if direct_valid:
                counters["n_direct_events_suppressed_by_summary"] += 1
            elif has_direct:
                counters[
                    "n_invalid_direct_records_ignored_because_summary_used"
                ] += 1
            continue

        if has_direct and direct_valid:
            events.append(
                (direct_start, direct_end, _normalize_sleep_stage(direct_stage))
            )
            counters["n_records_using_direct"] += 1
            counters["n_direct_events_selected"] += 1
            if has_summary:
                counters["n_records_summary_fallback_to_direct"] += 1
        elif has_direct:
            counters["n_records_dropped_invalid_direct"] += 1
            if has_summary:
                counters["n_records_dropped_incomplete_summary_invalid_direct"] += 1
        elif has_summary:
            counters["n_records_dropped_incomplete_summary_without_direct"] += 1

    audit.update(counters)
    audit["n_candidate_events"] = int(len(events))
    # Compatibility alias: V4 called parsed candidate events "raw events".
    audit["n_raw_events"] = int(len(events))
    audit["summary_outer_end_match_tolerance_seconds"] = 1.0
    audit["summary_outer_end_difference_minutes_max_abs"] = (
        float(max(abs(value) for value in summary_outer_end_differences))
        if summary_outer_end_differences
        else 0.0
    )
    audit["summary_outer_end_difference_minutes_signed_sum"] = float(
        sum(summary_outer_end_differences)
    )
    return events, audit


def _merge_adjacent_stage_intervals(events: Sequence[SleepEvent]) -> List[SleepEvent]:
    merged: List[SleepEvent] = []
    for start, end, stage in events:
        if merged and merged[-1][1] == start and merged[-1][2] == stage:
            merged[-1] = (merged[-1][0], end, stage)
        else:
            merged.append((start, end, stage))
    return merged


def _merge_adjacent_intervals(
    intervals: Sequence[Tuple[pd.Timestamp, pd.Timestamp]],
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    merged: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for start, end in intervals:
        if merged and merged[-1][1] == start:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _resolve_sleep_events(
    candidates: Sequence[SleepEvent],
) -> Tuple[List[SleepEvent], List[Tuple[pd.Timestamp, pd.Timestamp]], Dict[str, Any]]:
    """Deduplicate and deterministically resolve overlapping sleep intervals.

    The sweep partitions time into atomic half-open intervals.  Repeated or
    overlapping intervals of the same stage count once.  If multiple known
    stages cover an atomic interval, the stage is unknown rather than chosen
    from arbitrary source order.  Explicit unknown is lower priority than one
    unambiguous known stage.
    """

    valid: List[SleepEvent] = []
    invalid_count = 0
    candidate_minutes = 0.0
    for start, end, stage in candidates:
        if pd.isna(start) or pd.isna(end) or end <= start:
            invalid_count += 1
            continue
        normalized_stage = stage if stage in SLEEP_STAGE_NAMES else "unknown"
        valid.append((start, end, normalized_stage))
        candidate_minutes += float((end - start).total_seconds() / 60.0)

    signature_counts = Counter(valid)
    unique_events = sorted(
        signature_counts,
        key=lambda event: (event[0].value, event[1].value, event[2]),
    )
    duplicate_groups = sum(count > 1 for count in signature_counts.values())
    duplicate_extras = sum(count - 1 for count in signature_counts.values())
    unique_event_minutes = sum(
        float((end - start).total_seconds() / 60.0)
        for start, end, _ in unique_events
    )

    changes: Dict[pd.Timestamp, Dict[str, List[str]]] = defaultdict(
        lambda: {"start": [], "end": []}
    )
    for start, end, stage in unique_events:
        changes[start]["start"].append(stage)
        changes[end]["end"].append(stage)

    active: Counter[str] = Counter()
    previous: Optional[pd.Timestamp] = None
    atomic_events: List[SleepEvent] = []
    conflict_intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    same_stage_overlap_segments = 0
    same_stage_overlap_minutes = 0.0
    explicit_unknown_only_minutes = 0.0
    explicit_unknown_suppressed_minutes = 0.0
    conflict_minutes = 0.0
    conflict_combinations: Counter[str] = Counter()

    for boundary in sorted(changes):
        if previous is not None and boundary > previous and active:
            duration_minutes = float((boundary - previous).total_seconds() / 60.0)
            active_known = sorted(
                stage for stage, count in active.items() if count > 0 and stage != "unknown"
            )
            has_explicit_unknown = active.get("unknown", 0) > 0
            if any(count > 1 for count in active.values()):
                same_stage_overlap_segments += 1
                same_stage_overlap_minutes += duration_minutes

            if len(active_known) == 1:
                resolved_stage = active_known[0]
                if has_explicit_unknown:
                    explicit_unknown_suppressed_minutes += duration_minutes
            elif len(active_known) == 0:
                resolved_stage = "unknown"
                explicit_unknown_only_minutes += duration_minutes
            else:
                resolved_stage = "unknown"
                conflict_intervals.append((previous, boundary))
                conflict_minutes += duration_minutes
                conflict_combinations["+".join(active_known)] += 1
            atomic_events.append((previous, boundary, resolved_stage))

        for stage in changes[boundary]["end"]:
            active[stage] -= 1
            if active[stage] <= 0:
                del active[stage]
        for stage in changes[boundary]["start"]:
            active[stage] += 1
        previous = boundary

    resolved_events = _merge_adjacent_stage_intervals(atomic_events)
    merged_conflicts = _merge_adjacent_intervals(conflict_intervals)
    resolved_union_minutes = sum(
        float((end - start).total_seconds() / 60.0)
        for start, end, _ in resolved_events
    )
    audit: Dict[str, Any] = {
        "n_invalid_candidate_events": int(invalid_count),
        "n_valid_candidate_events": int(len(valid)),
        "candidate_event_minutes": float(candidate_minutes),
        "n_exact_duplicate_groups": int(duplicate_groups),
        "n_exact_duplicate_events_removed": int(duplicate_extras),
        "n_unique_events_after_exact_deduplication": int(len(unique_events)),
        "unique_event_minutes_before_overlap_resolution": float(unique_event_minutes),
        "n_atomic_segments_before_merge": int(len(atomic_events)),
        "n_same_stage_overlap_atomic_segments": int(same_stage_overlap_segments),
        "same_stage_overlap_minutes": float(same_stage_overlap_minutes),
        "n_cross_stage_conflict_atomic_segments": int(len(conflict_intervals)),
        "n_cross_stage_conflict_intervals_after_merge": int(len(merged_conflicts)),
        "cross_stage_conflict_minutes": float(conflict_minutes),
        "cross_stage_conflict_combinations": dict(sorted(conflict_combinations.items())),
        "explicit_unknown_only_minutes": float(explicit_unknown_only_minutes),
        "explicit_unknown_suppressed_by_known_stage_minutes": float(
            explicit_unknown_suppressed_minutes
        ),
        "n_resolved_intervals": int(len(resolved_events)),
        "resolved_union_minutes": float(resolved_union_minutes),
        "overlap_resolution_policy": (
            "exact_dedup_then_atomic_union; same_stage_union; "
            "multiple_known_stages_to_unknown; explicit_unknown_yields_to_one_known_stage"
        ),
    }
    return resolved_events, merged_conflicts, audit


def _sleep_events_with_audit(
    obj: Optional[dict],
) -> Tuple[List[SleepEvent], List[Tuple[pd.Timestamp, pd.Timestamp]], Dict[str, Any]]:
    candidates, parse_audit = _sleep_event_candidates(obj)
    resolved, conflicts, resolution_audit = _resolve_sleep_events(candidates)
    return resolved, conflicts, {**parse_audit, **resolution_audit}


def _sleep_events(obj: Optional[dict]) -> List[SleepEvent]:
    """Compatibility helper returning canonical, resolved sleep events."""

    events, _, _ = _sleep_events_with_audit(obj)
    return events


def allocate_sleep_events(
    events: Sequence[SleepEvent],
    anchor: ParticipantAnchor,
    spec: TimeGridSpec,
    conflict_intervals: Sequence[Tuple[pd.Timestamp, pd.Timestamp]] = (),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    n = spec.seq_len
    # Accumulate fragmented atomic intervals in float64.  Float32 incremental
    # addition can turn an exact five-minute partition into >5 minutes and
    # falsely trip the structural invariant; cast once after validation.
    stage_minutes_work = np.zeros((n, len(SLEEP_STAGE_NAMES)), dtype=np.float64)
    stage_mask = np.zeros_like(stage_minutes_work, dtype=bool)
    observed_mask = np.zeros(n, dtype=bool)
    conflict_mask = np.zeros(n, dtype=bool)
    anchor_utc = anchor.utc_timestamp()
    window_end = anchor_utc + pd.Timedelta(seconds=n * spec.slot_seconds)

    audit: Dict[str, Any] = {
        "n_resolved_events_for_allocation": int(len(events)),
        "n_resolved_invalid_events_for_allocation": 0,
        "n_resolved_events_before_window": 0,
        "n_resolved_events_after_window": 0,
        "n_resolved_events_partially_in_window": 0,
        "n_resolved_events_in_window": 0,
        "n_resolved_unknown_stage_events": 0,
        "resolved_minutes_in_window": 0.0,
        "resolved_minutes_outside_window": 0.0,
    }

    for start, end, stage in events:
        if pd.isna(start) or pd.isna(end) or end <= start:
            audit["n_resolved_invalid_events_for_allocation"] += 1
            continue
        stage = stage if stage in SLEEP_STAGE_NAMES else "unknown"
        if stage == "unknown":
            audit["n_resolved_unknown_stage_events"] += 1
        total_minutes = float((end - start).total_seconds() / 60.0)
        clipped_start = max(start, anchor_utc)
        clipped_end = min(end, window_end)
        if clipped_end <= clipped_start:
            if end <= anchor_utc:
                audit["n_resolved_events_before_window"] += 1
            else:
                audit["n_resolved_events_after_window"] += 1
            audit["resolved_minutes_outside_window"] += total_minutes
            continue
        if start < anchor_utc or end > window_end:
            audit["n_resolved_events_partially_in_window"] += 1
        audit["n_resolved_events_in_window"] += 1

        first_slot = max(0, _point_slot(clipped_start, anchor_utc, spec))
        last_slot = min(
            n - 1,
            int(
                math.floor(
                    ((clipped_end - anchor_utc).total_seconds() - 1e-9)
                    / spec.slot_seconds
                )
            ),
        )
        stage_index = SLEEP_STAGE_NAMES.index(stage)
        minutes_in = 0.0
        for slot in range(first_slot, last_slot + 1):
            slot_start = anchor_utc + pd.Timedelta(seconds=slot * spec.slot_seconds)
            slot_end = slot_start + pd.Timedelta(seconds=spec.slot_seconds)
            overlap_minutes = max(
                0.0,
                float((min(end, slot_end) - max(start, slot_start)).total_seconds() / 60.0),
            )
            if overlap_minutes <= 0:
                continue
            stage_minutes_work[slot, stage_index] += overlap_minutes
            stage_mask[slot, stage_index] = True
            observed_mask[slot] = True
            minutes_in += overlap_minutes
        audit["resolved_minutes_in_window"] += minutes_in
        audit["resolved_minutes_outside_window"] += max(
            0.0, total_minutes - minutes_in
        )

    any_minutes_work = stage_minutes_work.sum(axis=1)
    n_mathematical_violations = int(
        (any_minutes_work > float(spec.grid_minutes) + 1e-9).sum()
    )
    if n_mathematical_violations:
        raise ValueError(
            "resolved sleep stages exceed the grid duration in at least one slot"
        )
    stage_minutes = stage_minutes_work.astype(np.float32)
    any_minutes = stage_minutes.astype(np.float64).sum(axis=1)
    conflict_minutes_in_window = 0.0
    for start, end in conflict_intervals:
        clipped_start = max(start, anchor_utc)
        clipped_end = min(end, window_end)
        if clipped_end <= clipped_start:
            continue
        first_slot = max(0, _point_slot(clipped_start, anchor_utc, spec))
        last_slot = min(
            n - 1,
            int(
                math.floor(
                    ((clipped_end - anchor_utc).total_seconds() - 1e-9)
                    / spec.slot_seconds
                )
            ),
        )
        for slot in range(first_slot, last_slot + 1):
            slot_start = anchor_utc + pd.Timedelta(seconds=slot * spec.slot_seconds)
            slot_end = slot_start + pd.Timedelta(seconds=spec.slot_seconds)
            overlap_minutes = max(
                0.0,
                float(
                    (min(end, slot_end) - max(start, slot_start)).total_seconds()
                    / 60.0
                ),
            )
            if overlap_minutes <= 0:
                continue
            conflict_mask[slot] = True
            conflict_minutes_in_window += overlap_minutes

    audit["resolved_minutes_tensor_sum"] = float(any_minutes.sum())
    audit["n_observed_slots"] = int(observed_mask.sum())
    audit["n_slots_with_overlapping_stages"] = int(
        (any_minutes > float(spec.grid_minutes) + 1e-5).sum()
    )
    max_total = float(any_minutes.max()) if n else 0.0
    audit["max_total_stage_minutes_in_slot"] = max_total
    # Backward-compatible alias for V4 audits.  The old name was misleading:
    # the value has always been the total across all stages in a slot.
    audit["max_stage_minutes_in_slot"] = max_total
    audit["n_conflict_slots"] = int(conflict_mask.sum())
    audit["conflict_minutes_in_window"] = float(conflict_minutes_in_window)
    if audit["n_slots_with_overlapping_stages"]:
        raise ValueError(
            "float32 sleep tensor exceeds the grid duration beyond tolerance"
        )
    return stage_minutes, stage_mask, observed_mask, conflict_mask, audit


def build_activity(
    objects: Mapping[str, Optional[dict]] | Callable[[str], Optional[dict]],
    anchor: ParticipantAnchor,
    spec: TimeGridSpec,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    """Build the activity tensor from an eager mapping or a lazy source loader.

    The callable form lets the command-line pipeline load and release one large
    Garmin JSON file at a time.  This materially reduces peak memory for the
    5-second streams when several worker processes run in parallel.
    """
    get_object: Callable[[str], Optional[dict]]
    if callable(objects):
        get_object = objects
    else:
        get_object = objects.get

    n = spec.seq_len
    x = np.zeros((n, len(ACTIVITY_FEATURE_NAMES)), dtype=np.float32)
    feature_mask = np.zeros_like(x, dtype=bool)
    point_observed = np.zeros((n, len(POINT_SIGNAL_NAMES)), dtype=bool)
    point_valid = np.zeros_like(point_observed)
    interval_mask = np.zeros((n, len(INTERVAL_SIGNAL_NAMES)), dtype=bool)
    audit: Dict[str, Any] = {"point_signals": {}, "interval_signals": {}}

    for signal_index, signal in enumerate(POINT_SIGNAL_NAMES):
        body_key, value_path = POINT_SIGNAL_SPECS[signal]
        means, counts, offline, observed, valid, signal_audit = aggregate_point_signal(
            get_object(signal), body_key, value_path, anchor, spec
        )
        point_observed[:, signal_index] = observed
        point_valid[:, signal_index] = valid
        audit["point_signals"][signal] = signal_audit
        mean_name = f"activity__{signal}__mean"
        count_name = f"activity__{signal}__valid_count"
        offline_name = f"activity__{signal}__offline_count"
        x[:, ACTIVITY_FEATURE_INDEX[mean_name]] = means
        x[:, ACTIVITY_FEATURE_INDEX[count_name]] = counts
        x[:, ACTIVITY_FEATURE_INDEX[offline_name]] = offline
        feature_mask[:, ACTIVITY_FEATURE_INDEX[mean_name]] = valid
        feature_mask[:, ACTIVITY_FEATURE_INDEX[count_name]] = observed
        feature_mask[:, ACTIVITY_FEATURE_INDEX[offline_name]] = observed

    quantity_specs = {
        "steps": ("activity", ("base_movement_quantity",)),
        "calories": ("activity", ("calories_value", "kcal_burned")),
    }
    for signal_index, signal in enumerate(INTERVAL_SIGNAL_NAMES):
        body_key, fields = quantity_specs[signal]
        events = _quantity_events(get_object(signal), body_key, fields)
        values, event_counts, mask, signal_audit = allocate_quantity_events(
            events, anchor, spec
        )
        interval_mask[:, signal_index] = mask
        audit["interval_signals"][signal] = signal_audit
        value_name = f"activity__{signal}__allocated_value"
        count_name = f"activity__{signal}__event_count"
        x[:, ACTIVITY_FEATURE_INDEX[value_name]] = values
        x[:, ACTIVITY_FEATURE_INDEX[count_name]] = event_counts
        feature_mask[:, ACTIVITY_FEATURE_INDEX[value_name]] = mask
        feature_mask[:, ACTIVITY_FEATURE_INDEX[count_name]] = mask

    sleep_events, sleep_conflicts, sleep_resolution_audit = _sleep_events_with_audit(
        get_object("sleep")
    )
    (
        stage_minutes,
        stage_mask,
        sleep_observed,
        sleep_conflict,
        sleep_allocation_audit,
    ) = allocate_sleep_events(
        sleep_events,
        anchor,
        spec,
        conflict_intervals=sleep_conflicts,
    )
    audit["sleep"] = {**sleep_resolution_audit, **sleep_allocation_audit}
    any_sleep_name = "activity__sleep__any_minutes"
    x[:, ACTIVITY_FEATURE_INDEX[any_sleep_name]] = stage_minutes.sum(axis=1)
    feature_mask[:, ACTIVITY_FEATURE_INDEX[any_sleep_name]] = sleep_observed
    for stage_index, stage in enumerate(SLEEP_STAGE_NAMES):
        name = f"activity__sleep__{stage}_minutes"
        x[:, ACTIVITY_FEATURE_INDEX[name]] = stage_minutes[:, stage_index]
        # Once a slot has sleep-stage coverage, a stage that did not occur is
        # an observed zero rather than a missing value.  Keep ``stage_mask``
        # separately as the positive-occurrence audit mask, but expose all
        # stage values in a sleep-observed slot to scaling/model code.
        feature_mask[:, ACTIVITY_FEATURE_INDEX[name]] = sleep_observed

    time_features = local_time_features(anchor, spec)
    for name in LOCAL_TIME_FEATURE_NAMES:
        x[:, ACTIVITY_FEATURE_INDEX[name]] = time_features[name]
        feature_mask[:, ACTIVITY_FEATURE_INDEX[name]] = True
    utc_offsets = np.asarray(time_features["time__utc_offset_hours"], dtype=np.float32)
    audit["local_time"] = {
        "timezone": anchor.timezone,
        "utc_offset_hours_first": float(utc_offsets[0]) if utc_offsets.size else None,
        "utc_offset_hours_last": float(utc_offsets[-1]) if utc_offsets.size else None,
        "utc_offset_hours_unique": [
            float(value) for value in np.unique(utc_offsets).tolist()
        ],
        "n_utc_offset_transitions": int(np.count_nonzero(np.diff(utc_offsets)))
        if utc_offsets.size > 1
        else 0,
        "utc_offset_in_model_tensor": False,
    }

    observed_mask = (
        point_observed.any(axis=1) | interval_mask.any(axis=1) | sleep_observed
    )
    valid_mask = point_valid.any(axis=1) | interval_mask.any(axis=1) | sleep_observed
    audit["n_observed_slots"] = int(observed_mask.sum())
    audit["n_valid_slots"] = int(valid_mask.sum())
    audit["observed_gap_audit"] = _gap_audit(observed_mask)
    masks = {
        "point_observed": point_observed,
        "point_valid": point_valid,
        "interval": interval_mask,
        "sleep_stage": stage_mask,
        "sleep_observed": sleep_observed,
        "sleep_conflict": sleep_conflict,
    }
    return x, observed_mask, valid_mask, feature_mask, masks, audit


def _load_json(path: Optional[Path], source: str) -> Tuple[Optional[dict], Optional[Dict[str, Any]]]:
    if path is None:
        return None, None
    try:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            obj = json.load(handle)
        if not isinstance(obj, dict):
            raise ValueError("top-level JSON value is not an object")
        return obj, None
    except Exception as exc:
        return None, {"source": source, "path": str(path), "error": repr(exc)}


def _source_paths(
    dataset_root: Path,
    glucose_row: Optional[Mapping[str, Any]],
    activity_row: Optional[Mapping[str, Any]],
) -> Dict[str, Optional[Path]]:
    paths: Dict[str, Optional[Path]] = {
        "cgm": resolve_path(
            dataset_root,
            glucose_row.get("glucose_filepath") if glucose_row is not None else None,
        )
    }
    for source, column in ACTIVITY_PATH_COLUMNS.items():
        paths[source] = resolve_path(
            dataset_root,
            activity_row.get(column) if activity_row is not None else None,
        )
    return paths


def _provenance_files(paths: Mapping[str, Optional[Path]]) -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    for source, path in paths.items():
        if path is None:
            files.append({"source": source, "path": None, "declared": False, "exists": False})
            continue
        metadata = source_file_metadata([path])[0]
        metadata.update({"source": source, "declared": True})
        files.append(metadata)
    return files


def _torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.6
        return torch.load(path, map_location="cpu")


def _cache_is_current(
    out_path: Path,
    config_hash: str,
    anchor: ParticipantAnchor,
    provenance_files: List[Dict[str, Any]],
    glucose_manifest_row_count: int,
    activity_manifest_row_count: int,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    if not out_path.exists():
        return False, None
    try:
        obj = _torch_load(out_path)
        current = (
            obj.get("artifact_format") == ARTIFACT_FORMAT
            and obj.get("preprocessing_config", {}).get("config_hash") == config_hash
            and obj.get("anchor", {}).get("anchor_utc") == anchor.anchor_utc
            and obj.get("source_provenance", {}).get("files") == provenance_files
            and obj.get("source_provenance", {}).get("glucose_manifest_row_count")
            == glucose_manifest_row_count
            and obj.get("source_provenance", {}).get("activity_manifest_row_count")
            == activity_manifest_row_count
        )
        return bool(current), obj if current else None
    except Exception:
        return False, None


def _index_result(
    *,
    status: str,
    person_id: str,
    out_path: Path,
    anchor: ParticipantAnchor,
    config_hash: str,
    cgm_audit: Mapping[str, Any],
    activity_audit: Mapping[str, Any],
    has_glucose_manifest_row: bool,
    has_activity_manifest_row: bool,
    source_failures: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    sleep_audit = activity_audit.get("sleep", {})
    if not isinstance(sleep_audit, Mapping):
        sleep_audit = {}
    return {
        "status": status,
        "person_id": person_id,
        "out_path": str(out_path),
        "clinical_site": anchor.clinical_site,
        "timezone": anchor.timezone,
        "anchor_local": anchor.anchor_local,
        "anchor_utc": anchor.anchor_utc,
        "has_glucose_manifest_row": bool(has_glucose_manifest_row),
        "has_activity_manifest_row": bool(has_activity_manifest_row),
        "n_cgm_observed_slots": int(cgm_audit.get("n_observed_slots", 0)),
        "n_cgm_valid_slots": int(cgm_audit.get("n_valid_numeric_slots", 0)),
        "n_activity_observed_slots": int(activity_audit.get("n_observed_slots", 0)),
        "n_activity_valid_slots": int(activity_audit.get("n_valid_slots", 0)),
        "n_sleep_observed_slots": int(sleep_audit.get("n_observed_slots", 0)),
        "n_sleep_conflict_slots": int(sleep_audit.get("n_conflict_slots", 0)),
        "n_sleep_exact_duplicate_events_removed": int(
            sleep_audit.get("n_exact_duplicate_events_removed", 0)
        ),
        "sleep_cross_stage_conflict_minutes_all_source": float(
            sleep_audit.get("cross_stage_conflict_minutes", 0.0)
        ),
        "sleep_cross_stage_conflict_minutes_in_window": float(
            sleep_audit.get("conflict_minutes_in_window", 0.0)
        ),
        "max_total_sleep_stage_minutes_in_slot": float(
            sleep_audit.get("max_total_stage_minutes_in_slot", 0.0)
        ),
        "n_source_failures": int(len(source_failures)),
        "config_hash": config_hash,
    }


def process_person(
    person_id: str,
    glucose_row: Optional[Dict[str, Any]],
    activity_row: Optional[Dict[str, Any]],
    anchor: ParticipantAnchor,
    spec: TimeGridSpec,
    config: Dict[str, Any],
    dataset_root_str: str,
    out_dir_str: str,
    plausible_min: float,
    plausible_max: float,
    skip_existing: bool,
    input_provenance: Dict[str, Any],
) -> Dict[str, Any]:
    if torch is None:
        raise ImportError("PyTorch is required to serialize wearable .pt artifacts")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    dataset_root = Path(dataset_root_str)
    out_path = Path(out_dir_str) / f"{person_id}.pt"
    paths = _source_paths(dataset_root, glucose_row, activity_row)
    provenance_files = _provenance_files(paths)
    glucose_manifest_row_count = int(
        glucose_row.get("__manifest_row_count", 0) if glucose_row else 0
    )
    activity_manifest_row_count = int(
        activity_row.get("__manifest_row_count", 0) if activity_row else 0
    )
    if skip_existing:
        current, existing = _cache_is_current(
            out_path,
            str(config["config_hash"]),
            anchor,
            provenance_files,
            glucose_manifest_row_count,
            activity_manifest_row_count,
        )
        if current and existing is not None:
            audit = existing.get("audit", {})
            cached_source_failures = audit.get("source_failures", [])
            return {
                "index": _index_result(
                    status="skipped_current",
                    person_id=person_id,
                    out_path=out_path,
                    anchor=anchor,
                    config_hash=str(config["config_hash"]),
                    cgm_audit=audit.get("cgm", {}),
                    activity_audit=audit.get("activity", {}),
                    has_glucose_manifest_row=glucose_row is not None,
                    has_activity_manifest_row=activity_row is not None,
                    source_failures=cached_source_failures,
                ),
                # Preserve source failures in a regenerated run-level failure
                # report even when the participant artifact itself is current.
                "source_failures": cached_source_failures,
            }

    source_failures: List[Dict[str, Any]] = []
    cgm_obj, error = _load_json(paths["cgm"], "cgm")
    if error is not None:
        error["person_id"] = person_id
        source_failures.append(error)

    cgm, cgm_observed, cgm_valid, cgm_feature_mask, cgm_audit = aggregate_cgm(
        cgm_obj, anchor, spec, plausible_min, plausible_max
    )
    # CGM can itself be a large JSON document; release it before reading the
    # high-frequency Garmin streams.
    del cgm_obj

    def load_activity_source(source: str) -> Optional[dict]:
        obj, source_error = _load_json(paths[source], source)
        if source_error is not None:
            source_error["person_id"] = person_id
            source_failures.append(source_error)
        return obj

    (
        activity,
        activity_observed,
        activity_valid,
        activity_feature_mask,
        activity_masks,
        activity_audit,
    ) = build_activity(load_activity_source, anchor, spec)

    # A source is requested exactly once by build_activity.  Guard against a
    # future refactor accidentally duplicating error entries for the same file.
    if source_failures:
        deduplicated: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for source_error in source_failures:
            key = (
                str(source_error.get("source", "")),
                str(source_error.get("path", "")),
                str(source_error.get("error", "")),
            )
            deduplicated[key] = source_error
        source_failures = list(deduplicated.values())

    # Any non-finite generated value indicates a preprocessing bug.  Do not
    # silently replace it in a canonical artifact.
    if not np.isfinite(cgm).all():
        raise ValueError("non-finite value generated in CGM tensor")
    if not np.isfinite(activity).all():
        raise ValueError("non-finite value generated in activity tensor")

    artifact = {
        "artifact_format": ARTIFACT_FORMAT,
        "person_id": person_id,
        "cgm": torch.from_numpy(cgm),
        "cgm_observed_mask": torch.from_numpy(cgm_observed),
        "cgm_valid_mask": torch.from_numpy(cgm_valid),
        "cgm_feature_mask": torch.from_numpy(cgm_feature_mask),
        "cgm_feature_names": CGM_FEATURE_NAMES,
        "activity": torch.from_numpy(activity),
        "activity_observed_mask": torch.from_numpy(activity_observed),
        "activity_valid_mask": torch.from_numpy(activity_valid),
        "activity_feature_mask": torch.from_numpy(activity_feature_mask),
        "activity_point_observed_mask": torch.from_numpy(activity_masks["point_observed"]),
        "activity_point_valid_mask": torch.from_numpy(activity_masks["point_valid"]),
        "activity_interval_mask": torch.from_numpy(activity_masks["interval"]),
        "sleep_stage_mask": torch.from_numpy(activity_masks["sleep_stage"]),
        "sleep_observed_mask": torch.from_numpy(activity_masks["sleep_observed"]),
        "sleep_conflict_mask": torch.from_numpy(activity_masks["sleep_conflict"]),
        "activity_feature_names": ACTIVITY_FEATURE_NAMES,
        "activity_point_signal_names": POINT_SIGNAL_NAMES,
        "activity_interval_signal_names": INTERVAL_SIGNAL_NAMES,
        "sleep_stage_names": SLEEP_STAGE_NAMES,
        "anchor": asdict(anchor),
        "time_grid": {
            **asdict(spec),
            "seq_len": int(spec.seq_len),
            "slot_seconds": int(spec.slot_seconds),
        },
        "preprocessing_config": config,
        "source_paths": {key: str(value) if value is not None else None for key, value in paths.items()},
        "source_provenance": {
            **input_provenance,
            "glucose_manifest_row_count": glucose_manifest_row_count,
            "activity_manifest_row_count": activity_manifest_row_count,
            "files": provenance_files,
        },
        "audit": {
            "cgm": cgm_audit,
            "activity": activity_audit,
            "source_failures": source_failures,
        },
    }

    temp_path = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}")
    try:
        torch.save(artifact, temp_path)
        temp_path.replace(out_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    status = "processed_with_source_errors" if source_failures else "processed"
    if skip_existing and out_path.exists():
        # ``out_path`` necessarily exists after saving; stale/current status is
        # already handled above, so the ordinary processed label is clearer.
        status = "processed_with_source_errors" if source_failures else "processed"
    return {
        "index": _index_result(
            status=status,
            person_id=person_id,
            out_path=out_path,
            anchor=anchor,
            config_hash=str(config["config_hash"]),
            cgm_audit=cgm_audit,
            activity_audit=activity_audit,
            has_glucose_manifest_row=glucose_row is not None,
            has_activity_manifest_row=activity_row is not None,
            source_failures=source_failures,
        ),
        "source_failures": source_failures,
    }


def _manifest_map(
    frame: pd.DataFrame, manifest_name: str
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    if "person_id" not in frame.columns:
        raise ValueError(f"{manifest_name} missing required person_id column")
    records_by_id: Dict[str, List[Dict[str, Any]]] = {}
    issues: List[Dict[str, Any]] = []
    for row_number, record in enumerate(frame.to_dict("records"), start=2):
        person_id = as_person_id(record.get("person_id"))
        if not person_id:
            issues.append(
                {
                    "type": "manifest_invalid_person_id",
                    "manifest": manifest_name,
                    "row_number": row_number,
                }
            )
            continue
        records_by_id.setdefault(person_id, []).append(record)

    result: Dict[str, Dict[str, Any]] = {}
    for person_id, records in records_by_id.items():
        merged: Dict[str, Any] = {"person_id": person_id, "__manifest_row_count": len(records)}
        columns = sorted({column for record in records for column in record})
        for column in columns:
            if column == "person_id":
                continue
            values = [record.get(column) for record in records if _valid_manifest_value(record.get(column))]
            if values:
                merged[column] = values[0]
                distinct = list(dict.fromkeys(str(value).strip() for value in values))
                if len(distinct) > 1:
                    issues.append(
                        {
                            "type": "manifest_conflicting_values",
                            "manifest": manifest_name,
                            "person_id": person_id,
                            "column": column,
                            "values": distinct,
                            "selected": distinct[0],
                        }
                    )
            else:
                merged[column] = None
        result[person_id] = merged
    return result, issues


def _sort_person_ids(values: Iterable[str]) -> List[str]:
    return sorted(values, key=lambda value: (0, int(value)) if value.isdigit() else (1, value))


def _input_file_metadata(path: str | Path) -> Dict[str, Any]:
    return source_file_metadata([Path(path)])[0]


def _failed_index_row(
    person_id: str,
    out_dir: Path,
    anchor: ParticipantAnchor,
    config_hash: str,
    error: Exception,
    has_glucose_manifest_row: bool,
    has_activity_manifest_row: bool,
) -> Dict[str, Any]:
    return {
        "status": "failed",
        "person_id": person_id,
        "out_path": str(out_dir / f"{person_id}.pt"),
        "clinical_site": anchor.clinical_site,
        "timezone": anchor.timezone,
        "anchor_local": anchor.anchor_local,
        "anchor_utc": anchor.anchor_utc,
        "has_glucose_manifest_row": bool(has_glucose_manifest_row),
        "has_activity_manifest_row": bool(has_activity_manifest_row),
        "n_cgm_observed_slots": 0,
        "n_cgm_valid_slots": 0,
        "n_activity_observed_slots": 0,
        "n_activity_valid_slots": 0,
        "n_sleep_observed_slots": float("nan"),
        "n_sleep_conflict_slots": float("nan"),
        "n_sleep_exact_duplicate_events_removed": float("nan"),
        "sleep_cross_stage_conflict_minutes_all_source": float("nan"),
        "sleep_cross_stage_conflict_minutes_in_window": float("nan"),
        "max_total_sleep_stage_minutes_in_slot": float("nan"),
        "n_source_failures": 1,
        "config_hash": config_hash,
        "error": repr(error),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build separate canonical 5-minute CGM and activity tensors."
    )
    parser.add_argument("--participants", required=True, help="Path to participants.tsv")
    parser.add_argument(
        "--anchor_overrides",
        "--anchor-overrides",
        dest="anchor_overrides",
        default=None,
        help="Optional CSV/TSV with person_id,anchor_utc shared across longitudinal builders",
    )
    parser.add_argument(
        "--glucose_manifest",
        "--glucose-manifest",
        dest="glucose_manifest",
        required=True,
    )
    parser.add_argument(
        "--activity_manifest",
        "--activity-manifest",
        dest="activity_manifest",
        required=True,
    )
    parser.add_argument("--dataset_root", "--dataset-root", dest="dataset_root", required=True)
    parser.add_argument("--out_dir", "--out-dir", dest="out_dir", required=True)
    parser.add_argument("--grid_minutes", "--grid-minutes", dest="grid_minutes", type=int, default=5)
    parser.add_argument("--window_days", "--window-days", dest="window_days", type=int, default=10)
    parser.add_argument(
        "--anchor_hour_local",
        "--anchor-hour-local",
        dest="anchor_hour_local",
        type=int,
        default=0,
    )
    parser.add_argument("--cgm_plausible_min", type=float, default=40.0)
    parser.add_argument("--cgm_plausible_max", type=float, default=400.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=1)
    parser.add_argument("--skip_existing", "--skip-existing", dest="skip_existing", action="store_true")
    parser.add_argument("--no_progress", "--no-progress", dest="no_progress", action="store_true")
    args = parser.parse_args()

    if torch is None:
        parser.error("PyTorch is required to serialize wearable .pt artifacts")

    if args.cgm_plausible_min >= args.cgm_plausible_max:
        parser.error("--cgm_plausible_min must be less than --cgm_plausible_max")
    if args.num_workers < 1:
        parser.error("--num_workers must be at least 1")

    spec = TimeGridSpec(
        grid_minutes=args.grid_minutes,
        window_days=args.window_days,
        anchor_hour_local=args.anchor_hour_local,
    )
    anchor_overrides = read_anchor_overrides(args.anchor_overrides)
    anchors = load_participant_anchors(
        args.participants,
        spec,
        anchor_overrides_utc=anchor_overrides,
    )
    glucose = pd.read_csv(
        args.glucose_manifest, sep="\t", dtype={"person_id": "string"}
    )
    activity = pd.read_csv(
        args.activity_manifest, sep="\t", dtype={"person_id": "string"}
    )
    glucose_map, glucose_issues = _manifest_map(glucose, "glucose_manifest")
    activity_map, activity_issues = _manifest_map(activity, "activity_manifest")

    person_ids = _sort_person_ids(anchors)
    if args.limit is not None:
        if args.limit < 0:
            parser.error("--limit must be non-negative")
        person_ids = person_ids[: args.limit]

    config = artifact_config(
        modality="wearable_cgm_activity_5min",
        spec=spec,
        extra={
            "artifact_format": ARTIFACT_FORMAT,
            "builder_version": BUILDER_VERSION,
            "cgm_feature_names": CGM_FEATURE_NAMES,
            "activity_feature_names": ACTIVITY_FEATURE_NAMES,
            "cgm_plausible_range_mg_dl": [
                float(args.cgm_plausible_min),
                float(args.cgm_plausible_max),
            ],
            "point_offline_rule": "finite_value_less_than_zero",
            "point_aggregation": "mean_of_finite_nonnegative_values_plus_raw_counts",
            "interval_aggregation": "quantity_allocated_proportional_to_wall_clock_overlap",
            "sleep_aggregation": "resolved_stage_minutes_by_wall_clock_overlap",
            "sleep_record_policy": (
                "fully_parseable_stage_summary_replaces_outer_direct_event; "
                "otherwise_fallback_to_valid_direct_event"
            ),
            "sleep_summary_outer_interval_policy": (
                "retain_parseable_summary_and_audit_end_difference_with_1_second_tolerance"
            ),
            "sleep_duration_unit_policy": (
                "explicit_minute_millisecond_second_hour_day_units_or_iso8601; "
                "unrecognized_explicit_units_are_invalid"
            ),
            "sleep_exact_duplicate_policy": (
                "deduplicate_by_utc_start_utc_end_normalized_stage"
            ),
            "sleep_overlap_policy": (
                "atomic_time_partition; same_stage_union; "
                "multiple_known_stages_to_unknown; "
                "explicit_unknown_yields_to_one_known_stage"
            ),
            "interpolation": "none",
            "clipping": "none",
            "normalization": "none_canonical_raw_units",
            "required_participant_policy": "all_rows_in_participants_tsv",
            "anchor_policy": {
                "default": "study_visit_local_midnight"
                if spec.anchor_hour_local == 0
                else f"study_visit_local_hour_{spec.anchor_hour_local:02d}",
                "override": "explicit_anchor_utc_override",
                "n_explicit_overrides": int(len(anchor_overrides)),
                "override_content_hash": canonical_json_hash(anchor_overrides),
            },
        },
    )

    out_dir = Path(args.out_dir).resolve()
    dataset_root = str(Path(args.dataset_root).resolve())
    out_dir.mkdir(parents=True, exist_ok=True)
    input_provenance = {
        "participants_tsv": _input_file_metadata(args.participants),
        "glucose_manifest": _input_file_metadata(args.glucose_manifest),
        "activity_manifest": _input_file_metadata(args.activity_manifest),
        "anchor_overrides": _input_file_metadata(args.anchor_overrides)
        if args.anchor_overrides is not None
        else None,
    }

    index_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = [*glucose_issues, *activity_issues]
    required_id_set = set(anchors)
    for manifest_name, manifest_ids in [
        ("glucose_manifest", set(glucose_map)),
        ("activity_manifest", set(activity_map)),
    ]:
        failures.extend(
            {
                "type": "manifest_person_not_in_participants",
                "manifest": manifest_name,
                "person_id": person_id,
            }
            for person_id in _sort_person_ids(manifest_ids - required_id_set)
        )

    def record_result(result: Dict[str, Any]) -> None:
        index_rows.append(result["index"])
        failures.extend(result.get("source_failures", []))

    if args.num_workers == 1:
        iterator: Iterable[str] = person_ids
        if tqdm is not None and not args.no_progress:
            iterator = tqdm(
                person_ids,
                total=len(person_ids),
                desc="Wearable 5-minute preprocessing",
                unit="person",
                dynamic_ncols=True,
            )
        for person_id in iterator:
            try:
                result = process_person(
                    person_id,
                    glucose_map.get(person_id),
                    activity_map.get(person_id),
                    anchors[person_id],
                    spec,
                    config,
                    dataset_root,
                    str(out_dir),
                    args.cgm_plausible_min,
                    args.cgm_plausible_max,
                    args.skip_existing,
                    input_provenance,
                )
                record_result(result)
            except Exception as exc:
                failures.append(
                    {"type": "participant_failure", "person_id": person_id, "error": repr(exc)}
                )
                index_rows.append(
                    _failed_index_row(
                        person_id,
                        out_dir,
                        anchors[person_id],
                        config["config_hash"],
                        exc,
                        person_id in glucose_map,
                        person_id in activity_map,
                    )
                )
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_person = {
                executor.submit(
                    process_person,
                    person_id,
                    glucose_map.get(person_id),
                    activity_map.get(person_id),
                    anchors[person_id],
                    spec,
                    config,
                    dataset_root,
                    str(out_dir),
                    args.cgm_plausible_min,
                    args.cgm_plausible_max,
                    args.skip_existing,
                    input_provenance,
                ): person_id
                for person_id in person_ids
            }
            future_iterator: Iterable[Any] = as_completed(future_to_person)
            if tqdm is not None and not args.no_progress:
                future_iterator = tqdm(
                    future_iterator,
                    total=len(future_to_person),
                    desc="Wearable 5-minute preprocessing",
                    unit="person",
                    dynamic_ncols=True,
                )
            for future in future_iterator:
                person_id = future_to_person[future]
                try:
                    record_result(future.result())
                except Exception as exc:
                    failures.append(
                        {"type": "participant_failure", "person_id": person_id, "error": repr(exc)}
                    )
                    index_rows.append(
                        _failed_index_row(
                            person_id,
                            out_dir,
                            anchors[person_id],
                            config["config_hash"],
                            exc,
                            person_id in glucose_map,
                            person_id in activity_map,
                        )
                    )

    index = pd.DataFrame(index_rows)
    if not index.empty:
        order = {person_id: i for i, person_id in enumerate(person_ids)}
        index["__order"] = index["person_id"].map(order)
        index = index.sort_values("__order").drop(columns="__order")
    index.to_csv(out_dir / "wearable_5min_index.csv", index=False)
    (out_dir / "wearable_5min_failures.json").write_text(
        json.dumps(failures, indent=2, default=str), encoding="utf-8"
    )

    manifest_person_ids = set(glucose_map) | set(activity_map)
    required_ids = set(anchors)
    status_counts = (
        {str(key): int(value) for key, value in index["status"].value_counts().items()}
        if not index.empty
        else {}
    )
    schema = {
        "artifact_format": ARTIFACT_FORMAT,
        "raw_modality": "longitudinal_time_series",
        "model_representation": "separate_regular_5min_cgm_and_activity_sequences",
        "required_participant_policy": "all_rows_in_participants_tsv",
        "time_grid": {
            **asdict(spec),
            "seq_len": int(spec.seq_len),
            "slot_seconds": int(spec.slot_seconds),
            "anchor_strategy": config["anchor_policy"],
            "anchor_overrides_provenance": input_provenance["anchor_overrides"],
        },
        "cgm": {
            "tensor_key": "cgm",
            "shape": [int(spec.seq_len), len(CGM_FEATURE_NAMES)],
            "feature_names": CGM_FEATURE_NAMES,
            "mask_keys": ["cgm_observed_mask", "cgm_valid_mask", "cgm_feature_mask"],
            "observed_mask_semantics": "at_least_one_timestamped_raw_record_in_slot",
            "valid_mask_semantics": "at_least_one_finite_float32_representable_numeric_value_in_slot",
            "duplicate_count_semantics": "max(raw_record_count_in_slot_minus_one, zero)",
            "censor_semantics": "case-insensitive Low/Lo/< and High/Hi/> strings; no value imputation",
            "plausible_range_mg_dl": [
                float(args.cgm_plausible_min),
                float(args.cgm_plausible_max),
            ],
            "interpolation": "none",
            "clipping": "none",
        },
        "activity": {
            "tensor_key": "activity",
            "shape": [int(spec.seq_len), len(ACTIVITY_FEATURE_NAMES)],
            "feature_names": ACTIVITY_FEATURE_NAMES,
            "mask_keys": [
                "activity_observed_mask",
                "activity_valid_mask",
                "activity_feature_mask",
                "activity_point_observed_mask",
                "activity_point_valid_mask",
                "activity_interval_mask",
                "sleep_stage_mask",
                "sleep_observed_mask",
                "sleep_conflict_mask",
            ],
            "point_signal_names": POINT_SIGNAL_NAMES,
            "interval_signal_names": INTERVAL_SIGNAL_NAMES,
            "sleep_stage_names": SLEEP_STAGE_NAMES,
            "point_semantics": "slot mean plus valid and offline raw counts; no expected cadence",
            "offline_semantics": "finite negative raw point value",
            "interval_semantics": "steps/calories totals allocated proportional to interval overlap",
            "sleep_semantics": (
                "minutes of deterministic resolved stage overlap per slot; exact "
                "duplicates and same-stage overlaps count once; conflicting known "
                "stages map to unknown"
            ),
            "sleep_stage_mask_semantics": "positive minutes for that stage in the slot",
            "sleep_stage_feature_mask_semantics": "all stage values, including known zeros, are valid whenever sleep_observed_mask is true",
            "sleep_conflict_mask_semantics": (
                "slot contains time covered by more than one conflicting known "
                "raw sleep stage; those minutes are represented in unknown"
            ),
            "time_feature_semantics": {
                "tensor_features": LOCAL_TIME_FEATURE_NAMES,
                "description": "local day cycle and window progress derived from each absolute UTC slot",
                "timezone_and_utc_offset": "metadata/audit only; excluded from model tensor to avoid a clinical-site shortcut",
            },
            "clipping": "none",
            "normalization": "none in canonical artifact; fit model scaling on training data only",
        },
        "preprocessing_config": config,
        "counts": {
            "n_participants_tsv": int(len(anchors)),
            "n_required_selected": int(len(person_ids)),
            "n_glucose_manifest_rows": int(len(glucose)),
            "n_activity_manifest_rows": int(len(activity)),
            "n_glucose_manifest_persons": int(len(glucose_map)),
            "n_activity_manifest_persons": int(len(activity_map)),
            "n_manifest_persons_not_in_participants": int(len(manifest_person_ids - required_ids)),
            "n_required_persons_missing_glucose_manifest": int(len(set(person_ids) - set(glucose_map))),
            "n_required_persons_missing_activity_manifest": int(len(set(person_ids) - set(activity_map))),
            "n_anchor_overrides_total": int(len(anchor_overrides)),
            "n_anchor_overrides_selected": int(len(set(person_ids) & set(anchor_overrides))),
            "n_failures_logged": int(len(failures)),
            "status_counts": status_counts,
        },
        "notes": [
            "CGM and activity share the same participant grid but remain separate tensor keys.",
            "Zero is a storage fill value; masks determine whether a raw measurement supports a feature.",
            "Deterministic local-time features do not make an otherwise missing activity slot observed.",
            "Timezone and UTC offset remain in anchor/audit metadata but are not model input features.",
            "No expected device cadence or count_fraction is guessed.",
            "No task labels are used.",
        ],
    }
    (out_dir / "wearable_5min_schema.json").write_text(
        json.dumps(schema, indent=2, default=str), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "config_hash": config["config_hash"],
                "seq_len": spec.seq_len,
                "cgm_shape": [spec.seq_len, len(CGM_FEATURE_NAMES)],
                "activity_shape": [spec.seq_len, len(ACTIVITY_FEATURE_NAMES)],
                "n_required": len(person_ids),
                "n_anchor_overrides": len(anchor_overrides),
                "status_counts": status_counts,
                "n_failures_logged": len(failures),
            },
            indent=2,
        )
    )
    if failures[:5]:
        print("first failures:", failures[:5])


if __name__ == "__main__":
    main()
