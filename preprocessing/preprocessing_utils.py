#!/usr/bin/env python3
"""Shared, task-independent preprocessing helpers for AI-READI V2 artifacts.

The V2 contract deliberately separates raw-to-canonical preprocessing from
MoCA/CVD label construction.  Longitudinal sources use one participant-level
absolute UTC grid, while circadian features are computed in local civil time.

The default grid is 5 minutes for 10 days (2880 slots).  The anchor is the
participant's study-visit date at local midnight.  This is deterministic and
shared by environment, CGM, and activity builders.  The chosen strategy is
written into every artifact so a future release can replace it with an exact
device-deployment timestamp without silently reusing stale caches.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


PREPROCESSING_VERSION = "aireadi-ot2-v2"
DEFAULT_GRID_MINUTES = 5
DEFAULT_WINDOW_DAYS = 10

# AI-READI v3 sites in participants.tsv.  Keep this explicit and fail on an
# unknown site rather than silently assigning a wrong circadian clock.
SITE_TIMEZONES: Dict[str, str] = {
    "UW": "America/Los_Angeles",
    "UCSD": "America/Los_Angeles",
    "UAB": "America/Chicago",
}


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


@dataclass(frozen=True)
class TimeGridSpec:
    grid_minutes: int = DEFAULT_GRID_MINUTES
    window_days: int = DEFAULT_WINDOW_DAYS
    anchor_hour_local: int = 0

    def __post_init__(self) -> None:
        if self.grid_minutes <= 0 or 1440 % self.grid_minutes != 0:
            raise ValueError("grid_minutes must be a positive divisor of 1440")
        if self.window_days <= 0:
            raise ValueError("window_days must be positive")
        if not 0 <= self.anchor_hour_local <= 23:
            raise ValueError("anchor_hour_local must be in [0, 23]")

    @property
    def seq_len(self) -> int:
        return self.window_days * 1440 // self.grid_minutes

    @property
    def slot_seconds(self) -> int:
        return self.grid_minutes * 60


@dataclass(frozen=True)
class ParticipantAnchor:
    person_id: str
    clinical_site: str
    timezone: str
    study_visit_date: str
    anchor_local: str
    anchor_utc: str
    strategy: str = "study_visit_local_midnight"

    def utc_timestamp(self) -> pd.Timestamp:
        ts = pd.Timestamp(self.anchor_utc)
        return ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")


def load_participant_anchors(
    participants_tsv: str | Path,
    spec: TimeGridSpec,
    site_timezones: Optional[Mapping[str, str]] = None,
    anchor_overrides_utc: Optional[Mapping[str, Any]] = None,
) -> Dict[str, ParticipantAnchor]:
    """Build a deterministic, shared UTC anchor for every participant.

    ``study_visit_date`` is a date rather than an exact device-deployment time.
    Consequently this function does not pretend otherwise: it records the
    strategy explicitly and leaves observed coverage/out-of-window counts for
    each modality builder to audit.
    """
    tz_map = dict(SITE_TIMEZONES if site_timezones is None else site_timezones)
    df = pd.read_csv(participants_tsv, sep="\t", dtype={"person_id": "string"})
    required = {"person_id", "clinical_site", "study_visit_date"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"participants.tsv missing required columns: {missing}")

    anchors: Dict[str, ParticipantAnchor] = {}
    normalized_overrides: Dict[str, Any] = {}
    for key, value in (anchor_overrides_utc or {}).items():
        normalized_key = as_person_id(key)
        if not normalized_key:
            raise ValueError("Anchor overrides contain an empty person_id")
        if normalized_key in normalized_overrides:
            raise ValueError(
                "Anchor override IDs collide after normalization: "
                f"person_id={normalized_key}"
            )
        normalized_overrides[normalized_key] = value
    for row in df.to_dict("records"):
        pid = as_person_id(row["person_id"])
        site = str(row["clinical_site"]).strip()
        if not pid:
            raise ValueError("participants.tsv contains an empty person_id")
        if pid in anchors:
            raise ValueError(f"participants.tsv contains duplicate person_id={pid}")
        if site not in tz_map:
            raise ValueError(
                f"No timezone mapping for clinical_site={site!r}; "
                "pass an explicit mapping before preprocessing"
            )
        visit = pd.to_datetime(row["study_visit_date"], errors="coerce")
        if pd.isna(visit):
            raise ValueError(f"Invalid study_visit_date for person_id={pid}")
        timezone_name = tz_map[site]
        if pid in normalized_overrides:
            utc = parse_timestamp_utc(normalized_overrides[pid])
            if pd.isna(utc):
                raise ValueError(f"Invalid explicit UTC anchor for person_id={pid}")
            local = utc.tz_convert(ZoneInfo(timezone_name))
            strategy = "explicit_anchor_utc_override"
        else:
            local_naive = pd.Timestamp(visit.date()) + pd.Timedelta(
                hours=spec.anchor_hour_local
            )
            local = local_naive.tz_localize(ZoneInfo(timezone_name))
            utc = local.tz_convert("UTC")
            strategy = (
                "study_visit_local_midnight"
                if spec.anchor_hour_local == 0
                else f"study_visit_local_hour_{spec.anchor_hour_local:02d}"
            )
        anchors[pid] = ParticipantAnchor(
            person_id=pid,
            clinical_site=site,
            timezone=timezone_name,
            study_visit_date=str(visit.date()),
            anchor_local=local.isoformat(),
            anchor_utc=utc.isoformat(),
            strategy=strategy,
        )
    unknown_overrides = sorted(set(normalized_overrides) - set(anchors))
    if unknown_overrides:
        raise ValueError(
            "Anchor overrides contain person IDs absent from participants.tsv: "
            f"{unknown_overrides[:20]}"
        )
    return anchors


def read_anchor_overrides(path: str | Path | None) -> Dict[str, str]:
    """Read an optional ``person_id,anchor_utc`` CSV/TSV override table.

    Exact device-deployment timestamps are preferable to visit-date midnight
    when the former are available.  Keeping them in one shared table prevents
    environment, CGM and activity builders from independently choosing
    different starts.
    """
    if path is None:
        return {}
    source = Path(path)
    separator = "\t" if source.suffix.lower() in {".tsv", ".tab"} else ","
    frame = pd.read_csv(source, sep=separator, dtype={"person_id": "string"})
    required = {"person_id", "anchor_utc"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Anchor override table missing required columns: {missing}")
    frame = frame.copy()
    frame["person_id"] = frame["person_id"].map(as_person_id)
    if (frame["person_id"] == "").any():
        raise ValueError("Anchor override table contains an empty person_id")
    duplicates = frame.loc[
        frame["person_id"].duplicated(keep=False), "person_id"
    ].unique()
    if len(duplicates):
        raise ValueError(
            "Anchor override table contains duplicate person IDs: "
            f"{sorted(duplicates.tolist())[:20]}"
        )
    parsed = pd.to_datetime(frame["anchor_utc"], errors="coerce", utc=True)
    if parsed.isna().any():
        bad = frame.loc[parsed.isna(), "person_id"].tolist()
        raise ValueError(f"Invalid anchor_utc for person IDs: {bad[:20]}")
    return {
        pid: timestamp.isoformat()
        for pid, timestamp in zip(frame["person_id"], parsed)
    }


def parse_timestamp_utc(value: Any) -> pd.Timestamp:
    """Parse a timestamp as UTC, returning ``NaT`` on malformed input."""
    if value is None:
        return pd.NaT
    try:
        return pd.to_datetime(value, errors="coerce", utc=True)
    except Exception:
        return pd.NaT


def slot_indices(
    timestamps: pd.Series,
    anchor_utc: pd.Timestamp,
    spec: TimeGridSpec,
) -> pd.Series:
    """Return nullable integer grid indices relative to ``anchor_utc``."""
    ordinary = pd.to_datetime(timestamps, errors="coerce", utc=True)
    try:
        mixed = pd.to_datetime(timestamps, errors="coerce", utc=True, format="mixed")
        ts = mixed if int(mixed.notna().sum()) >= int(ordinary.notna().sum()) else ordinary
    except (TypeError, ValueError):
        ts = ordinary
    seconds = (ts - anchor_utc).dt.total_seconds()
    return np.floor(seconds / spec.slot_seconds).astype("Int64")


def in_window(slot_index: pd.Series, spec: TimeGridSpec) -> pd.Series:
    return slot_index.notna() & (slot_index >= 0) & (slot_index < spec.seq_len)


def local_time_features(anchor: ParticipantAnchor, spec: TimeGridSpec) -> Dict[str, np.ndarray]:
    """Build local-clock/cycle features for an absolute UTC grid.

    ``time__day_index_norm`` describes progress through the fixed-duration
    model window.  It intentionally does not count local calendar midnights,
    because a daylight-saving transition would otherwise create a spurious
    extra/missing model day.  UTC offset is returned for auditing but should
    normally remain metadata rather than a model feature because it is a close
    proxy for collection site.
    """
    utc = pd.date_range(
        start=anchor.utc_timestamp(),
        periods=spec.seq_len,
        freq=f"{spec.grid_minutes}min",
        tz="UTC",
    )
    local = utc.tz_convert(anchor.timezone)
    minute_of_day = (local.hour * 60 + local.minute).to_numpy(dtype=np.float32)
    slots_per_day = 1440 // spec.grid_minutes
    day_index = (
        np.arange(spec.seq_len, dtype=np.int64) // slots_per_day
    ).astype(np.float32)
    denom = max(float(spec.window_days - 1), 1.0)
    return {
        "time__day_sin": np.sin(2.0 * np.pi * minute_of_day / 1440.0).astype(np.float32),
        "time__day_cos": np.cos(2.0 * np.pi * minute_of_day / 1440.0).astype(np.float32),
        "time__day_index_norm": (day_index / denom).astype(np.float32),
        "time__utc_offset_hours": np.asarray(
            [x.utcoffset().total_seconds() / 3600.0 for x in local.to_pydatetime()],
            dtype=np.float32,
        ),
    }


def canonical_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_file_metadata(paths: Iterable[str | Path | None]) -> list[dict[str, Any]]:
    """Cheap provenance for large raw files without re-hashing terabytes."""
    out: list[dict[str, Any]] = []
    for value in paths:
        if value is None:
            continue
        path = Path(value)
        item: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if path.exists():
            stat = path.stat()
            item.update({"size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
        out.append(item)
    return out


def artifact_config(
    *,
    modality: str,
    spec: Optional[TimeGridSpec] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "modality": modality,
    }
    if spec is not None:
        payload["time_grid"] = asdict(spec)
        payload["time_grid"]["seq_len"] = spec.seq_len
    if extra:
        payload.update(dict(extra))
    payload["config_hash"] = canonical_json_hash(payload)
    return payload
