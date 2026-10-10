#!/usr/bin/env python3
"""Build a task-aware master index over task-independent V2 artifacts.

The modality builders never read labels.  This final join is intentionally the
first place where a downstream label is associated with canonical modality
artifacts.  It uses ``participants.tsv`` as the only split authority and keeps
retinal assets as a list instead of forcing heterogeneous protocols into a
fixed eight-image layout.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import pandas as pd

try:
    from .preprocessing_utils import (
        PREPROCESSING_VERSION,
        as_person_id,
        canonical_json_hash,
        source_file_metadata,
    )
except ImportError:
    from preprocessing_utils import (  # type: ignore
        PREPROCESSING_VERSION,
        as_person_id,
        canonical_json_hash,
        source_file_metadata,
    )


ALLOWED_SPLITS = {"train", "val", "test"}
PARTICIPANT_AVAILABILITY = {
    "clinical_data": "expected_clinical",
    "cardiac_ecg": "expected_ecg",
    "environment": "expected_environment",
    "wearable_blood_glucose": "expected_cgm",
    "wearable_activity_monitor": "expected_activity",
    "retinal_photography": "expected_retinal_photography",
    "retinal_oct": "expected_retinal_oct",
    "retinal_octa": "expected_retinal_octa",
    "retinal_flio": "expected_retinal_flio",
}
ENVIRONMENT_COVERAGE_QC_TIERS = (
    "not_indexed",
    "artifact_missing",
    "zero",
    "gt0_lt10pct",
    "10_to_lt50pct",
    "50_to_lt90pct",
    "90_to_lt99pct",
    "99_to_100pct",
)
ENVIRONMENT_COVERAGE_QC_BOUNDARIES = (0.0, 0.10, 0.50, 0.90, 0.99, 1.0)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_people_table(path: str | Path, separator: str = ",") -> pd.DataFrame:
    frame = pd.read_csv(
        path, sep=separator, low_memory=False, dtype={"person_id": "string"}
    )
    if "person_id" not in frame:
        raise ValueError(f"{path} is missing person_id")
    frame = frame.copy()
    frame["person_id"] = frame["person_id"].map(as_person_id)
    if (frame["person_id"] == "").any():
        raise ValueError(f"{path} contains an empty person_id")
    return frame


def _require_unique_people(frame: pd.DataFrame, name: str) -> None:
    duplicates = sorted(
        frame.loc[frame["person_id"].duplicated(keep=False), "person_id"].unique()
    )
    if duplicates:
        raise ValueError(f"{name} has duplicate person IDs: {duplicates[:20]}")


def _require_known_people(
    frame: pd.DataFrame, name: str, participant_ids: set[str]
) -> None:
    unknown = sorted(set(frame["person_id"]) - participant_ids)
    if unknown:
        raise ValueError(
            f"{name} contains people absent from participants.tsv: {unknown[:20]}"
        )


def _successful_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "status" not in frame:
        raise ValueError("V2 artifact index is missing required status column")
    status = frame["status"].fillna("").astype(str).str.lower()
    successful_statuses = {
        "processed", "cached", "skipped", "skipped_current",
        "processed_with_source_errors",
    }
    failed = status.isin({"failed", "error"})
    unknown = sorted(set(status) - successful_statuses - {"failed", "error"})
    if unknown:
        raise ValueError(f"V2 artifact index contains unknown statuses: {unknown[:20]}")
    return frame.loc[~failed].copy()


def _strict_bool(values: pd.Series, source: str) -> pd.Series:
    def parse(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if pd.isna(value):
            raise ValueError(f"{source} contains a missing availability flag")
        text = str(value).strip().lower()
        if text in {"true", "1"}:
            return True
        if text in {"false", "0"}:
            return False
        raise ValueError(f"{source} contains a non-boolean availability flag: {value!r}")
    return values.map(parse)


def _strict_binary_labels(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    if not numeric.isin([0, 1]).all():
        bad = values.loc[~numeric.isin([0, 1])].tolist()
        raise ValueError(f"labels y must be exact binary 0/1; bad values={bad[:20]}")
    return numeric.astype(int)


def _pick_path_column(frame: pd.DataFrame, candidates: Sequence[str], name: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"{name} index lacks an artifact path column; tried {list(candidates)}")


def _portable_path(
    value: Any,
    artifact_root: Path | None,
    reference_dir: Path | None = None,
) -> str:
    if value is None or pd.isna(value) or not str(value).strip():
        return ""
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        absolute = path.resolve()
    elif reference_dir is not None:
        # Relative paths inside an index are defined relative to that index,
        # never to the caller's current working directory.
        absolute = (reference_dir / path).resolve()
    else:
        absolute = path.resolve()
    if artifact_root is None:
        return str(absolute)
    try:
        return str(absolute.resolve().relative_to(artifact_root.resolve()))
    except ValueError as exc:
        raise ValueError(
            f"Artifact path {absolute} is outside --artifact_root {artifact_root}"
        ) from exc


def _validate_paths(
    paths: Iterable[str], artifact_root: Path | None, *, allow_missing: bool, label: str
) -> None:
    missing = []
    for value in paths:
        if not value:
            continue
        path = Path(value)
        if artifact_root is not None and not path.is_absolute():
            path = artifact_root / path
        if not path.is_file():
            missing.append(str(path))
    if missing and not allow_missing:
        raise FileNotFoundError(
            f"{label} index references {len(missing)} missing artifacts; first={missing[:5]}"
        )


def _path_exists(value: str, artifact_root: Path | None) -> bool:
    if not value:
        return False
    path = Path(value)
    if artifact_root is not None and not path.is_absolute():
        path = artifact_root / path
    return path.is_file()


def _frame_content_hash(
    frame: pd.DataFrame, columns: Sequence[str], *, sort_columns: Sequence[str]
) -> str:
    selected = frame[list(columns)].copy()
    selected = selected.sort_values(list(sort_columns), kind="mergesort")
    selected = selected.astype(object).where(pd.notna(selected), None)
    return canonical_json_hash(
        {"columns": list(columns), "records": selected.to_dict("records")}
    )


def _validate_environment_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize Environment coverage carried by its V4 index."""
    required = {"n_observed_slots", "slot_coverage_fraction"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"environment index is missing coverage columns: {missing}"
        )

    out = frame.copy()
    slots = pd.to_numeric(out["n_observed_slots"], errors="coerce")
    fractions = pd.to_numeric(out["slot_coverage_fraction"], errors="coerce")
    valid_slots = slots.notna() & slots.ge(0) & slots.mod(1).eq(0)
    valid_fractions = fractions.notna() & fractions.between(0.0, 1.0)
    if not valid_slots.all():
        people = out.loc[~valid_slots, "person_id"].tolist()
        raise ValueError(
            "environment index has invalid n_observed_slots for people: "
            f"{people[:20]}"
        )
    if not valid_fractions.all():
        people = out.loc[~valid_fractions, "person_id"].tolist()
        raise ValueError(
            "environment index has nonfinite/out-of-range slot_coverage_fraction "
            f"for people: {people[:20]}"
        )
    inconsistent = slots.eq(0) != fractions.eq(0.0)
    if inconsistent.any():
        people = out.loc[inconsistent, "person_id"].tolist()
        raise ValueError(
            "environment index disagrees on zero observed slots and zero coverage "
            f"for people: {people[:20]}"
        )
    out["n_observed_slots"] = slots.astype("int64")
    out["slot_coverage_fraction"] = fractions.astype("float64")
    return out


def _environment_coverage_qc_tier(
    indexed: bool, artifact_exists: bool, fraction: Any
) -> str:
    """Return an audit-only tier without changing modality availability."""
    if not bool(indexed):
        return "not_indexed"
    if not bool(artifact_exists):
        return "artifact_missing"
    if pd.isna(fraction):
        raise ValueError("Indexed Environment artifact has no coverage fraction")
    value = float(fraction)
    if value == 0.0:
        return "zero"
    if value < 0.10:
        return "gt0_lt10pct"
    if value < 0.50:
        return "10_to_lt50pct"
    if value < 0.90:
        return "50_to_lt90pct"
    if value < 0.99:
        return "90_to_lt99pct"
    if value <= 1.0:
        return "99_to_100pct"
    raise ValueError(f"Invalid Environment coverage fraction: {value!r}")


def _single_artifact_view(
    path: str | Path,
    *,
    name: str,
    path_candidates: Sequence[str],
    artifact_root: Path | None,
    allow_missing: bool,
    participant_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    index_path = Path(path).resolve()
    raw_frame = _read_people_table(index_path)
    if participant_ids is not None:
        _require_known_people(raw_frame, f"{name} index", participant_ids)
    frame = _successful_rows(raw_frame)
    if name == "ecg" and frame["person_id"].duplicated().any():
        if "is_primary_record" in frame:
            primary = frame["is_primary_record"].astype(str).str.lower().isin({"true", "1"})
            frame = frame.loc[primary].copy()
        elif "duplicate_rank" in frame:
            rank = pd.to_numeric(frame["duplicate_rank"], errors="coerce")
            frame = frame.loc[rank.eq(frame.assign(_rank=rank).groupby("person_id")["_rank"].transform("min"))].copy()
    _require_unique_people(frame, f"{name} index after successful/primary filtering")
    path_col = _pick_path_column(frame, path_candidates, name)
    out = frame[["person_id", path_col]].copy()
    out[f"{name}_path"] = out[path_col].map(
        lambda x: _portable_path(x, artifact_root, index_path.parent)
    )
    out = out.drop(columns=[path_col])
    _validate_paths(
        out[f"{name}_path"], artifact_root, allow_missing=allow_missing, label=name
    )

    hash_col = next(
        (column for column in ("artifact_config_hash", "config_hash") if column in frame),
        None,
    )
    if hash_col is None:
        raise ValueError(f"{name} index has no preprocessing config hash column")
    row_hashes = frame[hash_col].fillna("").astype(str).str.strip()
    if row_hashes.eq("").any():
        bad_people = frame.loc[row_hashes.eq(""), "person_id"].tolist()
        raise ValueError(
            f"{name} index has successful rows without a config hash: {bad_people[:20]}"
        )
    hashes = sorted(row_hashes.unique())
    if len(hashes) != 1:
        raise ValueError(f"{name} index mixes preprocessing config hashes: {hashes[:5]}")
    if name == "environment":
        frame = _validate_environment_coverage(frame)
    out[f"{name}_config_hash"] = row_hashes.to_numpy()
    out[f"{name}_artifact_exists"] = out[f"{name}_path"].map(
        lambda value: _path_exists(value, artifact_root)
    )

    for source, destination in {
        "n_observed_slots": f"{name}_observed_slots",
        "slot_coverage_fraction": f"{name}_slot_coverage_fraction",
        "n_model_leads": "ecg_model_leads",
        "n_samples_output": "ecg_samples",
        "n_cgm_observed_slots": "cgm_observed_slots",
        "n_cgm_valid_slots": "cgm_valid_slots",
        "n_activity_observed_slots": "activity_observed_slots",
        "n_activity_valid_slots": "activity_valid_slots",
    }.items():
        if source in frame and destination not in out:
            out[destination] = frame[source].to_numpy()
    mapping_columns = sorted(out.columns)
    return out, {
        "path": str(index_path),
        "n_successful_people": int(len(out)),
        "n_existing_artifacts": int(out[f"{name}_artifact_exists"].sum()),
        "config_hash": hashes[0],
        "mapping_hash": _frame_content_hash(
            out, mapping_columns, sort_columns=["person_id"]
        ),
    }


def _retinal_view(
    path: str | Path,
    *,
    artifact_root: Path | None,
    allow_missing: bool,
    participant_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    index_path = Path(path).resolve()
    raw_frame = _read_people_table(index_path)
    if participant_ids is not None:
        _require_known_people(raw_frame, "retinal index", participant_ids)
    frame = _successful_rows(raw_frame)
    path_col = _pick_path_column(
        frame,
        ("artifact_path", "out_path", "preprocessed_path"),
        "retinal",
    )
    frame = frame.copy()
    hash_col = next(
        (column for column in ("artifact_config_hash", "config_hash") if column in frame),
        None,
    )
    if hash_col is None:
        raise ValueError("retinal index has no preprocessing config hash column")
    frame["_row_config_hash"] = frame[hash_col].fillna("").astype(str).str.strip()
    if frame["_row_config_hash"].eq("").any():
        raise ValueError("retinal index has successful rows without a config hash")
    hashes = sorted(frame["_row_config_hash"].unique())
    if len(hashes) != 1:
        raise ValueError(f"retinal index mixes preprocessing config hashes: {hashes[:5]}")
    if "asset_id" in frame and frame["asset_id"].duplicated().any():
        duplicate_ids = frame.loc[
            frame["asset_id"].duplicated(keep=False), "asset_id"
        ].tolist()
        raise ValueError(f"retinal index repeats asset_id values: {duplicate_ids[:20]}")
    frame["_path"] = frame[path_col].map(
        lambda x: _portable_path(x, artifact_root, index_path.parent)
    )
    frame = frame[frame["_path"].ne("")].copy()
    if frame["_path"].duplicated().any():
        duplicate_paths = frame.loc[
            frame["_path"].duplicated(keep=False), "_path"
        ].tolist()
        raise ValueError(
            f"retinal index repeats preprocessed artifact paths: {duplicate_paths[:20]}"
        )
    _validate_paths(frame["_path"], artifact_root, allow_missing=allow_missing, label="retinal")
    frame["_artifact_exists"] = frame["_path"].map(
        lambda value: _path_exists(value, artifact_root)
    )
    sort_cols = [c for c in ["person_id", "modality", "laterality", "protocol", "slab_or_wavelength", "_path"] if c in frame]
    frame = frame.sort_values(sort_cols, kind="mergesort")

    rows = []
    for person_id, group in frame.groupby("person_id", sort=True):
        existing = group[group["_artifact_exists"]]
        item: Dict[str, Any] = {
            "person_id": person_id,
            "retinal_asset_paths_json": json.dumps(existing["_path"].tolist(), separators=(",", ":")),
            "retinal_indexed_asset_count": int(len(group)),
            "retinal_asset_count": int(len(existing)),
        }
        if "modality" in existing:
            modalities = existing["modality"].fillna("").astype(str).str.lower()
            for modality in ("photography", "oct", "octa", "flio"):
                item[f"retinal_{modality}_asset_count"] = int(modalities.eq(modality).sum())
        rows.append(item)
    out = pd.DataFrame(
        rows,
        columns=(
            [
                "person_id", "retinal_asset_paths_json", "retinal_asset_count",
                "retinal_indexed_asset_count",
                "retinal_photography_asset_count", "retinal_oct_asset_count",
                "retinal_octa_asset_count", "retinal_flio_asset_count",
            ]
            if not rows
            else None
        ),
    )
    mapping_columns = sorted(out.columns)
    return out, {
        "path": str(index_path),
        "n_successful_assets": int(len(frame)),
        "n_existing_artifacts": int(frame["_artifact_exists"].sum()),
        "n_successful_people": int(out["person_id"].nunique()) if not out.empty else 0,
        "config_hash": hashes[0],
        "mapping_hash": _frame_content_hash(
            out, mapping_columns, sort_columns=["person_id"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--clinical_features", required=True)
    parser.add_argument("--clinical_schema", required=True)
    parser.add_argument("--retinal_index")
    parser.add_argument("--environment_index")
    parser.add_argument("--wearable_index")
    parser.add_argument("--ecg_index")
    parser.add_argument("--artifact_root")
    parser.add_argument("--allow_missing_artifacts", action="store_true")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--metadata_json")
    args = parser.parse_args()

    participants = _read_people_table(args.participants, "\t")
    _require_unique_people(participants, "participants.tsv")
    participant_ids = set(participants["person_id"])
    required_participant_columns = {"recommended_split", "clinical_site", "study_group", "age"}
    missing = sorted(required_participant_columns - set(participants.columns))
    if missing:
        raise ValueError(f"participants.tsv missing required columns: {missing}")
    participants["recommended_split"] = (
        participants["recommended_split"].astype(str).str.strip().str.lower()
    )
    bad_splits = sorted(set(participants["recommended_split"]) - ALLOWED_SPLITS)
    if bad_splits:
        raise ValueError(f"Unexpected recommended_split values: {bad_splits}")

    labels = _read_people_table(args.labels)
    _require_unique_people(labels, "labels")
    _require_known_people(labels, "labels", participant_ids)
    required_label_columns = {"task", "task_config_hash", "y", "label_value"}
    missing = sorted(required_label_columns - set(labels.columns))
    if missing:
        raise ValueError(f"labels missing required columns: {missing}")
    task_values = labels["task"].fillna("").astype(str).str.strip()
    tasks = sorted(task_values.unique())
    if len(tasks) != 1:
        raise ValueError(f"labels must contain exactly one task, got {tasks}")
    task = tasks[0]
    if not task:
        raise ValueError("labels task cannot be empty")
    labels["task"] = task_values
    labels["y"] = _strict_binary_labels(labels["y"])
    numeric_label_value = pd.to_numeric(labels["label_value"], errors="coerce")
    finite_label_value = (
        numeric_label_value.notna()
        & numeric_label_value.ne(float("inf"))
        & numeric_label_value.ne(float("-inf"))
    )
    if not finite_label_value.all():
        bad_people = labels.loc[~finite_label_value, "person_id"].tolist()
        raise ValueError(
            f"labels label_value must be finite numeric; bad IDs={bad_people[:20]}"
        )
    labels["label_value"] = numeric_label_value
    label_task_hashes = sorted(
        labels["task_config_hash"].fillna("").astype(str).str.strip().unique()
    )
    if len(label_task_hashes) != 1 or not label_task_hashes[0]:
        raise ValueError(
            "labels must contain one non-empty task_config_hash, got "
            f"{label_task_hashes[:5]}"
        )
    label_task_config_hash = label_task_hashes[0]

    clinical = _read_people_table(args.clinical_features)
    _require_unique_people(clinical, "clinical features")
    _require_known_people(clinical, "clinical features", participant_ids)
    clinical_schema = json.loads(Path(args.clinical_schema).read_text(encoding="utf-8"))
    if clinical_schema.get("task") != task:
        raise ValueError(
            f"Clinical schema task={clinical_schema.get('task')!r} does not match labels task={task!r}"
        )
    if str(clinical_schema.get("fit_split", "")).strip().lower() != "train":
        raise ValueError("Clinical schema must declare fit_split=train")
    expected_train_hash = canonical_json_hash({
        "person_ids": sorted(
            participants.loc[
                participants["recommended_split"].eq("train"), "person_id"
            ]
        )
    })
    if clinical_schema.get("fit_person_ids_hash") != expected_train_hash:
        raise ValueError(
            "Clinical schema fit_person_ids_hash does not match current official train IDs"
        )
    preprocessing_config = clinical_schema.get("preprocessing_config")
    if not isinstance(preprocessing_config, Mapping):
        raise ValueError("Clinical schema lacks preprocessing_config")
    if clinical_schema.get("config_hash") != canonical_json_hash(preprocessing_config):
        raise ValueError("Clinical schema config_hash does not bind preprocessing_config")
    if clinical_schema.get("task_config_hash") != label_task_config_hash:
        raise ValueError(
            "Clinical schema and labels were built from different task configs"
        )
    raw_clinical_ids = {
        as_person_id(value)
        for value in clinical_schema.get("raw_clinical_person_ids", [])
    }
    raw_clinical_ids.discard("")
    declared_raw_hash = clinical_schema.get("raw_clinical_person_ids_hash")
    if not declared_raw_hash or declared_raw_hash != canonical_json_hash({
        "person_ids": sorted(raw_clinical_ids)
    }):
        raise ValueError(
            "Clinical schema raw_clinical_person_ids/hash is absent or inconsistent"
        )
    unknown_raw_clinical = sorted(raw_clinical_ids - participant_ids)
    if unknown_raw_clinical:
        raise ValueError(
            "Clinical schema contains raw clinical IDs absent from participants.tsv: "
            f"{unknown_raw_clinical[:20]}"
        )
    expected_features = list(clinical_schema.get("feature_names", []))
    observed_features = [c for c in clinical.columns if c != "person_id"]
    if observed_features != expected_features:
        raise ValueError("Clinical wide columns do not exactly match clinical schema feature_names")

    participant_split = participants.set_index("person_id")["recommended_split"]
    if "recommended_split" in labels:
        supplied_split = labels["recommended_split"].astype(str).str.lower()
        authoritative = labels["person_id"].map(participant_split)
        mismatch = authoritative.notna() & supplied_split.ne(authoritative)
        if mismatch.any():
            raise ValueError(
                "Label file split disagrees with participants.tsv for IDs: "
                f"{labels.loc[mismatch, 'person_id'].tolist()[:20]}"
            )
    label_columns = ["person_id", "task", "task_config_hash", "y"]
    label_columns.extend(
        column for column in labels.columns
        if column.startswith("label_") and column not in label_columns
    )
    availability_columns = [
        column for column in PARTICIPANT_AVAILABILITY if column in participants
    ]
    cohort = labels[label_columns].merge(
        participants[
            ["person_id", "recommended_split", "clinical_site", "study_group", "age"]
            + availability_columns
        ],
        on="person_id",
        how="left",
        validate="one_to_one",
    )
    if cohort["recommended_split"].isna().any():
        missing_ids = cohort.loc[cohort["recommended_split"].isna(), "person_id"].tolist()
        raise ValueError(f"Labeled people absent from participants.tsv: {missing_ids[:20]}")
    cohort = cohort.rename(columns={"recommended_split": "split"})
    for source, destination in PARTICIPANT_AVAILABILITY.items():
        if source in cohort:
            cohort[destination] = _strict_bool(
                cohort.pop(source), f"participants.tsv:{source}"
            )
    retinal_expected_columns = [
        f"expected_retinal_{name}" for name in ("photography", "oct", "octa", "flio")
        if f"expected_retinal_{name}" in cohort
    ]
    if retinal_expected_columns:
        cohort["expected_retinal"] = cohort[retinal_expected_columns].any(axis=1)

    cohort["has_raw_clinical"] = cohort["person_id"].isin(raw_clinical_ids)
    cohort["has_clinical"] = (
        cohort["person_id"].isin(set(clinical["person_id"]))
        & (cohort["has_raw_clinical"] | pd.to_numeric(cohort["age"], errors="coerce").notna())
    )
    cohort["clinical_features_path"] = _portable_path(
        Path(args.clinical_features).resolve(),
        Path(args.artifact_root).resolve() if args.artifact_root else None,
    )

    artifact_root = Path(args.artifact_root).resolve() if args.artifact_root else None
    source_indexes: Dict[str, Any] = {}
    for name, path, candidates in [
        ("environment", args.environment_index, ("artifact_path", "out_path")),
        ("wearable", args.wearable_index, ("out_path", "artifact_path")),
        ("ecg", args.ecg_index, ("artifact_path", "out_path")),
    ]:
        if not path:
            cohort[f"indexed_{name}"] = False
            cohort[f"{name}_artifact_exists"] = False
            cohort[f"has_{name}"] = False
            cohort[f"{name}_path"] = ""
            continue
        view, metadata = _single_artifact_view(
            path,
            name=name,
            path_candidates=candidates,
            artifact_root=artifact_root,
            allow_missing=args.allow_missing_artifacts,
            participant_ids=participant_ids,
        )
        cohort = cohort.merge(view, on="person_id", how="left", validate="one_to_one")
        cohort[f"{name}_path"] = cohort[f"{name}_path"].fillna("")
        cohort[f"{name}_artifact_exists"] = (
            cohort[f"{name}_artifact_exists"].fillna(False).astype(bool)
        )
        cohort[f"indexed_{name}"] = cohort[f"{name}_path"].ne("")
        cohort[f"has_{name}"] = (
            cohort[f"indexed_{name}"] & cohort[f"{name}_artifact_exists"]
        )
        if name == "environment" and "environment_observed_slots" in cohort:
            cohort["has_environment"] &= (
                pd.to_numeric(cohort["environment_observed_slots"], errors="coerce")
                .fillna(0)
                .gt(0)
            )
        if name == "ecg" and "ecg_model_leads" in cohort:
            cohort["has_ecg"] &= (
                pd.to_numeric(cohort["ecg_model_leads"], errors="coerce")
                .fillna(0)
                .gt(0)
            )
        source_indexes[name] = metadata

    if "environment_slot_coverage_fraction" not in cohort:
        cohort["environment_slot_coverage_fraction"] = float("nan")
    cohort["environment_coverage_qc_tier"] = [
        _environment_coverage_qc_tier(indexed, exists, fraction)
        for indexed, exists, fraction in zip(
            cohort["indexed_environment"],
            cohort["environment_artifact_exists"],
            cohort["environment_slot_coverage_fraction"],
        )
    ]

    if args.retinal_index:
        retinal, metadata = _retinal_view(
            args.retinal_index,
            artifact_root=artifact_root,
            allow_missing=args.allow_missing_artifacts,
            participant_ids=participant_ids,
        )
        cohort = cohort.merge(retinal, on="person_id", how="left", validate="one_to_one")
        cohort["retinal_indexed_asset_count"] = (
            cohort["retinal_indexed_asset_count"].fillna(0).astype(int)
        )
        cohort["retinal_asset_count"] = cohort["retinal_asset_count"].fillna(0).astype(int)
        cohort["retinal_asset_paths_json"] = cohort["retinal_asset_paths_json"].fillna("[]")
        cohort["has_retinal"] = cohort["retinal_asset_count"].gt(0)
        cohort["indexed_retinal"] = cohort["retinal_indexed_asset_count"].gt(0)
        for modality in ("photography", "oct", "octa", "flio"):
            column = f"retinal_{modality}_asset_count"
            if column not in cohort:
                cohort[column] = 0
            cohort[column] = cohort[column].fillna(0).astype(int)
            cohort[f"has_retinal_{modality}"] = cohort[column].gt(0)
        source_indexes["retinal"] = metadata
    else:
        cohort["retinal_asset_paths_json"] = "[]"
        cohort["retinal_asset_count"] = 0
        cohort["retinal_indexed_asset_count"] = 0
        cohort["indexed_retinal"] = False
        cohort["has_retinal"] = False
        for modality in ("photography", "oct", "octa", "flio"):
            cohort[f"retinal_{modality}_asset_count"] = 0
            cohort[f"has_retinal_{modality}"] = False

    if "has_wearable" in cohort:
        cohort["has_cgm"] = cohort["has_wearable"] & cohort.get(
            "cgm_observed_slots", pd.Series(0, index=cohort.index)
        ).fillna(0).gt(0)
        cohort["has_activity"] = cohort["has_wearable"] & cohort.get(
            "activity_observed_slots", pd.Series(0, index=cohort.index)
        ).fillna(0).gt(0)
    else:
        cohort["has_cgm"] = False
        cohort["has_activity"] = False

    modality_columns = [
        "has_clinical", "has_retinal", "has_environment", "has_cgm", "has_activity", "has_ecg"
    ]
    for column in modality_columns:
        if column not in cohort:
            cohort[column] = False
        cohort[column] = cohort[column].fillna(False).astype(bool)
    cohort["available_modality_count"] = cohort[modality_columns].sum(axis=1).astype(int)

    preferred = [
        "person_id", "task", "split", "y", "label_value", "label_datetime",
        "clinical_site", "study_group", "age", "has_raw_clinical", *modality_columns,
        "available_modality_count", "clinical_features_path", "retinal_asset_paths_json",
        "environment_path", "environment_observed_slots",
        "environment_slot_coverage_fraction", "environment_coverage_qc_tier",
        "wearable_path", "ecg_path",
    ]
    ordered = [c for c in preferred if c in cohort]
    cohort = cohort[ordered + [c for c in cohort if c not in ordered]]
    cohort = cohort.sort_values("person_id", kind="mergesort")

    out_csv = Path(args.out_csv).resolve()
    _atomic_text(out_csv, cohort.to_csv(index=False))
    metadata_path = Path(args.metadata_json).resolve() if args.metadata_json else out_csv.with_suffix(".metadata.json")
    coverage = {
        split: {
            column: int(group[column].sum()) for column in modality_columns
        }
        for split, group in cohort.groupby("split")
    }
    environment_coverage_tier_by_split = {
        split: {
            tier: int(group["environment_coverage_qc_tier"].eq(tier).sum())
            for tier in ENVIRONMENT_COVERAGE_QC_TIERS
        }
        for split, group in cohort.groupby("split")
    }
    core_source_files = source_file_metadata(
        [args.participants, args.labels, args.clinical_features, args.clinical_schema]
    )
    participant_binding_columns = [
        "person_id", "recommended_split", "clinical_site", "study_group", "age",
        *availability_columns,
    ]
    label_binding_columns = [
        column for column in labels.columns
        if column not in {"recommended_split", "clinical_site", "study_group"}
    ]
    input_bindings = {
        "participant_assignment_hash": _frame_content_hash(
            participants,
            participant_binding_columns,
            sort_columns=["person_id"],
        ),
        "label_assignment_hash": _frame_content_hash(
            labels,
            label_binding_columns,
            sort_columns=["person_id"],
        ),
        "cohort_content_hash": _frame_content_hash(
            cohort, list(cohort.columns), sort_columns=["person_id"]
        ),
        "core_source_metadata_hash": canonical_json_hash(
            {"source_files": core_source_files}
        ),
        "source_index_mapping_hashes": {
            name: value.get("mapping_hash") for name, value in source_indexes.items()
        },
    }
    master_config = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "builder": "build_multimodal_index.py",
        "task": task,
        "task_config_hash": label_task_config_hash,
        "clinical_schema_hash": canonical_json_hash(clinical_schema),
        "source_index_config_hashes": {
            name: value.get("config_hash") for name, value in source_indexes.items()
        },
        "path_semantics": (
            f"relative_to:{artifact_root}" if artifact_root else "resolved_absolute"
        ),
        "allow_missing_artifacts": bool(args.allow_missing_artifacts),
        "environment_coverage_qc": {
            "audit_only": True,
            "changes_has_environment": False,
            "fraction_boundaries": list(ENVIRONMENT_COVERAGE_QC_BOUNDARIES),
            "tiers": list(ENVIRONMENT_COVERAGE_QC_TIERS),
        },
        "input_bindings": input_bindings,
    }
    metadata = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "config_hash": canonical_json_hash(master_config),
        "preprocessing_config": master_config,
        "task": task,
        "n_rows": int(len(cohort)),
        "label_positive_count": int(cohort["y"].sum()),
        "split_counts": cohort["split"].value_counts().to_dict(),
        "coverage_by_split": coverage,
        "environment_coverage_tier_by_split": environment_coverage_tier_by_split,
        "participant_flag_vs_artifact_mismatches": {
            modality: {
                "expected_but_not_available": int(
                    (cohort[f"expected_{modality}"] & ~cohort[f"has_{modality}"]).sum()
                ),
                "available_but_not_expected": int(
                    (~cohort[f"expected_{modality}"] & cohort[f"has_{modality}"]).sum()
                ),
            }
            for modality in ("clinical", "retinal", "environment", "cgm", "activity", "ecg")
            if f"expected_{modality}" in cohort
        },
        "fit_split_authority": "participants.tsv:recommended_split",
        "clinical_schema_hash": canonical_json_hash(clinical_schema),
        "source_indexes": source_indexes,
        "source_files": core_source_files,
        "input_bindings": input_bindings,
        "path_semantics": (
            f"relative_to:{artifact_root}" if artifact_root else "resolved_absolute"
        ),
        "notes": [
            "This is the first stage that joins labels to task-independent modality artifacts.",
            "Missing modalities remain explicit booleans and are not used to drop participants.",
            "indexed_* records manifest membership; has_* additionally requires an existing artifact and valid modality support.",
            "Environment coverage fraction/tier are audit-only and do not change has_environment, cohort membership, scaler fitting, or model inputs.",
            "Retinal protocols remain a JSON list of individual assets rather than fixed semantic slots.",
        ],
    }
    _atomic_text(metadata_path, json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps({
        "out_csv": str(out_csv), "metadata_json": str(metadata_path),
        "task": task, "n_rows": len(cohort), "coverage": coverage,
    }, indent=2))


if __name__ == "__main__":
    main()
