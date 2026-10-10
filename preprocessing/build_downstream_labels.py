#!/usr/bin/env python3
"""Build task labels independently of task-agnostic preprocessing artifacts.

This module is the only place where MoCA/CVD label definitions live.  Raw
clinical preprocessing must not depend on the selected downstream task.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd

try:
    from .preprocessing_utils import (
        PREPROCESSING_VERSION, as_person_id, canonical_json_hash, source_file_metadata,
    )
except ImportError:
    from preprocessing_utils import (  # type: ignore
        PREPROCESSING_VERSION, as_person_id, canonical_json_hash, source_file_metadata,
    )


def source_code(value: Any) -> str:
    """Return the exact AI-READI variable code before the first comma."""
    if value is None or pd.isna(value):
        return ""
    return str(value).split(",", 1)[0].strip().lower()


def read_participants(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype={"person_id": "string"})
    required = {"person_id", "recommended_split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"participants.tsv missing required columns: {missing}")
    df = df.copy()
    df["person_id"] = df["person_id"].map(as_person_id)
    if (df["person_id"] == "").any():
        raise ValueError("participants.tsv contains an empty person_id")
    if df["person_id"].duplicated().any():
        duplicate_ids = df.loc[
            df["person_id"].duplicated(keep=False), "person_id"
        ].unique().tolist()
        raise ValueError(f"participants.tsv contains duplicate person IDs: {duplicate_ids[:20]}")
    split = df["recommended_split"].astype(str).str.strip().str.lower()
    bad_splits = sorted(set(split) - {"train", "val", "test"})
    if bad_splits:
        raise ValueError(f"participants.tsv has unexpected recommended_split: {bad_splits}")
    df["recommended_split"] = split
    return df


def validate_source_people(
    frame: pd.DataFrame, source: str, participant_ids: set[str]
) -> pd.DataFrame:
    """Normalize and validate every source row before task filtering."""
    if "person_id" not in frame:
        raise ValueError(f"{source} is missing person_id")
    out = frame.copy()
    out["person_id"] = out["person_id"].map(as_person_id)
    if (out["person_id"] == "").any():
        raise ValueError(f"{source} contains an empty person_id")
    unknown = sorted(set(out["person_id"]) - participant_ids)
    if unknown:
        raise ValueError(
            f"{source} contains people absent from participants.tsv: {unknown[:20]}"
        )
    return out


def _joined_codes(values: Iterable[str]) -> str:
    return ";".join(sorted({str(value) for value in values if str(value)}))


def build_moca_labels(measurement: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    concept_id = int(spec["concept_id"])
    threshold = float(spec.get("threshold", 26.0))
    work = measurement.copy()
    work["person_id"] = work["person_id"].map(as_person_id)
    concepts = pd.to_numeric(work["measurement_concept_id"], errors="coerce")
    work = work.loc[concepts == concept_id].copy()
    numeric = pd.to_numeric(work.get(spec.get("value_column", "value_as_number")), errors="coerce")
    if "value_source_value" in work.columns:
        numeric = numeric.combine_first(pd.to_numeric(work["value_source_value"], errors="coerce"))
    work["label_value"] = numeric
    low, high = map(float, spec.get("valid_range", [0, 30]))
    work = work[work["label_value"].between(low, high, inclusive="both")].copy()
    if work.empty:
        return pd.DataFrame(columns=[
            "person_id", "label_value", "y", "label_datetime",
            "label_record_count", "label_unique_value_count", "label_unique_class_count",
            "label_conflict", "label_class_conflict",
        ])

    work["_candidate_y"] = work["label_value"].lt(threshold).astype(int)
    record_audit = work.groupby("person_id").agg(
        label_record_count=("label_value", "size"),
        label_unique_value_count=("label_value", "nunique"),
        label_unique_class_count=("_candidate_y", "nunique"),
    ).reset_index()
    record_audit["label_conflict"] = record_audit["label_unique_value_count"].gt(1)
    record_audit["label_class_conflict"] = record_audit["label_unique_class_count"].gt(1)

    date_col = next(
        (c for c in ["measurement_datetime", "measurement_date"] if c in work.columns),
        None,
    )
    if date_col:
        work["_date"] = pd.to_datetime(work[date_col], errors="coerce")
        work["_has_date"] = work["_date"].notna()
        tie_breakers = [c for c in ["person_id", "_has_date", "_date", "measurement_id"] if c in work]
        work = work.sort_values(tie_breakers, kind="mergesort").groupby("person_id", as_index=False).tail(1)
        label_datetime = work["_date"].dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("")
    else:
        work = work.sort_values("person_id", kind="mergesort").groupby("person_id", as_index=False).tail(1)
        label_datetime = pd.Series("", index=work.index)

    out = work[["person_id", "label_value"]].copy()
    out["y"] = (out["label_value"] < threshold).astype(int)
    out["label_datetime"] = label_datetime.to_numpy()
    return out.merge(record_audit, on="person_id", how="left", validate="one_to_one").reset_index(drop=True)


def build_cvd_labels(
    condition: pd.DataFrame,
    observation: pd.DataFrame,
    participants: pd.DataFrame,
    spec: Dict[str, Any],
) -> pd.DataFrame:
    source_col = spec.get("source_column", "condition_source_value")
    if source_col not in condition.columns:
        raise ValueError(f"condition_occurrence missing {source_col!r}")
    positive_codes = {str(x).lower() for x in spec["positive_source_codes"]}
    response_spec = spec.get("response_evidence", {})
    if response_spec.get("table") != "observation":
        raise ValueError("CVD label response_evidence.table must be observation")
    response_source_col = response_spec.get(
        "source_column", "observation_source_value"
    )
    response_value_col = response_spec.get(
        "value_concept_column", "value_as_concept_id"
    )
    missing_response_columns = sorted(
        {response_source_col, response_value_col} - set(observation.columns)
    )
    if missing_response_columns:
        raise ValueError(
            "observation.csv lacks CVD response-evidence columns: "
            f"{missing_response_columns}"
        )
    yes_concept = int(response_spec["yes_concept_id"])
    no_concept = int(response_spec["no_concept_id"])
    unknown_concepts = {
        int(value) for value in response_spec.get("unknown_concept_ids", [])
    }

    work = condition[["person_id", source_col]].copy()
    work["person_id"] = work["person_id"].map(as_person_id)
    work["source_code"] = work[source_col].map(source_code)
    work["_positive"] = work["source_code"].isin(positive_codes)
    positive_rows = work.loc[work["_positive"]].copy()
    condition_code_sets = (
        positive_rows.groupby("person_id")["source_code"]
        .agg(lambda values: frozenset(values))
        .to_dict()
    )
    condition_record_counts = positive_rows.groupby("person_id").size().to_dict()

    responses = observation[["person_id", response_source_col, response_value_col]].copy()
    responses["person_id"] = responses["person_id"].map(as_person_id)
    responses["source_code"] = responses[response_source_col].map(source_code)
    responses = responses[responses["source_code"].isin(positive_codes)].copy()
    responses["response_concept_id"] = (
        pd.to_numeric(responses[response_value_col], errors="coerce")
        .fillna(0)
        .astype("int64")
    )
    response_sets = (
        responses.groupby(["person_id", "source_code"])["response_concept_id"]
        .agg(lambda values: frozenset(int(value) for value in values))
        .to_dict()
    )

    rows = []
    for person_id in participants["person_id"].drop_duplicates():
        condition_yes = set(condition_code_sets.get(person_id, frozenset()))
        response_yes: set[str] = set()
        response_no: set[str] = set()
        response_unknown: set[str] = set()
        response_conflict: set[str] = set()
        missing_codes: set[str] = set()
        for code in positive_codes:
            concepts = set(response_sets.get((person_id, code), frozenset()))
            if not concepts:
                missing_codes.add(code)
            elif concepts == {yes_concept}:
                response_yes.add(code)
            elif concepts == {no_concept}:
                response_no.add(code)
            elif concepts.issubset(unknown_concepts):
                response_unknown.add(code)
            else:
                response_conflict.add(code)

        source_mismatch = condition_yes.symmetric_difference(response_yes)
        if response_conflict:
            y: Any = pd.NA
            status = "excluded_response_conflict"
        elif source_mismatch:
            y = pd.NA
            status = "excluded_condition_observation_mismatch"
        elif condition_yes:
            # The task is an OR across conditions: one verified yes is enough
            # even if another question was unknown.
            y = 1
            status = "positive_verified"
        elif response_no == positive_codes:
            y = 0
            status = "negative_all_questions_explicit_no"
        else:
            y = pd.NA
            status = "excluded_unknown_or_incomplete_negative"

        rows.append({
            "person_id": person_id,
            "y": y,
            "label_value": pd.NA if pd.isna(y) else float(y),
            "label_datetime": "",
            "label_status": status,
            "label_record_count": int(condition_record_counts.get(person_id, 0)),
            "label_evidence_codes": _joined_codes(condition_yes),
            "label_response_yes_codes": _joined_codes(response_yes),
            "label_response_no_codes": _joined_codes(response_no),
            "label_response_unknown_codes": _joined_codes(response_unknown),
            "label_response_missing_codes": _joined_codes(missing_codes),
            "label_response_conflict_codes": _joined_codes(response_conflict),
            "label_source_mismatch_codes": _joined_codes(source_mismatch),
            "label_question_count_expected": int(len(positive_codes)),
            "label_question_count_observed": int(
                len(positive_codes - missing_codes)
            ),
        })
    out = pd.DataFrame(rows)
    out["y"] = out["y"].astype("Int64")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", required=True)
    ap.add_argument("--clinical_dir", required=True)
    ap.add_argument("--task_config", default=str(Path(__file__).parent / "configs" / "downstream_tasks.json"))
    ap.add_argument("--tasks", nargs="+", default=["moca_binary", "cvd_broad_binary", "cvd_strict_binary"])
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    participants = read_participants(args.participants)
    config = json.loads(Path(args.task_config).read_text(encoding="utf-8"))
    clinical_dir = Path(args.clinical_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    measurement = None
    condition = None
    observation = None
    task_config_hash = canonical_json_hash(config)
    participant_ids = set(participants["person_id"])
    summaries = {}
    for task in args.tasks:
        if task not in config:
            raise KeyError(f"Unknown task {task!r}; available={sorted(config)}")
        task_spec = config[task]
        label_spec = task_spec["label"]
        table = label_spec["table"]
        if table == "measurement":
            if measurement is None:
                measurement = pd.read_csv(
                    clinical_dir / "measurement.csv", low_memory=False,
                    dtype={"person_id": "string"},
                )
                measurement = validate_source_people(
                    measurement, "measurement.csv", participant_ids
                )
            candidates = build_moca_labels(measurement, label_spec)
            candidates["label_status"] = "selected_latest_valid"
            conflict = candidates["label_class_conflict"].fillna(False)
            candidates.loc[conflict, "label_status"] = (
                "excluded_conflicting_binary_label"
            )
            excluded = candidates.loc[conflict].copy()
            excluded.to_csv(
                out_dir / f"label_exclusions_{task}.csv", index=False
            )
            labels = candidates.loc[~conflict].copy()
        elif table == "condition_occurrence":
            if condition is None:
                condition = pd.read_csv(
                    clinical_dir / "condition_occurrence.csv", low_memory=False,
                    dtype={"person_id": "string"},
                )
                condition = validate_source_people(
                    condition, "condition_occurrence.csv", participant_ids
                )
            if observation is None:
                observation = pd.read_csv(
                    clinical_dir / "observation.csv", low_memory=False,
                    dtype={"person_id": "string"},
                )
                observation = validate_source_people(
                    observation, "observation.csv", participant_ids
                )
            candidates = build_cvd_labels(
                condition, observation, participants, label_spec
            )
            excluded = candidates[candidates["y"].isna()].copy()
            excluded.to_csv(
                out_dir / f"label_exclusions_{task}.csv", index=False
            )
            labels = candidates[candidates["y"].notna()].copy()
            labels["y"] = labels["y"].astype(int)
        else:
            raise ValueError(f"Unsupported label table {table!r}")

        unknown_people = sorted(set(labels["person_id"]) - set(participants["person_id"]))
        if unknown_people:
            raise ValueError(
                f"{task} label rows contain people absent from participants.tsv: "
                f"{unknown_people[:20]}"
            )
        if labels["person_id"].duplicated().any():
            raise ValueError(f"{task} label construction produced duplicate person IDs")

        labels = labels.merge(
            participants[[c for c in ["person_id", "recommended_split", "clinical_site", "study_group"] if c in participants]],
            on="person_id",
            how="left",
            validate="one_to_one",
        )
        labels.insert(1, "task", task)
        labels.insert(2, "task_config_hash", task_config_hash)
        labels.to_csv(out_dir / f"labels_{task}.csv", index=False)
        summaries[task] = {
            "n": int(len(labels)),
            "n_positive": int(labels["y"].sum()),
            "positive_rate": float(labels["y"].mean()) if len(labels) else None,
            "by_split": labels.groupby("recommended_split")["y"].agg(["count", "sum"]).to_dict("index"),
        }
        if table == "condition_occurrence":
            summaries[task]["n_label_candidates"] = int(len(candidates))
            summaries[task]["n_excluded_unknown_or_inconsistent"] = int(
                len(excluded)
            )
            summaries[task]["candidate_status_counts"] = {
                str(key): int(value)
                for key, value in candidates["label_status"].value_counts().items()
            }
        if "label_conflict" in candidates:
            summaries[task]["n_people_with_conflicting_valid_label_values"] = int(
                candidates["label_conflict"].fillna(False).sum()
            )
            summaries[task]["n_people_with_conflicting_binary_labels"] = int(
                candidates["label_class_conflict"].fillna(False).sum()
            )
            summaries[task]["n_excluded_conflicting_binary_labels"] = int(
                len(excluded)
            )

    metadata = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "task_config": str(Path(args.task_config)),
        "task_config_hash": task_config_hash,
        "task_specs": {task: config[task]["label"] for task in args.tasks},
        "source_files": source_file_metadata([
            args.participants,
            args.task_config,
            clinical_dir / "measurement.csv" if measurement is not None else None,
            clinical_dir / "condition_occurrence.csv" if condition is not None else None,
            clinical_dir / "observation.csv" if observation is not None else None,
        ]),
        "summaries": summaries,
        "notes": [
            "Labels are generated outside task-agnostic modality preprocessing.",
            "CVD matching uses the exact AI-READI source code before the first comma.",
            "A CVD negative requires an explicit no response for every task question; unknown/incomplete negatives are excluded.",
            "Condition-occurrence positives are cross-checked against observation yes responses.",
            "Task-specific clinical feature exclusions are defined in the same task config.",
        ],
    }
    (out_dir / "label_build_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
