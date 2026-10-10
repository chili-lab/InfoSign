#!/usr/bin/env python3
"""Split-safe clinical tabular views for AI-READI downstream tasks.

Raw OMOP tables remain the task-independent source of truth.  This builder
creates a task-aware *view* whose feature vocabulary is fitted exclusively on
the official training split.  It writes raw-scale values with NaNs plus a
feature catalog; imputation/scaling remains a downstream train-only operation.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .preprocessing_utils import (
        PREPROCESSING_VERSION, as_person_id, canonical_json_hash, source_file_metadata,
    )
except ImportError:
    from preprocessing_utils import (  # type: ignore
        PREPROCESSING_VERSION, as_person_id, canonical_json_hash, source_file_metadata,
    )


# Cohort-verified task-independent clinical data-quality rules.  These are
# embedded in the hashed schema below so changing one invalidates downstream
# artifacts instead of silently changing a feature definition.
IDENTIFIER_SOURCE_CODES = frozenset({"studyid"})
IDENTIFIER_CONCEPT_IDS = frozenset({42528934})
OBSERVATION_NUMERIC_SENTINELS = frozenset({777.0, 888.0, 999.0})
OBSERVATION_NUMERIC_SENTINEL_SOURCE_CODES = frozenset(
    {"sumrjfr", "pxne9", "pdrx9"}
)


def source_code(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).split(",", 1)[0].strip().lower()


def safe_name(value: Any) -> str:
    text = source_code(value)
    text = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return text or "unknown"


def normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    if "person_id" not in df.columns:
        raise ValueError("Clinical table is missing person_id")
    out = df.copy()
    out["person_id"] = out["person_id"].map(as_person_id)
    return out


def validate_clinical_people(
    df: pd.DataFrame, source: str, participant_ids: set[str]
) -> None:
    if "person_id" not in df:
        raise ValueError(f"{source} is missing person_id")
    normalized = df["person_id"].map(as_person_id)
    if normalized.eq("").any():
        raise ValueError(f"{source} contains an empty person_id")
    observed = set(normalized)
    unknown = sorted(observed - participant_ids)
    if unknown:
        raise ValueError(
            f"{source} contains people absent from participants.tsv: {unknown[:20]}"
        )


def load_task_spec(path: str | Path, task: str) -> Tuple[dict, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if task not in payload:
        raise KeyError(f"Unknown task={task!r}; available={sorted(payload)}")
    return payload[task], canonical_json_hash(payload)


def exclusion_mask(
    df: pd.DataFrame,
    *,
    table: str,
    concept_col: str | None,
    source_col: str,
    exclusions: Mapping[str, Any],
) -> Tuple[pd.Series, List[dict]]:
    mask = pd.Series(False, index=df.index)
    audit: List[dict] = []

    concept_key = {
        "measurement": "measurement_concept_ids",
        "observation": "observation_concept_ids",
    }.get(table)
    if concept_key and concept_col and concept_col in df.columns:
        ids = {int(x) for x in exclusions.get(concept_key, [])}
        if ids:
            hit = pd.to_numeric(df[concept_col], errors="coerce").isin(ids)
            mask |= hit
            audit.append({"table": table, "rule": concept_key, "rows": int(hit.sum())})

    # Medical-history answers occur in both observation.csv and
    # condition_occurrence.csv.  Apply exact source-code exclusions to every
    # table so a label cannot survive through a duplicated OMOP representation.
    codes = {str(x).lower() for x in exclusions.get("condition_source_codes", [])}
    if codes:
        hit = df[source_col].map(source_code).isin(codes)
        mask |= hit
        audit.append({"table": table, "rule": "condition_source_codes", "rows": int(hit.sum())})

    patterns = [safe_name(x) for x in exclusions.get("source_key_patterns", [])]
    if patterns:
        src = df[source_col].map(safe_name)
        hit = pd.Series(False, index=df.index)
        for pattern in patterns:
            hit |= src.str.contains(pattern, regex=False, na=False)
        mask |= hit
        audit.append({"table": table, "rule": "source_key_patterns", "rows": int(hit.sum())})

    return mask, audit


def identifier_mask(
    df: pd.DataFrame,
    *,
    table: str,
    concept_col: str | None,
    source_col: str,
) -> Tuple[pd.Series, List[dict]]:
    """Remove participant identifiers before any feature vocabulary is fit."""
    source_hit = df[source_col].map(source_code).isin(IDENTIFIER_SOURCE_CODES)
    concept_hit = pd.Series(False, index=df.index)
    if concept_col and concept_col in df:
        concept_hit = pd.to_numeric(df[concept_col], errors="coerce").isin(
            IDENTIFIER_CONCEPT_IDS
        )
    hit = source_hit | concept_hit
    return hit, [{
        "table": table,
        "rule": "participant_identifier",
        "rows": int(hit.sum()),
    }]


def select_train_features(
    long_df: pd.DataFrame,
    train_ids: set[str],
    min_train_persons: int,
) -> List[str]:
    train = long_df[long_df["person_id"].isin(train_ids)]
    counts = train.groupby("feature_name")["person_id"].nunique()
    return sorted(counts[counts >= min_train_persons].index.tolist())


def pivot_long(
    long_df: pd.DataFrame,
    all_ids: Sequence[str],
    train_ids: set[str],
    min_train_persons: int,
    aggregation: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    if long_df.empty:
        return pd.DataFrame(index=pd.Index(all_ids, name="person_id")), pd.DataFrame(), []
    selected = select_train_features(long_df, train_ids, min_train_persons)
    work = long_df[long_df["feature_name"].isin(selected)].copy()
    if aggregation == "mean":
        wide = work.pivot_table(index="person_id", columns="feature_name", values="value", aggfunc="mean")
    elif aggregation == "sum":
        wide = work.pivot_table(index="person_id", columns="feature_name", values="value", aggfunc="sum")
    elif aggregation == "max":
        wide = work.pivot_table(index="person_id", columns="feature_name", values="value", aggfunc="max")
    else:
        raise ValueError(f"Unsupported aggregation={aggregation}")
    wide = wide.reindex(index=all_ids, columns=selected)

    train_before_fill = wide.loc[[pid for pid in all_ids if pid in train_ids]]
    nonmissing = train_before_fill.notna().sum(axis=0)
    # For event/presence tables, no row means the event was not recorded and is
    # represented as zero.  Numeric measurements retain NaN as true missingness.
    if aggregation in {"max", "sum"}:
        wide = wide.fillna(0.0)
    train = wide.loc[[pid for pid in all_ids if pid in train_ids]]
    nunique = train.nunique(axis=0, dropna=True)
    constants = sorted(nunique[nunique <= 1].index.tolist())
    kept = [x for x in selected if x not in set(constants)]
    wide = wide[kept]

    meta_cols = [
        "feature_name", "table", "source_key", "concept_id", "unit",
        "unit_concept_id", "response_concept_id",
        "value_kind", "aggregation",
    ]
    catalog = (
        work[meta_cols]
        .drop_duplicates("feature_name")
        .set_index("feature_name")
        .reindex(kept)
        .reset_index()
    )
    catalog["train_nonmissing_persons"] = catalog["feature_name"].map(nonmissing).fillna(0).astype(int)
    catalog["train_prevalence"] = catalog["train_nonmissing_persons"] / max(len(train_ids), 1)
    return wide, catalog, constants


def numeric_long(
    df: pd.DataFrame,
    *,
    table: str,
    source_col: str,
    concept_col: str,
    value_col: str,
    unit_col: str | None,
    unit_concept_col: str | None,
    exclusions: Mapping[str, Any],
) -> Tuple[pd.DataFrame, List[dict]]:
    work = normalize_ids(df)
    quality_drop, quality_audit = identifier_mask(
        work,
        table=table,
        concept_col=concept_col,
        source_col=source_col,
    )
    drop, audit = exclusion_mask(
        work,
        table=table,
        concept_col=concept_col,
        source_col=source_col,
        exclusions=exclusions,
    )
    audit = quality_audit + audit
    drop |= quality_drop
    if table == "observation":
        # OMOP concept-valued observations are categorical responses.  Keeping
        # their numeric encodings as continuous values duplicates the one-hot
        # representation below and turns unknown/refused codes into magnitudes.
        response_concept = pd.to_numeric(
            work.get("value_as_concept_id"), errors="coerce"
        ).fillna(0)
        categorical_numeric = response_concept.ne(0)
        numeric_value = pd.to_numeric(work.get(value_col), errors="coerce")
        sentinel = (
            numeric_value.isin(OBSERVATION_NUMERIC_SENTINELS)
            & work[source_col].map(source_code).isin(
                OBSERVATION_NUMERIC_SENTINEL_SOURCE_CODES
            )
        )
        drop |= categorical_numeric | sentinel
        audit.extend([
            {
                "table": table,
                "rule": "concept_valued_observation_not_continuous",
                "rows": int(categorical_numeric.sum()),
            },
            {
                "table": table,
                "rule": "observation_numeric_missing_sentinel",
                "rows": int(sentinel.sum()),
            },
        ])
    work = work.loc[~drop].copy()
    work["value"] = pd.to_numeric(work.get(value_col), errors="coerce")
    work = work[work["value"].notna()].copy()
    work["source_key"] = work[source_col].map(source_code)
    work["concept_id"] = pd.to_numeric(work.get(concept_col), errors="coerce").fillna(0).astype("int64")
    work["unit"] = work[unit_col].fillna("").astype(str).str.strip() if unit_col and unit_col in work else ""
    if unit_concept_col and unit_concept_col in work:
        work["unit_concept_id"] = (
            pd.to_numeric(work[unit_concept_col], errors="coerce")
            .fillna(0)
            .astype("int64")
        )
    else:
        work["unit_concept_id"] = 0
    # Prefer the standardized OMOP unit concept.  When it is unavailable,
    # retain a normalized source-unit fallback so two raw units are never
    # silently averaged merely because both have concept ID zero.
    work["unit_key"] = work["unit_concept_id"].astype(str)
    missing_unit_concept = work["unit_concept_id"].eq(0)
    work.loc[missing_unit_concept, "unit_key"] = (
        "0s" + work.loc[missing_unit_concept, "unit"].map(safe_name)
    )
    work["response_concept_id"] = 0
    work["feature_name"] = (
        table[:4]
        + "__"
        + work["source_key"].map(safe_name)
        + "__c"
        + work["concept_id"].astype(str)
        + "__u"
        + work["unit_key"]
    )
    work["table"] = table
    work["value_kind"] = "numeric"
    work["aggregation"] = "mean"
    return work[["person_id", "feature_name", "value", "table", "source_key", "concept_id", "unit", "unit_concept_id", "response_concept_id", "value_kind", "aggregation"]], audit


def observation_concept_long(
    observation: pd.DataFrame,
    exclusions: Mapping[str, Any],
) -> Tuple[pd.DataFrame, List[dict]]:
    work = normalize_ids(observation)
    quality_drop, quality_audit = identifier_mask(
        work,
        table="observation",
        concept_col="observation_concept_id",
        source_col="observation_source_value",
    )
    drop, audit = exclusion_mask(
        work,
        table="observation",
        concept_col="observation_concept_id",
        source_col="observation_source_value",
        exclusions=exclusions,
    )
    drop |= quality_drop
    audit = quality_audit + audit
    work = work.loc[~drop].copy()
    response = pd.to_numeric(work.get("value_as_concept_id"), errors="coerce").fillna(0).astype("int64")
    work = work[response > 0].copy()
    response = response.loc[work.index]
    work["source_key"] = work["observation_source_value"].map(source_code)
    work["concept_id"] = pd.to_numeric(work["observation_concept_id"], errors="coerce").fillna(0).astype("int64")
    work["unit"] = ""
    work["unit_concept_id"] = 0
    work["response_concept_id"] = response
    work["feature_name"] = (
        "obsc__"
        + work["source_key"].map(safe_name)
        + "__response_c"
        + response.astype(str)
    )
    work["value"] = 1.0
    work["table"] = "observation"
    work["value_kind"] = "categorical_concept_presence"
    work["aggregation"] = "max"
    return work[["person_id", "feature_name", "value", "table", "source_key", "concept_id", "unit", "unit_concept_id", "response_concept_id", "value_kind", "aggregation"]], audit


def condition_long(condition: pd.DataFrame, exclusions: Mapping[str, Any]) -> Tuple[pd.DataFrame, List[dict]]:
    work = normalize_ids(condition)
    quality_drop, quality_audit = identifier_mask(
        work,
        table="condition",
        concept_col="condition_concept_id",
        source_col="condition_source_value",
    )
    drop, audit = exclusion_mask(
        work,
        table="condition",
        concept_col="condition_concept_id",
        source_col="condition_source_value",
        exclusions=exclusions,
    )
    drop |= quality_drop
    audit = quality_audit + audit
    work = work.loc[~drop].copy()
    work["source_key"] = work["condition_source_value"].map(source_code)
    work["concept_id"] = pd.to_numeric(work["condition_concept_id"], errors="coerce").fillna(0).astype("int64")
    work["unit"] = ""
    work["unit_concept_id"] = 0
    work["response_concept_id"] = 0
    work["feature_name"] = "cond__" + work["source_key"].map(safe_name) + "__c" + work["concept_id"].astype(str)
    work["value"] = 1.0
    work["table"] = "condition"
    work["value_kind"] = "presence"
    work["aggregation"] = "max"
    return work[["person_id", "feature_name", "value", "table", "source_key", "concept_id", "unit", "unit_concept_id", "response_concept_id", "value_kind", "aggregation"]], audit


def procedure_long(
    procedure: pd.DataFrame, exclusions: Mapping[str, Any]
) -> Tuple[pd.DataFrame, List[dict]]:
    work = normalize_ids(procedure)
    quality_drop, quality_audit = identifier_mask(
        work,
        table="procedure",
        concept_col="procedure_concept_id",
        source_col="procedure_source_value",
    )
    drop, audit = exclusion_mask(
        work,
        table="procedure",
        concept_col="procedure_concept_id",
        source_col="procedure_source_value",
        exclusions=exclusions,
    )
    drop |= quality_drop
    audit = quality_audit + audit
    work = work.loc[~drop].copy()
    work["source_key"] = work["procedure_source_value"].map(source_code)
    work["concept_id"] = pd.to_numeric(work["procedure_concept_id"], errors="coerce").fillna(0).astype("int64")
    work["unit"] = ""
    work["unit_concept_id"] = 0
    work["response_concept_id"] = 0
    work["feature_name"] = "proc__" + work["source_key"].map(safe_name) + "__c" + work["concept_id"].astype(str)
    work["value"] = pd.to_numeric(work.get("quantity"), errors="coerce").fillna(1.0)
    work["table"] = "procedure"
    work["value_kind"] = "count"
    work["aggregation"] = "sum"
    return work[["person_id", "feature_name", "value", "table", "source_key", "concept_id", "unit", "unit_concept_id", "response_concept_id", "value_kind", "aggregation"]], audit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical_dir", required=True)
    ap.add_argument("--participants", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--task_config", default=str(Path(__file__).parent / "configs" / "downstream_tasks.json"))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_train_persons", type=int, default=20)
    ap.add_argument("--include_condition", action="store_true")
    ap.add_argument("--include_procedure", action="store_true")
    args = ap.parse_args()

    participants = pd.read_csv(
        args.participants, sep="\t", dtype={"person_id": "string"}
    )
    required_participant_columns = {"person_id", "recommended_split", "age"}
    missing_participant_columns = sorted(
        required_participant_columns - set(participants.columns)
    )
    if missing_participant_columns:
        raise ValueError(
            f"participants.tsv missing required columns: {missing_participant_columns}"
        )
    participants["person_id"] = participants["person_id"].map(as_person_id)
    if (participants["person_id"] == "").any() or participants["person_id"].duplicated().any():
        raise ValueError("participants.tsv must contain one non-empty row per person_id")
    participants["recommended_split"] = (
        participants["recommended_split"].astype(str).str.strip().str.lower()
    )
    bad_splits = sorted(
        set(participants["recommended_split"]) - {"train", "val", "test"}
    )
    if bad_splits:
        raise ValueError(f"Unexpected recommended_split values: {bad_splits}")
    all_ids = participants["person_id"].drop_duplicates().tolist()
    all_id_set = set(all_ids)
    train_ids = set(participants.loc[participants["recommended_split"] == "train", "person_id"])
    if not train_ids:
        raise ValueError("No training participants found")

    task_spec, task_config_hash = load_task_spec(args.task_config, args.task)
    exclusions = task_spec.get("clinical_feature_exclusions", {})
    clinical_dir = Path(args.clinical_dir)
    measurement = pd.read_csv(
        clinical_dir / "measurement.csv", low_memory=False, dtype={"person_id": "string"}
    )
    observation = pd.read_csv(
        clinical_dir / "observation.csv", low_memory=False, dtype={"person_id": "string"}
    )
    validate_clinical_people(measurement, "measurement.csv", all_id_set)
    validate_clinical_people(observation, "observation.csv", all_id_set)
    raw_clinical_ids = set(measurement["person_id"].map(as_person_id)) | set(
        observation["person_id"].map(as_person_id)
    )
    raw_clinical_ids.discard("")

    long_tables: List[Tuple[pd.DataFrame, str]] = []
    audits: List[dict] = []
    meas_long, audit = numeric_long(
        measurement,
        table="measurement",
        source_col="measurement_source_value",
        concept_col="measurement_concept_id",
        value_col="value_as_number",
        unit_col="unit_source_value",
        unit_concept_col="unit_concept_id",
        exclusions=exclusions,
    )
    long_tables.append((meas_long, "mean")); audits.extend(audit)
    obs_num, audit = numeric_long(
        observation,
        table="observation",
        source_col="observation_source_value",
        concept_col="observation_concept_id",
        value_col="value_as_number",
        unit_col="unit_source_value",
        unit_concept_col="unit_concept_id",
        exclusions=exclusions,
    )
    long_tables.append((obs_num, "mean")); audits.extend(audit)
    obs_cat, audit = observation_concept_long(observation, exclusions)
    long_tables.append((obs_cat, "max")); audits.extend(audit)

    if args.include_condition:
        condition = pd.read_csv(
            clinical_dir / "condition_occurrence.csv", low_memory=False,
            dtype={"person_id": "string"},
        )
        validate_clinical_people(
            condition, "condition_occurrence.csv", all_id_set
        )
        raw_clinical_ids.update(condition["person_id"].map(as_person_id))
        cond_long, audit = condition_long(condition, exclusions)
        long_tables.append((cond_long, "max")); audits.extend(audit)
    if args.include_procedure:
        procedure = pd.read_csv(
            clinical_dir / "procedure_occurrence.csv", low_memory=False,
            dtype={"person_id": "string"},
        )
        validate_clinical_people(
            procedure, "procedure_occurrence.csv", all_id_set
        )
        raw_clinical_ids.update(procedure["person_id"].map(as_person_id))
        proc_long, audit = procedure_long(procedure, exclusions)
        long_tables.append((proc_long, "sum")); audits.extend(audit)

    wide_parts: List[pd.DataFrame] = []
    catalogs: List[pd.DataFrame] = []
    constants: List[str] = []
    for long_df, agg in long_tables:
        wide, catalog, dropped = pivot_long(
            long_df,
            all_ids=all_ids,
            train_ids=train_ids,
            min_train_persons=args.min_train_persons,
            aggregation=agg,
        )
        wide_parts.append(wide)
        if not catalog.empty:
            catalogs.append(catalog)
        constants.extend(dropped)

    merged = pd.concat(wide_parts, axis=1)
    if merged.columns.duplicated().any():
        duplicated = merged.columns[merged.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate clinical feature names: {duplicated[:10]}")
    merged.insert(0, "person_id", merged.index)
    age = participants.set_index("person_id")["age"].reindex(all_ids)
    merged.insert(1, "person__age", pd.to_numeric(age, errors="coerce").to_numpy())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / f"clinical_features_{args.task}_wide.csv", index=False)

    age_catalog = pd.DataFrame([{
        "feature_name": "person__age", "table": "participants", "source_key": "age",
        "concept_id": 0, "unit": "years", "unit_concept_id": 0,
        "response_concept_id": 0,
        "value_kind": "numeric", "aggregation": "identity",
        "train_nonmissing_persons": int(age.loc[list(train_ids)].notna().sum()),
        "train_prevalence": float(age.loc[list(train_ids)].notna().mean()),
    }])
    catalog = pd.concat([age_catalog] + catalogs, ignore_index=True)
    catalog.to_csv(out_dir / f"clinical_feature_catalog_{args.task}.csv", index=False)
    pd.DataFrame(audits).to_csv(out_dir / f"clinical_exclusion_audit_{args.task}.csv", index=False)

    feature_names = [c for c in merged.columns if c != "person_id"]
    preprocessing_config = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "builder": "build_clinical_tabular_features.py",
        "task": args.task,
        "task_config_hash": task_config_hash,
        "fit_split": "train",
        "fit_person_ids_hash": canonical_json_hash({"person_ids": sorted(train_ids)}),
        "min_train_persons": int(args.min_train_persons),
        "include_condition": bool(args.include_condition),
        "include_procedure": bool(args.include_procedure),
        "feature_names": feature_names,
        "numeric_feature_key": "source_code+concept_id+(unit_concept_id_or_source_unit)",
        "event_absence_fill": 0.0,
        "numeric_missing_fill": None,
        "clinical_data_quality_rules": {
            "identifier_source_codes": sorted(IDENTIFIER_SOURCE_CODES),
            "identifier_concept_ids": sorted(IDENTIFIER_CONCEPT_IDS),
            "observation_numeric_sentinels": sorted(OBSERVATION_NUMERIC_SENTINELS),
            "observation_numeric_sentinel_source_codes": sorted(
                OBSERVATION_NUMERIC_SENTINEL_SOURCE_CODES
            ),
            "concept_valued_observation_representation": "categorical_only",
        },
    }
    config_hash = canonical_json_hash(preprocessing_config)
    schema = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "config_hash": config_hash,
        "preprocessing_config": preprocessing_config,
        "task": args.task,
        "task_config": str(Path(args.task_config)),
        "task_config_hash": task_config_hash,
        "fit_split": "train",
        "fit_person_ids_hash": preprocessing_config["fit_person_ids_hash"],
        "min_train_persons": int(args.min_train_persons),
        "n_all_persons": int(len(all_ids)),
        "n_train_persons": int(len(train_ids)),
        "n_raw_clinical_persons": int(len(raw_clinical_ids)),
        "raw_clinical_person_ids": sorted(raw_clinical_ids),
        "raw_clinical_person_ids_hash": canonical_json_hash({
            "person_ids": sorted(raw_clinical_ids)
        }),
        "n_features": int(len(feature_names)),
        "feature_names": feature_names,
        "source_files": source_file_metadata([
            args.participants,
            args.task_config,
            clinical_dir / "measurement.csv",
            clinical_dir / "observation.csv",
            clinical_dir / "condition_occurrence.csv" if args.include_condition else None,
            clinical_dir / "procedure_occurrence.csv" if args.include_procedure else None,
        ]),
        "dropped_train_constant_features": sorted(constants),
        "raw_scale": True,
        "missing_value": "NaN",
        "notes": [
            "Feature vocabulary and constant removal are fit on recommended_split=train only.",
            "Task label/proxy exclusions come from downstream_tasks.json.",
            "Categorical observation responses use source-variable plus response concept, not question-concept presence alone.",
            "Numeric features include the OMOP unit concept in their key; incompatible units are never averaged together.",
            "When unit_concept_id is zero, the normalized unit_source_value is used as a fallback key.",
            "Participant identifiers are excluded; concept-valued survey responses are represented categorically rather than as numeric codes.",
            "Cohort-verified 777/888/999 survey sentinels are removed only for their exact source codes; legitimate elapsed-time values are retained outside MoCA masking.",
            "Absent condition/categorical/procedure events are explicit zero; unobserved numeric measurements remain NaN.",
            "Fit imputation and scaling on the training split only.",
        ],
    }
    (out_dir / f"clinical_schema_{args.task}.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(json.dumps({k: schema[k] for k in ["task", "n_all_persons", "n_train_persons", "n_features"]}, indent=2))


if __name__ == "__main__":
    main()
