#!/usr/bin/env python3
"""Build a task-independent, metadata-rich retinal asset index.

Unlike the legacy eight-slot index, this index does not silently collapse CFP,
IR, AF, OCTA slabs, or FLIO wavelengths into one semantic slot.  It expands the
four manifests into a common long schema and marks deterministic candidates for
model views without reading DICOM pixels.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
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


MISSING = {"", "nan", "none", "not reported", "not_provided"}


def valid(value: Any) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip().lower() not in MISSING


def norm(value: Any) -> str:
    return "" if value is None or pd.isna(value) else re.sub(r"\s+", " ", str(value).strip())


def norm_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", norm(value).lower()).strip("_")


def region_score(region: Any, modality: str) -> float:
    value = norm(region).lower()
    if "macula" in value:
        if "6 x 6" in value or "6x6" in value:
            return 120.0
        if "12 x 12" in value or "12x12" in value:
            return 115.0
        if "20 x 20" in value or "20x20" in value:
            return 110.0
        return 125.0
    if "wide" in value or "mosaic" in value:
        return 90.0
    if "optic" in value or "disc" in value:
        return 70.0
    if modality == "photography" and ("nasal" in value or "temporal" in value):
        return 50.0
    return 0.0


def photo_protocol(value: Any) -> str:
    text = norm(value).lower()
    if "color" in text:
        return "cfp"
    if "infrared" in text:
        return "ir"
    if "autofluorescence" in text:
        return "af"
    return norm_token(value) or "unknown"


def common_row(row: pd.Series, *, modality: str, protocol: str, path: Any, sop_uid: Any) -> Dict[str, Any]:
    return {
        "person_id": as_person_id(row.get("person_id")),
        "modality": modality,
        "protocol": protocol,
        "laterality": norm(row.get("laterality")).upper(),
        "manufacturer": norm(row.get("manufacturer")),
        "device_model": norm(row.get("manufacturers_model_name")),
        "anatomic_region": norm(row.get("anatomic_region")),
        "slab_or_wavelength": "",
        "filepath": norm(path),
        "sop_instance_uid": norm(sop_uid),
        "height": np.nan,
        "width": np.nan,
        "number_of_frames": np.nan,
        "pixel_spacing": "",
        "slice_thickness": np.nan,
        "is_projection_removed": False,
        "source_variant": "primary",
        "source_manifest_row": int(row.name),
    }


def photography_assets(df: pd.DataFrame) -> List[Dict[str, Any]]:
    assets = []
    for _, row in df.iterrows():
        if not valid(row.get("filepath")):
            continue
        item = common_row(
            row, modality="photography", protocol=photo_protocol(row.get("imaging")),
            path=row.get("filepath"), sop_uid=row.get("sop_instance_uid"),
        )
        item.update({
            "height": pd.to_numeric(row.get("height"), errors="coerce"),
            "width": pd.to_numeric(row.get("width"), errors="coerce"),
            "color_channels": pd.to_numeric(row.get("color_channel_dimension"), errors="coerce"),
        })
        assets.append(item)
    return assets


def oct_assets(df: pd.DataFrame) -> List[Dict[str, Any]]:
    assets = []
    for _, row in df.iterrows():
        if not valid(row.get("filepath")):
            continue
        item = common_row(row, modality="oct", protocol="volume", path=row.get("filepath"), sop_uid=row.get("sop_instance_uid"))
        item.update({
            "height": pd.to_numeric(row.get("height"), errors="coerce"),
            "width": pd.to_numeric(row.get("width"), errors="coerce"),
            "number_of_frames": pd.to_numeric(row.get("number_of_frames"), errors="coerce"),
            "pixel_spacing": norm(row.get("pixel_spacing")),
            "slice_thickness": pd.to_numeric(row.get("slice_thickness"), errors="coerce"),
            "reference_filepath": norm(row.get("reference_filepath")),
        })
        assets.append(item)
    return assets


def octa_assets(df: pd.DataFrame) -> List[Dict[str, Any]]:
    assets: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        if valid(row.get("flow_cube_file_path")):
            item = common_row(
                row, modality="octa", protocol="flow_cube",
                path=row.get("flow_cube_file_path"), sop_uid=row.get("flow_cube_sop_instance_uid"),
            )
            item.update({
                "height": pd.to_numeric(row.get("flow_cube_height"), errors="coerce"),
                "width": pd.to_numeric(row.get("flow_cube_width"), errors="coerce"),
                "number_of_frames": pd.to_numeric(row.get("flow_cube_number_of_frames"), errors="coerce"),
                "slab_or_wavelength": "flow_cube",
            })
            assets.append(item)

        for index in range(1, 5):
            base = f"associated_enface_{index}"
            projection_path = row.get(f"{base}_projection_removed_filepath")
            projection_uid = row.get(f"{base}_projection_removed_sop_instance_uid")
            standard_path = row.get(f"{base}_file_path")
            standard_uid = row.get(f"{base}_sop_instance_uid")
            slab = norm_token(row.get(f"{base}_ophthalmic_image_type")) or f"enface_{index}"
            variants = [
                ("standard", standard_path, standard_uid, False),
                ("projection_removed", projection_path, projection_uid, True),
            ]
            for variant, path, uid, is_projection_removed in variants:
                if not valid(path):
                    continue
                item = common_row(
                    row, modality="octa", protocol="enface", path=path, sop_uid=uid
                )
                item.update({
                    # The manifest reports flow-cube dimensions, not en-face
                    # dimensions.  Do not mislabel those values as 2D asset size.
                    "height": np.nan,
                    "width": np.nan,
                    "number_of_frames": 1,
                    "slab_or_wavelength": slab,
                    "is_projection_removed": bool(is_projection_removed),
                    "source_variant": variant,
                    "segmentation_surface_1": norm(row.get(f"{base}_segmentation_surface_1")),
                    "segmentation_surface_2": norm(row.get(f"{base}_segmentation_surface_2")),
                })
                assets.append(item)
    return assets


def flio_assets(df: pd.DataFrame) -> List[Dict[str, Any]]:
    assets = []
    for _, row in df.iterrows():
        if not valid(row.get("filepath")):
            continue
        wavelength = norm_token(row.get("wavelength")) or "unknown"
        item = common_row(row, modality="flio", protocol="decay_cube", path=row.get("filepath"), sop_uid=row.get("sop_instance_uid"))
        item.update({
            "height": pd.to_numeric(row.get("height"), errors="coerce"),
            "width": pd.to_numeric(row.get("width"), errors="coerce"),
            "number_of_frames": pd.to_numeric(row.get("number_of_frames"), errors="coerce"),
            "slab_or_wavelength": wavelength,
        })
        assets.append(item)
    return assets


def rank_assets(assets: pd.DataFrame) -> pd.DataFrame:
    out = assets.copy()
    out["_area"] = pd.to_numeric(out["height"], errors="coerce").fillna(0) * pd.to_numeric(out["width"], errors="coerce").fillna(0)
    out["selection_score"] = out.apply(lambda r: region_score(r["anatomic_region"], r["modality"]), axis=1)
    out["selection_score"] += np.log1p(pd.to_numeric(out["number_of_frames"], errors="coerce").fillna(0)) / 100.0
    out["selection_score"] += np.log1p(out["_area"]) / 1000.0
    out["selection_score"] += out["is_projection_removed"].astype(float) * 5.0

    group_cols = ["person_id", "laterality", "modality", "protocol", "slab_or_wavelength"]
    out = out.sort_values(
        group_cols + ["selection_score", "filepath"],
        ascending=[True] * len(group_cols) + [False, True],
        kind="mergesort",
    )
    out["selection_rank"] = out.groupby(group_cols, dropna=False).cumcount() + 1
    out["selected_for_default_view"] = out["selection_rank"].eq(1)
    return out.drop(columns=["_area"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", required=True)
    ap.add_argument("--retinal_photography", required=True)
    ap.add_argument("--retinal_oct", required=True)
    ap.add_argument("--retinal_octa", required=True)
    ap.add_argument("--retinal_flio", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    participants = pd.read_csv(
        args.participants, sep="\t", dtype={"person_id": "string"}
    )
    required_participant_columns = {
        "person_id", "recommended_split", "clinical_site"
    }
    missing = sorted(required_participant_columns - set(participants.columns))
    if missing:
        raise ValueError(f"participants.tsv missing required columns: {missing}")
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
    participant_meta = participants.set_index("person_id")

    photo = pd.read_csv(args.retinal_photography, sep="\t", low_memory=False, dtype={"person_id": "string"})
    oct_df = pd.read_csv(args.retinal_oct, sep="\t", low_memory=False, dtype={"person_id": "string"})
    octa = pd.read_csv(args.retinal_octa, sep="\t", low_memory=False, dtype={"person_id": "string"})
    flio = pd.read_csv(args.retinal_flio, sep="\t", low_memory=False, dtype={"person_id": "string"})
    rows = photography_assets(photo) + oct_assets(oct_df) + octa_assets(octa) + flio_assets(flio)
    assets = pd.DataFrame(rows)
    assets["person_id"] = assets["person_id"].map(as_person_id)
    if (assets["person_id"] == "").any():
        raise ValueError("Retinal manifests contain an empty person_id")
    assets["asset_id"] = assets.apply(
        lambda row: f"{row['modality']}-" + canonical_json_hash({
            "person_id": row["person_id"],
            "filepath": row["filepath"],
            "sop_instance_uid": row["sop_instance_uid"],
            "protocol": row["protocol"],
            "slab_or_wavelength": row["slab_or_wavelength"],
            "source_variant": row.get("source_variant", "primary"),
        })[:20],
        axis=1,
    )
    if assets["asset_id"].duplicated().any():
        duplicate_ids = assets.loc[
            assets["asset_id"].duplicated(keep=False), "asset_id"
        ].tolist()
        raise ValueError(f"Retinal index contains duplicate asset identities: {duplicate_ids[:20]}")
    assets = rank_assets(assets)
    assets["split"] = assets["person_id"].map(participant_meta["recommended_split"])
    assets["clinical_site"] = assets["person_id"].map(participant_meta["clinical_site"])

    missing_people = sorted(set(assets.loc[assets["split"].isna(), "person_id"]))
    if missing_people:
        raise ValueError(f"Retinal assets have person IDs absent from participants.tsv: {missing_people[:20]}")

    uid = assets[assets["sop_instance_uid"].astype(str).str.len() > 0]
    uid_people = uid.groupby("sop_instance_uid")["person_id"].nunique()
    duplicate_uids = set(uid_people[uid_people > 1].index)
    duplicate_audit = uid[uid["sop_instance_uid"].isin(duplicate_uids)].copy()

    manifest_paths = [
        args.participants,
        args.retinal_photography,
        args.retinal_oct,
        args.retinal_octa,
        args.retinal_flio,
    ]
    preprocessing_config = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "builder": "build_retinal_asset_index.py",
        "selection_group": [
            "person_id", "laterality", "modality", "protocol", "slab_or_wavelength"
        ],
        "selection_priority": [
            "macular_region", "projection_removed", "frames", "pixel_area", "filepath"
        ],
        "preserve_standard_and_projection_removed_octa_enface": True,
    }
    config_hash = canonical_json_hash(preprocessing_config)
    source_files = source_file_metadata(manifest_paths)
    source_fingerprint = canonical_json_hash({"source_files": source_files})
    assets["config_hash"] = config_hash
    assets["source_fingerprint"] = source_fingerprint
    duplicate_audit = assets.loc[duplicate_audit.index].copy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets.to_csv(out_dir / "retinal_asset_index.csv", index=False)
    assets[assets["selected_for_default_view"]].to_csv(
        out_dir / "retinal_default_model_view.csv", index=False
    )
    duplicate_audit.to_csv(
        out_dir / "retinal_cross_person_duplicate_uid_audit.csv", index=False
    )
    summary = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "config_hash": config_hash,
        "preprocessing_config": preprocessing_config,
        "source_files": source_files,
        "source_fingerprint": source_fingerprint,
        "n_assets": int(len(assets)),
        "n_persons": int(assets["person_id"].nunique()),
        "assets_by_modality": assets["modality"].value_counts().to_dict(),
        "selected_by_modality": assets.loc[assets["selected_for_default_view"], "modality"].value_counts().to_dict(),
        "cross_person_duplicate_sop_uids": int(len(duplicate_uids)),
        "notes": [
            "CFP, IR, and AF remain separate photography protocols.",
            "OCTA flow cubes and each en-face slab remain separate assets; projection-removed assets are preferred.",
            "When both standard and projection-removed en-face files exist, both remain indexed.",
            "FLIO short/long wavelength decay cubes remain separate.",
            "Selection is task-independent and deterministic; pixel preprocessing is a separate stage.",
        ],
    }
    (out_dir / "retinal_asset_schema.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
