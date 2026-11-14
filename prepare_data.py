"""Очистка объявлений и отчёт о качестве без обучения модели."""

import json
import math
import re
from pathlib import Path

import pandas as pd
import yaml

from preprocess import process_excel_file

NUMERIC = ["rooms", "area", "floor", "total_floors", "price"]
CONTENT = NUMERIC + ["address", "property_type"]


def normalize_address(value):
    # Не удаляем корпус/строение и не пытаемся угадывать отсутствующий дом.
    return re.sub(r"\s*,\s*", ", ", " ".join(str(value).casefold().split())).strip()


def invalid_reasons(row, exclude_apartments=True):
    reasons = []
    if pd.isna(row["id"]) or not str(row["id"]).strip():
        reasons.append("missing_id")
    if not row["address"]:
        reasons.append("missing_address")
    if row["property_type"] == "unknown":
        reasons.append("unknown_property_type")
    if exclude_apartments and row["property_type"] == "apartment":
        reasons.append("apartment_excluded")
    for field in NUMERIC:
        value = row[field]
        if pd.isna(value) or not math.isfinite(value):
            reasons.append(f"invalid_{field}")
        elif field in ("rooms", "floor", "total_floors") and value != int(value):
            reasons.append(f"non_integer_{field}")
        elif field == "rooms" and value < 0:
            reasons.append("out_of_range_rooms")
        elif field in ("floor", "total_floors") and value < 1:
            reasons.append(f"out_of_range_{field}")
        elif field in ("area", "price") and value <= 0:
            reasons.append(f"out_of_range_{field}")
    if row["floor"] > row["total_floors"]:
        reasons.append("floor_above_building")
    return reasons


def clean_records(frame, exclude_apartments=True):
    df = frame.copy()
    df["address"] = df["address"].fillna("").map(normalize_address)
    for field in NUMERIC:
        df[field] = pd.to_numeric(df[field], errors="coerce")
    known = df[df["id"].notna() & df["id"].astype(str).str.strip().ne("")]
    conflicts = known.groupby("id")[CONTENT].nunique(dropna=False).gt(1).any(axis=1)
    conflict_ids = set(conflicts[conflicts].index)
    unique = pd.concat([
        known.drop_duplicates("id", keep="first"),
        df.loc[~df.index.isin(known.index)],
    ]).sort_values(["id", "source_file", "source_row"], na_position="last")
    reasons = []
    for _, row in unique.iterrows():
        issues = invalid_reasons(row, exclude_apartments)
        if row["id"] in conflict_ids:
            issues.append("conflicting_id")
        reasons.append(";".join(issues))
    unique["reason"] = reasons
    excluded = unique[unique["reason"].ne("")].copy()
    clean = unique[unique["reason"].eq("")].drop(columns="reason").copy()
    for field in ("rooms", "floor", "total_floors", "price"):
        clean[field] = clean[field].astype("int64")
    stats = {
        "input_rows": len(df),
        "unique_ids": int(known["id"].nunique()),
        "duplicate_rows": len(known) - int(known["id"].nunique()),
        "conflicting_ids": sorted(conflict_ids),
        "excluded_records": len(excluded),
        "exclusion_reasons": excluded["reason"].str.split(";").explode().value_counts().to_dict(),
        "output_rows": len(clean),
        "address_groups": int(clean["address"].nunique()),
    }
    return clean.reset_index(drop=True), excluded.reset_index(drop=True), stats


def prepare(folder="data", output="features/all_features.csv", report_dir="reports"):
    files = sorted(Path(folder).glob("*.xlsx"))
    if not files:
        raise ValueError(f"Нет .xlsx в {folder}; сначала импортируйте архив и выполните dvc pull")
    frame = pd.concat([process_excel_file(str(f)) for f in files], ignore_index=True)
    if frame.empty:
        raise ValueError("В выгрузках нет объявлений")
    config = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    clean, excluded, report = clean_records(frame, config["prepare"]["exclude_apartments"])
    report["source_files"] = {p.name: int((frame["source_file"] == p.name).sum()) for p in files}
    report["numeric_summary"] = (
        clean[NUMERIC].describe().astype(object).where(lambda x: x.notna(), None).to_dict()
        if not clean.empty else {}
    )
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    excluded.to_csv(Path(report_dir) / "excluded_rows.csv", index=False)
    (Path(report_dir) / "data_quality.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if clean.empty:
        raise ValueError("После очистки не осталось объявлений; см. reports/excluded_rows.csv")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output, index=False)
    print(f"Подготовлено {len(clean)} квартир; исключено {len(excluded)} записей")
    return clean
