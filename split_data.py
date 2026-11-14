"""Фиксированный holdout по домам; никаких случайных split по строкам."""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def split_records(frame, test_size=0.2, random_state=42, n_splits=5):
    if frame.empty or frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("Нужны непустые данные с уникальными ID")
    if frame["address"].isna().any() or frame["address"].astype(str).str.strip().eq("").any():
        raise ValueError("Для группировки нужны адреса у всех объявлений")
    if not 0 < test_size < 1 or n_splits < 2:
        raise ValueError("Неверные параметры split")
    frame = frame.sort_values("id").reset_index(drop=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_index, test_index = next(splitter.split(frame, groups=frame["address"]))
    train = frame.iloc[train_index].copy().reset_index(drop=True)
    test = frame.iloc[test_index].copy().reset_index(drop=True)
    if train["address"].nunique() < n_splits:
        raise ValueError(f"Для GroupKFold({n_splits}) недостаточно train-групп")
    if set(train["address"]) & set(test["address"]) or set(train["id"]) & set(test["id"]):
        raise ValueError("Утечка между train и holdout")
    train["cv_fold"] = -1
    for fold, (_, validation) in enumerate(
        GroupKFold(n_splits=n_splits).split(train, groups=train["address"])
    ):
        train.loc[validation, "cv_fold"] = fold
    return train, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="features/all_features.csv")
    parser.add_argument("--output-dir", default="features")
    parser.add_argument("--report", default="reports/split.json")
    args = parser.parse_args()
    config = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))["split"]
    frame = pd.read_csv(args.input, dtype={"id": str})
    train, test = split_records(frame, **config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train.to_csv(output / "train.csv", index=False)
    test.to_csv(output / "test.csv", index=False)
    assignment = pd.concat([
        train[["id", "cv_fold"]].assign(split="train"),
        test[["id"]].assign(cv_fold=-1, split="test"),
    ]).sort_values("id")
    assignment.to_csv(output / "split_assignments.csv", index=False)
    report = {
        "parameters": config,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_groups": int(train["address"].nunique()),
        "test_groups": int(test["address"].nunique()),
        "overlapping_ids": 0,
        "overlapping_addresses": 0,
        "cv_validation_rows": train["cv_fold"].value_counts().sort_index().to_dict(),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Train: {len(train)}; holdout: {len(test)}; пересечений адресов нет")


if __name__ == "__main__":
    main()
