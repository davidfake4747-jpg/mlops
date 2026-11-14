"""Reference строится только по train, без адресов, цен и ID."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from service.drift import build_reference


def main():
    train = pd.read_csv("features/train.csv")
    data_hash = yaml.safe_load(Path("data.dvc").read_text(encoding="utf-8"))["outs"][0]["md5"]
    model_sha256 = hashlib.sha256(Path("models/selected.joblib").read_bytes()).hexdigest()
    reference = build_reference(train, data_hash, model_sha256)
    Path("reports/reference.json").write_text(
        json.dumps(reference, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Reference: {len(train)} train-объектов, признаки area/rooms")


if __name__ == "__main__":
    main()
