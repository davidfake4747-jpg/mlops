import hashlib
import json
import zipfile

import pandas as pd
import pytest

from import_data import import_archive
from prepare_data import clean_records, normalize_address
from preprocess import process_excel_file
from split_data import split_records


def record(ident="1", **changes):
    return dict({
        "id": ident, "area": 45.5, "rooms": 2, "floor": 3, "total_floors": 9,
        "price": 15_000_000, "address": "Москва, Улица Тестовая, 1 к2",
        "property_type": "flat", "source_file": "example.xlsx", "source_row": 2,
    }, **changes)


def test_extract_uses_fields_instead_of_description(monkeypatch):
    raw = pd.DataFrame([{
        "sNO6x  href": "https://domclick.ru/card/sale__flat__123",
        "s6KKu": "Студия", "s6KKu 2": "25,5 м²", "s6KKu 3": "2/9 эт.",
        "sS4xs": "Москва, Тестовая, 1", "s5oAg": "12\u202f500\u00a0000 ₽",
        "description": "Раньше 99 000 000 ₽, площадь всего дома 5000 м²",
        "sa3w3": "490 000 ₽/м²",
    }])
    monkeypatch.setattr(pd, "read_excel", lambda *args, **kwargs: raw.copy())
    result = process_excel_file("example.xlsx").iloc[0]
    assert result["price"] == 12_500_000
    assert result["area"] == 25.5
    assert result["rooms"] == 0
    assert result["id"] == "123"


def test_unknown_excel_schema_fails(monkeypatch):
    monkeypatch.setattr(pd, "read_excel", lambda *a, **kw: pd.DataFrame({"new": ["format"]}))
    with pytest.raises(ValueError, match="отсутствуют колонки"):
        process_excel_file("changed.xlsx")


def test_negative_values_in_excel_are_not_read_as_positive(monkeypatch):
    raw = pd.DataFrame([{
        "sNO6x href": "https://domclick.ru/card/sale__flat__123",
        "s6KKu": "-2-комн. квартира", "s6KKu 2": "-25,5 м²", "s6KKu 3": "-2/9 эт.",
        "sS4xs": "Москва, Тестовая, 1", "s5oAg": "-12 500 000 ₽",
    }])
    monkeypatch.setattr(pd, "read_excel", lambda *args, **kwargs: raw.copy())
    result = process_excel_file("example.xlsx").iloc[0]
    assert all(pd.isna(result[field]) for field in ["rooms", "area", "floor", "price"])


def test_address_preserves_building_parts():
    assert normalize_address(" Москва , Улица  Тестовая, 1 к2 ") == "москва, улица тестовая, 1 к2"
    assert normalize_address("Москва, Дом, 1 к2") != normalize_address("Москва, Дом, 1 к3")


def test_duplicates_and_conflicts_are_handled_before_split():
    rows = [record(), record(source_file="other.xlsx"), record("2"), record("2", price=1)]
    clean, excluded, report = clean_records(pd.DataFrame(rows))
    assert clean["id"].tolist() == ["1"]
    assert report["duplicate_rows"] == 2
    assert report["conflicting_ids"] == ["2"]
    assert excluded.iloc[0]["reason"] == "conflicting_id"


@pytest.mark.parametrize("changes,reason", [
    ({"area": float("inf")}, "invalid_area"),
    ({"area": -1}, "out_of_range_area"),
    ({"price": 0}, "out_of_range_price"),
    ({"floor": 10}, "floor_above_building"),
    ({"rooms": 1.5}, "non_integer_rooms"),
    ({"address": " "}, "missing_address"),
    ({"id": None}, "missing_id"),
    ({"property_type": "apartment"}, "apartment_excluded"),
])
def test_invalid_records_have_reasons(changes, reason):
    clean, excluded, _ = clean_records(pd.DataFrame([record(**changes)]))
    assert clean.empty
    assert reason in excluded.iloc[0]["reason"]


def test_studios_and_large_flats_are_retained():
    clean, _, _ = clean_records(pd.DataFrame([record("1", rooms=0), record("2", rooms=5)]))
    assert clean["rooms"].tolist() == [0, 5]


def test_split_and_cv_keep_houses_together_and_are_reproducible():
    frame = pd.DataFrame([
        record(str(i), address=f"москва, тестовая, {i // 2}") for i in range(30)
    ])
    train, test = split_records(frame)
    assert not set(train["id"]) & set(test["id"])
    assert not set(train["address"]) & set(test["address"])
    assert train.groupby("address")["cv_fold"].nunique().eq(1).all()
    assert set(train["cv_fold"]) == set(range(5))
    again_train, again_test = split_records(frame.sample(frac=1, random_state=7))
    pd.testing.assert_frame_equal(train, again_train)
    pd.testing.assert_frame_equal(test, again_test)


def test_not_enough_groups_fails():
    frame = pd.DataFrame([record(str(i), address=f"дом {i}") for i in range(4)])
    with pytest.raises(ValueError, match="недостаточно"):
        split_records(frame)


def test_archive_import_preserves_bytes_and_can_repeat(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr("data/example.xlsx", b"unchanged source bytes")
    output = tmp_path / "data"
    import_archive(archive, output)
    import_archive(archive, output)
    assert (output / "example.xlsx").read_bytes() == b"unchanged source bytes"
    manifest = json.loads((output / "source_manifest.json").read_text())
    assert manifest["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()


def test_archive_does_not_write_unexpected_paths(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr("data/../escape.xlsx", b"invalid")
    with pytest.raises(ValueError, match="Неожиданный"):
        import_archive(archive, tmp_path / "output")
