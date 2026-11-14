import argparse
import os
import re

import pandas as pd

# Возможные варианты написания заголовка колонки со ссылкой
ID_COLUMN_CANDIDATES = ["sNO6x href", "sNO6x  href", "href"]


def find_id_column(df: pd.DataFrame) -> int | None:
    """Ищет индекс колонки, содержащей ссылки на объявления domclick."""
    # 1. По точному совпадению заголовка в первых строках
    for col_idx in df.columns:
        for row_idx in range(min(5, len(df))):
            cell = str(df.iat[row_idx, col_idx]).strip().lower()
            if cell in [c.lower() for c in ID_COLUMN_CANDIDATES]:
                return col_idx

    # 2. Fallback: ищем колонку, где большинство значений — ссылки на domclick card
    for col_idx in df.columns:
        col_vals = df[col_idx].dropna().astype(str)
        if len(col_vals) == 0:
            continue
        hits = col_vals.str.contains(r"/card/[a-z_]+__\d+", regex=True).sum()
        if hits >= max(1, len(col_vals) * 0.5):
            return col_idx

    return None


def extract_id_from_url(text: str) -> str | None:
    """Достаёт числовой id из URL объявления domclick."""
    if not isinstance(text, str):
        return None
    m = re.search(r"sale__flat__(\d+)", text)
    if m:
        return m.group(1)
    m = re.search(r"/card/[a-z_]+__(\d+)", text)
    if m:
        return m.group(1)
    return None


def extract_features_from_text(text: str) -> dict:
    """Извлекает признаки (кроме id) из склеенного текста строки."""
    result = {
        "rooms": None,
        "area": None,
        "floor": None,
        "total_floors": None,
        "price": None,
    }
    if not isinstance(text, str):
        return result

    m = re.search(r"(\d+)\s*-\s*комн", text, re.IGNORECASE)
    if m:
        result["rooms"] = int(m.group(1))
    elif re.search(r"студи", text, re.IGNORECASE):
        result["rooms"] = 0

    m = re.search(r"(\d+(?:[.,]\d+)?)\s*м[²2]", text)
    if m:
        result["area"] = float(m.group(1).replace(",", "."))

    m = re.search(r"(\d+)\s*/\s*(\d+)\s*эт", text)
    if m:
        result["floor"] = int(m.group(1))
        result["total_floors"] = int(m.group(2))

    m = re.search(r"([\d\s\u00a0\u202f]+)\s*₽", text)
    if m:
        price_str = re.sub(r"[\s\u00a0\u202f]", "", m.group(1))
        try:
            result["price"] = int(price_str)
        except ValueError:
            pass

    return result


def process_excel_file(path: str) -> pd.DataFrame:
    """Читает поля выгрузки; описания и ипотечные суммы не используются."""
    df = pd.read_excel(path, dtype=str, engine="openpyxl").fillna("")
    df.columns = [" ".join(str(c).split()) for c in df.columns]
    columns = {
        "id": "sNO6x href",
        "rooms": "s6KKu",
        "area": "s6KKu 2",
        "floor": "s6KKu 3",
        "price": "s5oAg",
        "address": "sS4xs",
    }
    missing = set(columns.values()) - set(df.columns)
    if missing:
        raise ValueError(f"{os.path.basename(path)}: отсутствуют колонки {sorted(missing)}")
    if df.columns.duplicated().any():
        raise ValueError(f"{os.path.basename(path)}: повторяющиеся заголовки")

    records = []
    for index, row in df.iterrows():
        if not any(row[c] for c in columns.values()):
            continue
        # Изолированное извлечение не позволяет взять цену/площадь из описания.
        room_text = row[columns["rooms"]].strip()
        room_match = re.fullmatch(
            r"(\d+)\s*-\s*комн\.?\s+(?:квартира|апартаменты)", room_text, re.IGNORECASE
        )
        studio = re.fullmatch(r"(?:(?:квартира|апартаменты)-)?студия", room_text, re.IGNORECASE)
        rooms = int(room_match[1]) if room_match else 0 if studio else None
        area_match = re.fullmatch(r"(\d+(?:[.,]\d+)?)\s*м[²2]", row[columns["area"]].strip())
        area = float(area_match[1].replace(",", ".")) if area_match else None
        floor_match = re.fullmatch(r"(\d+)\s*/\s*(\d+)\s*эт\.?", row[columns["floor"]].strip())
        price_text = row[columns["price"]].strip()
        price = None
        if re.fullmatch(r"[\d\s\u00a0\u202f]+\s*₽", price_text):
            price = int(re.sub(r"\s", "", price_text.removesuffix("₽")))
        kind = row[columns["rooms"]].casefold()
        property_type = (
            "apartment" if "апартамент" in kind
            else "flat" if "квартир" in kind or "студи" in kind
            else "unknown"
        )
        records.append({
            "id": extract_id_from_url(row[columns["id"]]),
            "rooms": rooms,
            "area": area,
            "floor": int(floor_match[1]) if floor_match else None,
            "total_floors": int(floor_match[2]) if floor_match else None,
            "price": price,
            "address": row[columns["address"]],
            "property_type": property_type,
            "source_file": os.path.basename(path),
            "source_row": int(index) + 2,
        })
    return pd.DataFrame(records, columns=[
        "id", "rooms", "area", "floor", "total_floors", "price", "address",
        "property_type", "source_file", "source_row",
    ])


def main(folder: str, out_dir: str = "features", out_name: str = "all_features.csv"):
    from prepare_data import prepare

    prepare(folder, os.path.join(out_dir, out_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Подготовка выгрузок Domclick")
    parser.add_argument("--input", default="data")
    parser.add_argument("--output", default="features/all_features.csv")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()
    from prepare_data import prepare

    prepare(args.input, args.output, args.report_dir)
