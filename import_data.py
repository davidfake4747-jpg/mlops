"""Однократный импорт исходных .xlsx без изменения содержимого файлов."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


def import_archive(archive, output="data"):
    archive = Path(archive)
    output = Path(output)
    entries = []
    with zipfile.ZipFile(archive) as source:
        names = set()
        for item in sorted(source.infolist(), key=lambda x: x.filename):
            path = PurePosixPath(item.filename)
            if item.is_dir():
                continue
            if len(path.parts) != 2 or path.parts[0] != "data" or path.suffix != ".xlsx":
                raise ValueError(f"Неожиданный файл архива: {item.filename}")
            if path.name in names:
                raise ValueError(f"Повтор имени в архиве: {path.name}")
            names.add(path.name)
            content = source.read(item)
            entries.append((path.name, content))
    if not entries:
        raise ValueError("Архив не содержит .xlsx")
    output.mkdir(parents=True, exist_ok=True)
    unexpected = set(p.name for p in output.iterdir()) - names - {"source_manifest.json"}
    if unexpected:
        raise ValueError(f"В каталоге данных есть посторонние файлы: {sorted(unexpected)}")
    # Проверка всех существующих файлов до записи, чтобы не перетирать чужие данные.
    for name, content in entries:
        target = output / name
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f"Существующий файл отличается: {target}")
    manifest = {
        "archive": archive.name,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files": {name: hashlib.sha256(content).hexdigest() for name, content in entries},
    }
    for name, content in entries:
        (output / name).write_bytes(content)
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Импортировано {len(entries)} Excel-файлов в {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("--output", default="data")
    args = parser.parse_args()
    import_archive(args.archive, args.output)
