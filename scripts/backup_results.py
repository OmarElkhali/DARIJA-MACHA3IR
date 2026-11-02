"""Snapshot vérifié avant modification ; aucune métrique n'est recalculée."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(source, destination, directories=("notebooks", "results", "models", "experiments", "src", "scripts")):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or destination.is_relative_to(source):
        raise ValueError("La sauvegarde doit être hors du projet source.")
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    files = [p for name in directories for p in (source / name).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts]
    files += [p for p in source.glob("requirements*.txt")]
    files += [p for p in source.glob("README*.md")]
    for path in files:
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        digest = checksum(path)
        if checksum(target) != digest:
            raise OSError(f"Copie différente : {path}")
        hashes[path.relative_to(source).as_posix()] = digest
    notebook = source / "notebooks/TP2_Darija.ipynb"
    if notebook.exists():
        nb = json.loads(notebook.read_text(encoding="utf-8"))
        logs = "\n".join("".join(o.get("text", [])) for c in nb["cells"] for o in c.get("outputs", []))
        (destination / "notebook_run_log.txt").write_text(logs, encoding="utf-8")
        measured = re.findall(r"([A-Z]_\S+): dev F1=([\d.]+), macro=([\d.]+), ([\d.]+)s", logs)
        import csv
        with (destination / "notebook_measured_dev.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["run_id", "dev_weighted_f1_displayed", "dev_macro_f1_displayed", "seconds_displayed"])
            writer.writerows(measured)
    (destination / "snapshot_manifest.json").write_text(json.dumps({
        "source": str(source), "created_utc": datetime.now(timezone.utc).isoformat(), "sha256": hashes,
        "scope": "Fichiers locaux et sorties du notebook ; les poids Colab non téléchargés ne sont pas inclus. Les scores extraits sont arrondis comme dans les sorties."
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(snapshot(args.source, args.destination))
