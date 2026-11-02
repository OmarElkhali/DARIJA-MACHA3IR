"""Matérialiser un fichier Drive dans la copie locale de campagne."""
from pathlib import Path
import argparse
import shutil
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "experiments/campaign_v4"
    target = (root / args.target).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Destination hors de la campagne locale")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".download")
    try:
        with urllib.request.urlopen(args.url, timeout=120) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=4 * 1024 * 1024)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
