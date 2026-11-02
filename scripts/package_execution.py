"""Archive portable du code et des données ; aucun entraînement ni poids historique."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
target = ROOT / "Darija-Sentiment-execution.zip"
files = [ROOT / name for name in (
    "README.md", "requirements.txt", "requirements-notebook.txt", "notebooks/TP2_Darija.ipynb", "notebooks/TP2_Darija.html",
    "notebooks/TP2_Darija_observe_avant_v4.ipynb", "research/observed_v3_dev.csv",
    "scripts/build_notebook.py", "scripts/execute_notebook.py", "scripts/package_execution.py",
    "scripts/notebook_bootstrap.py", "scripts/notebook_setup.py",
    "scripts/backup_results.py", "scripts/cache_pretrained.py", "research/migration_v4.md", "tests/check_improvements.py",
    "research/verification_v4.json",
    "tests/check_campaign.py", "tests/check_notebook_setup.py", "research/questions_officielles.md", "research/papers.md", "data/manifest.json",
)]
files += list((ROOT / "src").glob("*.py"))
files += list((ROOT / "data/ary").glob("*.tsv"))
files += [p for p in (ROOT / "data/review").glob("*") if p.is_file()]
files += [p for p in (ROOT / "data/provenance").glob("*") if p.is_file()]
for path in files:
    if not path.is_file():
        raise FileNotFoundError(path)
with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
    for path in sorted(set(files)):
        archive.write(path, str(Path("Darija-Sentiment") / path.relative_to(ROOT)))
with ZipFile(target) as archive:
    assert archive.testzip() is None
    assert "Darija-Sentiment/src/campaign.py" in archive.namelist()
    assert not any("/models/" in name or "/results/" in name for name in archive.namelist())
print(f"{target} : {len(files)} fichiers, {target.stat().st_size} octets")
