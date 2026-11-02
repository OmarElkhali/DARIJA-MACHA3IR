"""Après C et l'extraction D, exécuter B/D et compléter les graines F2."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.train import classical_run, run_campaign
from src.preprocess import ROOT, load_split

# Mise à jour descriptive du détecteur, avant verrouillage ; aucun label ou score ne change.
dev = load_split("dev").set_index("row_id")
for file in (ROOT / "results").glob("*_dev_*.csv"):
    predictions = pd.read_csv(file)
    assert (predictions.label.to_numpy() == predictions.row_id.map(dev.label).to_numpy()).all()
    predictions["script"] = predictions.row_id.map(dev.script)
    predictions.to_csv(file, index=False)

for script in ("arabic", "latin"):
    for seed in (43, 44):
        classical_run("union", seed, train_script=script)
run_campaign("B")
run_campaign("D")
