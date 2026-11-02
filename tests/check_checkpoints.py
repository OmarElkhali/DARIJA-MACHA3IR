"""Recharge tous les checkpoints disponibles sans réentraîner ni choisir sur test."""
from pathlib import Path
import gc
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from src.preprocess import ROOT, LABELS, load_split, save_json
from src.evaluate import assert_locked
from src.train import load_predictor


def main():
    assert_locked()
    # Un vrai exemple dev de chaque écriture ; aucun label test ne participe au contrôle.
    examples = load_split("dev").groupby("script", sort=False).head(1)
    checks = []
    for metadata_file in sorted((ROOT / "models").glob("*/meta.json")):
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        started = time.perf_counter()
        predictor = load_predictor(metadata_file.parent)
        scores = predictor(examples.text.tolist())
        expected = pd.read_csv(ROOT / f"results/{metadata['run_id']}_dev_tuned.csv").set_index("row_id")
        expected = expected.loc[examples.row_id, [f"p_{label}" for label in LABELS]].to_numpy()
        assert scores.shape == (len(examples), 3), metadata["run_id"]
        assert np.isfinite(scores).all() and (scores >= 0).all(), metadata["run_id"]
        assert np.allclose(scores.sum(axis=1), 1, atol=1e-5), metadata["run_id"]
        # Les états D sont stockés en float16 ; la taille des batches peut changer les arrondis.
        maximum_error = float(np.max(np.abs(scores - expected)))
        assert np.allclose(scores, expected, atol=5e-4, rtol=1e-4), (metadata["run_id"], maximum_error)
        checks.append({"run_id": metadata["run_id"], "family": metadata["family"], "n_dev_examples": len(examples), "scripts": examples.script.tolist(),
                       "max_absolute_score_difference": maximum_error, "seconds": time.perf_counter() - started, "passed": True})
        print(f"{metadata['run_id']} : rechargement OK, écart max={maximum_error:.2g}", flush=True)
        del predictor, scores, expected
        gc.collect()
    assert_locked()
    save_json(ROOT / "results/checkpoint_verification.json", {"scope": "inférence et fidélité aux scores dev sauvegardés, sans nouvel entraînement", "count": len(checks), "checks": checks})
    print(f"{len(checks)} checkpoints rechargés et vérifiés.", flush=True)


if __name__ == "__main__":
    main()
