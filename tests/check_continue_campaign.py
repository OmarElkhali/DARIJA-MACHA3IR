"""Cache invalidé sur modification et cumul sans doublon de lignes d'évaluation."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import json
import runpy
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    helper = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/continue_campaign.py"))
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        helper["install_storage_writes"]()
        from src.preprocess import save_json
        target = root / "concurrent.json"
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda index: save_json(target, {"value": index}), range(40)))
        assert json.loads(target.read_text())["value"] in range(40)
        before = target.stat().st_mtime_ns
        save_json(target, json.loads(target.read_text()))
        assert target.stat().st_mtime_ns == before and not list(root.glob("*.tmp"))
        folder = root / "models/E_test_s42"
        folder.mkdir(parents=True)
        (root / "results").mkdir()
        (folder / "weights.pt").write_bytes(b"first")
        (folder / "artifact_sha256.json").write_text(json.dumps({"weights.pt": "signature"}))
        (folder / "meta.json").write_text(json.dumps({"train_seconds": 150, "config": {"resumed_epoch_seconds": 60}}))
        calls = []
        campaign = SimpleNamespace(ROOT=root, complete=lambda run: calls.append(run) or True)
        helper["install_resume_checks"](campaign)
        assert campaign.complete("E_test_s42") and campaign.complete("E_test_s42") and len(calls) == 1
        from src.preprocess import sha256
        first_hash = sha256(folder / "weights.pt")
        assert sha256(folder / "weights.pt") == first_hash
        (folder / "weights.pt").write_bytes(b"modified")
        assert sha256(folder / "weights.pt") != first_hash
        assert campaign.complete("E_test_s42") and len(calls) == 2
        (folder / ".in_progress").touch()
        assert campaign.complete("E_test_s42") and len(calls) == 3
        rows = [dict(run_id="E_test_s42", family="E", variant="test", split=split, decision=decision)
                for split, decision in [("dev", "raw"), ("dev", "dev_offsets"), ("test", "dev_offsets")]]
        times = helper["training_times"](root, pd.DataFrame(rows))
        assert len(times) == 1
        assert times.iloc[0].avant_reprise_s == 60 and times.iloc[0].derniere_session_s == 90
        assert times.total_entrainement_s.sum() == 150
        print("OK : vérification réutilisée puis invalidée, 60 + 90 = 150 s sans doublons.")


if __name__ == "__main__":
    main()
