"""Régressions réelles CPU : reprise exacte, cache incomplet, nettoyages et revue humaine."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, Mock
import copy
import os
import sys
import runpy
import shutil
from types import ModuleType, SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    with TemporaryDirectory(prefix="darija_improvements_") as temporary:
        os.environ.update(DARIJA_OUTPUT_ROOT=temporary, DARIJA_CLEANING="conservative_v2", DARIJA_MAX_LENGTH="16")
        import numpy as np
        import pandas as pd
        import torch
        from src import train, evaluate, campaign, validation
        from src.preprocess import clean_text, emoji_parts, save_json, ROOT
        project_root = Path(__file__).resolve().parents[1]
        # Sauvegarde Colab simulée sur disque réel, sans accès au compte Google.
        helpers = runpy.run_path(str(project_root / "scripts/notebook_bootstrap.py"))
        content = ROOT / "content"
        (content / "drive/MyDrive").mkdir(parents=True)
        old = content / "Darija-Sentiment-runtime-old"
        old_run = old / "experiments/campaign_v3"
        (old_run / "models/E_darijabert_s42").mkdir(parents=True)
        save_json(old_run / "campaign.json", {"campaign": "v3"})
        weight = old_run / "models/E_darijabert_s42/weights.pt"
        weight.write_bytes(b"old checkpoint fixture")
        colab = ModuleType("google.colab")
        colab.drive = SimpleNamespace(mount=lambda _: (_ for _ in ()).throw(AssertionError("Drive déjà monté")))
        def sandbox_path(value):
            value = str(value)
            return content / value.removeprefix("/content").lstrip("/") if value.startswith("/content") else Path(value)
        storage = helpers["prepare_storage"]
        with patch.dict(storage.__globals__, {"Path": sandbox_path, "PROJECT_BACKUP_SCRIPT": project_root / "scripts/backup_results.py"}), patch.dict(sys.modules, {"google.colab": colab}), patch.dict(os.environ, {"COLAB_RELEASE_TAG": "test"}):
            output = storage("campaign_v4")
            assert "campaign_v4" in output
            manifests = list((content / "drive/MyDrive/Darija-Sentiment/archives").glob("*/snapshot_manifest.json"))
            assert len(manifests) == 1
            storage("campaign_v4")
            assert len(list(manifests[0].parent.parent.glob("*/snapshot_manifest.json"))) == 1
            weight.write_bytes(b"new checkpoint fixture")
            storage("campaign_v4")
            assert len(list(manifests[0].parent.parent.glob("*/snapshot_manifest.json"))) == 2
            assert (manifests[0].parent / weight.relative_to(old)).read_bytes() == b"old checkpoint fixture"
        import csv
        for path in (Path(__file__).resolve().parents[1] / "data/review").glob("*.csv"):
            assert all(None not in row and all(v is not None for v in row.values()) for row in csv.DictReader(path.open(encoding="utf-8")))
        text = "ma3jbnich ما عجبنيش 👩🏽‍💻 🇲🇦 1️⃣ :) !!! 3 7 9"
        assert len(emoji_parts(text)) == 4
        cleaned = clean_text(text, "without_both")
        assert not emoji_parts(cleaned) and "ما عجبنيش" in cleaned and "3 7 9" in cleaned
        assert "1️⃣" in clean_text(text, "without_punctuation")
        assert clean_text("bad😡good", "without_emojis") == "bad good"
        assert clean_text("ma3jbnich 😡!!!", "emoji_only") == "😡"
        assert clean_text("ma3jbnich 😡!!!", "punctuation_only") == "! ! !"
        try:
            evaluate.select_on_dev()
        except RuntimeError as error:
            assert "Aucun résultat" in str(error)
        else:
            raise AssertionError("Sélection vide acceptée")

        class Encoder:
            calls = 0
            def extract(self, texts, pooling, max_length, output_path, **kwargs):
                self.calls += 1
                array = np.lib.format.open_memmap(output_path, mode="w+", dtype="float16", shape=(len(texts), max_length, 2))
                array[:] = 1
                return array, np.ones((len(texts), max_length), dtype="int64")
        cache = train.feature_cache_path("train", "states")
        np.save(cache, np.zeros((2, 16, 2), dtype="float16"))
        assert not train.valid_feature_cache("train", "states")
        encoder = Encoder()
        with patch.object(train, "load_split", return_value=pd.DataFrame({"text": ["a", "b"]})):
            array, mask = train.frozen_features(encoder, "train", "states")
            array._mmap.close()
            assert encoder.calls == 1 and train.valid_feature_cache("train", "states")
            np.save(cache.with_suffix(".mask.npy"), np.ones((1, 3)))
            array, mask = train.frozen_features(encoder, "train", "states")
            array._mmap.close()
            assert encoder.calls == 2 and mask.shape == (2, 16)
        folder = ROOT / "models/D_states_s42"
        save_json(folder / "meta.json", {"kind": "frozen", "config": {"pooling": "states", "model_id": "fixture", "hidden_size": 2}})
        head = Mock()
        with patch.dict(os.environ, {"DARIJA_EVAL_DEVICE": "cuda"}), patch.object(train, "valid_feature_cache", return_value=False), patch.object(train, "FrozenEncoder") as factory, patch.object(train, "frozen_features", return_value=(np.ones((2, 16, 2), dtype="float16"), np.ones((2, 16), dtype="int64"))), patch.object(train, "FrozenSequence", return_value=head), patch.object(torch, "load", return_value={}), patch.object(train, "predict_tensor", return_value=np.ones((2, 3)) / 3) as predict:
            train.predict_split(folder, pd.DataFrame(), "test")
            factory.assert_called_once_with("fixture", device="cuda")
            head.to.assert_called_once_with("cuda")
            assert predict.call_args.args[-1] == "cuda"

        # Même initialisation et mêmes lots : interrompre l'époque 2 et reprendre.
        config = dict(device="cpu", micro_batch=2, effective_batch=4, epochs=3, lr=0.01,
                      patience=99, weighted_loss=True, warmup_ratio=0.1)
        x = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 24
        y = np.array([0, 1, 2, 0, 1, 2])
        make_model = lambda: torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(8, 3))
        train.seed_everything(42)
        baseline, expected_history = train.fit_tensor(make_model(), (x,), y, (x,), y, config, checkpoint_path=ROOT / "full.pt")
        expected_state = torch.load(ROOT / "full.pt", weights_only=True)["model"]
        train.seed_everything(42)
        original_predict = train.predict_tensor
        calls = 0
        def interrupt(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt("interruption volontaire")
            return original_predict(*args, **kwargs)
        with patch.object(train, "predict_tensor", side_effect=interrupt):
            try:
                train.fit_tensor(make_model(), (x,), y, (x,), y, config, checkpoint_path=ROOT / "resume.pt")
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("Interruption non déclenchée")
        assert torch.load(ROOT / "resume.pt", weights_only=True)["epoch"] == 1
        train.seed_everything(123)
        resumed, actual_history = train.fit_tensor(make_model(), (x,), y, (x,), y, config, checkpoint_path=ROOT / "resume.pt")
        actual_state = torch.load(ROOT / "resume.pt", weights_only=True)["model"]
        assert all(torch.equal(value, actual_state[key]) for key, value in expected_state.items())
        assert [h["loss"] for h in expected_history] == [h["loss"] for h in actual_history]
        assert all(torch.equal(value, resumed.state_dict()[key]) for key, value in baseline.state_dict().items())
        try:
            train.fit_tensor(make_model(), (x + 1,), y, (x,), y, config, checkpoint_path=ROOT / "resume.pt")
        except RuntimeError as error:
            assert "incompatible" in str(error)
        else:
            raise AssertionError("Checkpoint d'autres données accepté")

        (ROOT / "results").mkdir(exist_ok=True)
        frame = pd.DataFrame({"row_id": ["a", "b", "c"], "text": ["good 😡!!!", "bad", "normal"],
                              "label": ["positive", "negative", "neutral"], "script": ["latin"] * 3})
        probabilities = np.array([[0, 0, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        predictor = lambda texts: np.tile([1., 0., 0.], (len(texts), 1))
        stats = evaluate.robustness_f4(frame, probabilities, predictor, [0, 0, 0])
        detail = pd.read_csv(ROOT / "results/f4_perturbations.csv")
        assert stats["eligible"] == 3 and detail.altered_prediction.notna().all()
        assert stats["accuracy_before"] == 2 / 3
        save_json(ROOT / "results/selection_lock.json", {"artifacts": {}})
        validation.robustness_review(frame, predictor, [0, 0, 0])
        path = ROOT / "results/f4_human_review.csv"
        review = pd.read_csv(path, keep_default_na=False)
        review.loc[0, ["human_label", "reviewer", "validation_humaine", "sentiment_preserved"]] = ["negative", "reviewer1", "validé", "non"]
        review.to_csv(path, index=False)
        report = validation.robustness_review(frame, predictor, [0, 0, 0])
        assert report.n_reviewed.sum() == 1
        assert pd.read_csv(path).reviewer.eq("reviewer1").sum() == 1
        near = validation.near_duplicates(frame, frame.assign(row_id=lambda f: "new:" + f.row_id))
        assert len(near) == 3
        review_dir = ROOT / "human_review"
        shutil.copytree(project_root / "data/review", review_dir)
        pending = validation.evaluate_human_sets(predictor, [0, 0, 0])
        assert pending.n.eq(0).all()  # Les suggestions des exemples ne sont jamais des labels.
        cases = pd.read_csv(review_dir / "behavioral_cases.csv", keep_default_na=False)
        cases.loc[0, ["label", "reviewer", "validation_humaine"]] = ["positive", "reviewer1", "validé"]
        cases.to_csv(review_dir / "behavioral_cases.csv", index=False)
        reviewed = validation.evaluate_human_sets(predictor, [0, 0, 0])
        assert reviewed.loc[reviewed["set"].eq("behavioral_cases") & reviewed.get("category").isna(), "n"].iloc[0] == 1
        pd.DataFrame([{"text": frame.text.iloc[0], "label": "positive", "reviewer": "reviewer1",
                       "validation_humaine": "validé", "source": "fixture", "collected_at": "2026-11-01"}]).to_csv(review_dir / "independent.csv", index=False)
        with patch.object(validation, "load_split", return_value=frame):
            try:
                validation.evaluate_human_sets(predictor, [0, 0, 0])
            except ValueError as error:
                assert "proches du benchmark" in str(error)
            else:
                raise AssertionError("Faux ensemble indépendant accepté")
        campaign.prepare({**campaign.default_config(), "max_length": 16}, dry_run=True)
        assert len(campaign.jobs("A", campaign.default_config())) == 8
        assert len(campaign.jobs("D", campaign.default_config())) == 5
        assert len(campaign.jobs("P", campaign.default_config())) == 15
        print("OK : reprise CPU identique après interruption, refus de données différentes, réparation cache, nettoyages, F4 commun et annotations conservées.")


if __name__ == "__main__":
    main()
