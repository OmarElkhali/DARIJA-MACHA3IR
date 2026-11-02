"""Contrôles sans entraînement, téléchargement ou exécution de notebook."""
import ast
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    # Toute écriture de contrôle est confinée à un dossier temporaire.
    with TemporaryDirectory(prefix="darija_campaign_check_") as temporary:
        os.environ["DARIJA_OUTPUT_ROOT"] = temporary
        os.environ["DARIJA_CLEANING"] = "conservative_v2"
        os.environ["DARIJA_MAX_LENGTH"] = "128"
        import nbformat
        import numpy as np
        import pandas as pd
        from src import campaign, evaluate, train, models
        from src.preprocess import ROOT, clean_text, load_split, save_json

        for path in [*PROJECT.glob("src/*.py"), PROJECT / "scripts/build_notebook.py"]:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        notebook = nbformat.read(PROJECT / "notebooks/TP2_Darija.ipynb", as_version=4)
        nbformat.validate(notebook)
        code = [c for c in notebook.cells if c.cell_type == "code"]
        assert len(code) == 49
        for i, cell in enumerate(code):
            compile(cell.source, f"notebook_cell_{i}", "exec")
            assert cell.execution_count is None and not cell.outputs
        # Exécuter uniquement la détection de chemin, pas la cellule de configuration.
        finder_node = next(n for n in ast.parse((PROJECT / "scripts/notebook_bootstrap.py").read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef) and n.name == "find_project")
        namespace = {"Path": Path}
        exec(compile(ast.Module(body=[finder_node], type_ignores=[]), "project_finder", "exec"), namespace)
        find_project = namespace["find_project"]
        cloud = ROOT / "fake_cloud"
        orphan = ROOT / "notebook_only"
        orphan.mkdir()
        def fake_project(path):
            for name in ("src/preprocess.py", "src/train.py", "src/campaign.py", "src/resources.py", "requirements.txt", "requirements-notebook.txt"):
                target = path / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("", encoding="utf-8")
            return path.resolve()
        with patch.object(Path, "cwd", return_value=orphan):
            try:
                find_project(search_roots=[cloud])
            except FileNotFoundError as error:
                assert "notebook seul" in str(error)
            else:
                raise AssertionError("Projet absent accepté")
            actual = fake_project(cloud / "dataset" / "Darija-Sentiment")
            assert find_project(search_roots=[cloud]) == actual
            assert find_project(str(actual)) == actual
            other = fake_project(cloud / "other" / "Darija-Sentiment")
            try:
                find_project(search_roots=[cloud])
            except RuntimeError as error:
                assert "Plusieurs" in str(error)
            else:
                raise AssertionError("Projet ambigu choisi silencieusement")
            assert find_project(str(other)) == other
            try:
                find_project(str(orphan))
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("Chemin explicite invalide accepté")
        assert clean_text("@user ma3jbnich 😡!!! https://t.co/x") == "ma3jbnich 😡!!!"
        assert "label" not in load_split("test")
        # D2 : cache en mémoire mappée, écrit complètement avant publication.
        class FakeEncoder:
            calls = 0
            def extract(self, texts, pooling, max_length, output_path, **unused):
                self.calls += 1
                values = np.lib.format.open_memmap(output_path, mode="w+", dtype="float16", shape=(len(texts), max_length, 2))
                values[:] = 0.5
                return values, np.ones((len(texts), max_length), dtype="int64")
        encoder = FakeEncoder()
        with patch.object(train, "load_split", return_value=pd.DataFrame({"text": ["one", "two"]})):
            values, mask = train.frozen_features(encoder, "train", "states")
            assert isinstance(values, np.memmap) and values.shape == (2, 128, 2)
            tensor_view = train.torch.from_numpy(values)
            assert tensor_view.data_ptr() == values.ctypes.data
            del tensor_view
            values._mmap.close()
            cached, mask = train.frozen_features(None, "train", "states")
            assert encoder.calls == 1 and cached[0, 0, 0] == 0.5
            cached._mmap.close()
        # Même avec un traceback conservé, une extraction interrompue libère son mmap.
        from types import SimpleNamespace
        class Batch(dict):
            def to(self, device):
                return self
            def __getattr__(self, key):
                return self[key]
        class Tokenizer:
            def __call__(self, texts, max_length, **unused):
                return Batch(input_ids=train.torch.ones((len(texts), max_length), dtype=train.torch.long), attention_mask=train.torch.ones((len(texts), max_length), dtype=train.torch.long))
        class FailingEncoder:
            config = SimpleNamespace(hidden_size=2)
            calls = 0
            def __call__(self, input_ids, attention_mask):
                self.calls += 1
                if self.calls == 2:
                    raise KeyboardInterrupt()
                return SimpleNamespace(last_hidden_state=train.torch.ones((*input_ids.shape, 2)))
        extractor = models.FrozenEncoder.__new__(models.FrozenEncoder)
        extractor.device, extractor.tokenizer, extractor.model = "cpu", Tokenizer(), FailingEncoder()
        partial_cache = ROOT / "interrupted.npy.part"
        try:
            extractor.extract(["one", "two"], "states", batch_size=1, max_length=3, output_path=partial_cache)
        except KeyboardInterrupt as error:
            held_traceback = error
        else:
            raise AssertionError("Interruption non propagée")
        partial_cache.unlink()
        del held_traceback
        progress = ROOT / "models/A_interrupted_s42"
        progress.mkdir(parents=True)
        (progress / "meta.json").write_text("{}", encoding="utf-8")
        (progress / ".in_progress").touch()
        assert campaign.complete("A_interrupted_s42") is False
        try:
            load_split("test", labels=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Labels test ouverts sans verrou")
        config = campaign.default_config()
        config["device"] = "cpu"
        campaign.prepare(config, dry_run=True)
        assert not (ROOT / "campaign.json").exists()
        campaign.prepare(config)
        # Publication d'un run sans modèle : faux scores et checkpoint opaque, uniquement dans le dossier temporaire.
        folder = train.run_folder("A_dummy_s42")
        assert (folder / ".in_progress").exists()
        (folder / "classifier.joblib").write_bytes(b"test fixture, not a model")
        dummy_dev = pd.DataFrame({"row_id": ["d1", "d2", "d3"], "script": ["latin"] * 3, "label": ["negative", "neutral", "positive"]})
        train.finish_run(folder, "A", "dummy", 42, {}, 0.0, 0, np.eye(3), dummy_dev, "classical")
        assert not (folder / ".in_progress").exists() and campaign.complete("A_dummy_s42")
        (folder / ".in_progress").touch()
        assert campaign.complete("A_dummy_s42") is False
        (folder / ".in_progress").unlink()
        expected = {"A": 8, "B": 4, "C": 39, "D": 5, "E1": 6, "P": 15}
        names = []
        for part, count in expected.items():
            planned = campaign.jobs(part, config)
            assert len(planned) == count, (part, len(planned))
            names += [job[0] for job in planned]
        assert len(names) == len(set(names))
        # Une configuration modifiée ne peut reprendre la même campagne.
        try:
            campaign.prepare({**config, "max_length": 64})
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("Configuration mutable")
        try:
            campaign.finalize(config)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Test ouvert malgré des runs manquants")
        assert not (ROOT / "results/selection_lock.json").exists()
        save_json(ROOT / "results/c_imbalance_selection.json", {"variant": "transformer_fasttext_nopos", "config": {"kind": "transformer", "embedding": "fasttext", "positional": False, "max_length": 128, "open_fasttext": True}})
        balance = campaign.jobs("C_balance", config)
        assert len(balance) == 6 and all("transformer_fasttext_nopos" in job[0] for job in balance)
        assert all(not job[3]["positional"] and job[3]["open_fasttext"] for job in balance)
        save_json(ROOT / "results/e2_encoder_selection.json", {"variant": "arabizi", "model_id": "SI2M-Lab/DarijaBERT-arabizi"})
        e2 = campaign.jobs("E2", config)
        assert len(e2) == 9
        assert {job[3]["auxiliary_weight"] for job in e2} == {0.1, 0.3, 0.7}
        assert all(job[2][0].endswith("arabizi") for job in e2)
        # arq : uniquement train, sans collision avec les textes ary ni conflit conservé.
        import hashlib
        auxiliary_root = ROOT / "auxiliary_test"
        path = auxiliary_root / "data/arq/train.tsv"
        path.parent.mkdir(parents=True)
        raw = "tweet\tlabel\ngood\tpositive\nbad\tnegative\nordinary\tneutral\ncollision\tpositive\nambiguous\tnegative\nambiguous\tpositive\n"
        path.write_text(raw, encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        save_json(path.with_name("provenance.json"), {"commit": campaign.COMMIT, "sha256": digest})
        safe_frame = pd.DataFrame({"text": ["collision"]})
        with patch.object(campaign, "PROJECT_ROOT", auxiliary_root), patch.object(campaign, "ARQ_TRAIN_SHA256", digest), patch.object(campaign, "load_split", return_value=safe_frame) as reader:
            auxiliary = campaign.auxiliary_train()
            assert set(auxiliary.label) == {"negative", "neutral", "positive"} and len(auxiliary) == 3
            assert all(call.kwargs.get("labels", False) is False for call in reader.call_args_list)
            assert not (auxiliary_root / "data/arq/test.tsv").exists()
        # Aucun appel des fonctions d'apprentissage : vérifier leurs réglages par AST.
        function = next(n for n in ast.parse((PROJECT / "src/train.py").read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef) and n.name == "finetune_run")
        assert any(isinstance(n, ast.Constant) and n.value == "weighted_loss" for n in ast.walk(function))
        assert "MultiTaskSentiment" in ast.unparse(function)
        # Injection de vecteurs artificiels : pas d'entraînement fastText.
        class FakeVectors:
            def __getitem__(self, word):
                return np.ones(128, dtype="float32")
        with patch.object(models, "fasttext_vectors", return_value=FakeVectors()):
            ids, vectors = models.word_inputs(["known rare"], {"<PAD>": 0, "<UNK>": 1, "known": 2}, 4, "fake")
        assert ids.tolist() == [[2, 1, 0, 0]]
        assert not vectors[0, 0].any() and vectors[0, 1].sum().item() == 128
        frame = pd.DataFrame({"label": ["positive", "negative"], "text": ["zwin 😍!!!", "khayb"], "row_id": ["a", "b"], "script": ["latin", "latin"]})
        probabilities = np.array([[0.05, 0.05, 0.9], [0.9, 0.05, 0.05]])
        predictor = lambda texts: np.tile([0.9, 0.05, 0.05], (len(texts), 1))
        robustness = evaluate.robustness_f4(frame, probabilities, predictor, [0, 0, 0])
        assert robustness["eligible"] == 2 and robustness["changed"] == 1
        assert robustness["accuracy_drop"] == 0.5 and robustness["weighted_f1_drop"] > 0
        # Une revue humaine existante ne doit pas être effacée par F6.
        save_json(ROOT / "results/selection_lock.json", {"artifacts": {}, "candidate": {"family": "A", "variant": "char"}})
        with patch.object(evaluate, "annotation_audit", side_effect=lambda x: x):
            errors = evaluate.error_review(frame, np.array([[0.9, 0.05, 0.05], [0.05, 0.05, 0.9]]), [0, 0, 0])
            errors.loc[:, "validation_humaine"] = "validé"
            errors.loc[:, "categorie_humaine"] = "ironie"
            errors.to_csv(ROOT / "results/f6_30_erreurs.csv", index=False)
            repeated = evaluate.error_review(frame, np.array([[0.9, 0.05, 0.05], [0.05, 0.05, 0.9]]), [0, 0, 0])
            assert repeated.validation_humaine.eq("validé").all()
        perfect = evaluate.error_review(frame, np.array([[0.05, 0.05, 0.9], [0.9, 0.05, 0.05]]), [0, 0, 0])
        assert perfect.empty and "validation_humaine" in pd.read_csv(ROOT / "results/f6_30_erreurs.csv")
        # Le balayage post hoc ne doit changer ni le checkpoint ni ses décalages.
        meta = ROOT / "models/A_char_s42/meta.json"
        save_json(meta, {"offsets": [0.0, 0.3, 0.0]})
        before = meta.read_bytes()
        fake = pd.DataFrame({"label": ["positive", "negative"], "p_negative": [0.1, 0.8], "p_neutral": [0.2, 0.1], "p_positive": [0.7, 0.1]})
        for split in ("dev", "test"):
            fake.to_csv(ROOT / f"results/A_char_s42_{split}_raw.csv", index=False)
        grid = campaign.threshold_diagnostic()
        assert len(grid) == 441 and meta.read_bytes() == before
        # F/G sur prédictions artificielles : tableau détaillé, résumé et candidat distincts.
        fake_frame = pd.DataFrame({"row_id": ["x1", "x2", "x3"], "script": ["latin"] * 3,
                                   "text": ["positive text", "negative text", "ordinary text"],
                                   "label": ["positive", "negative", "neutral"]})
        probabilities = np.tile([0.1, 0.8, 0.1], (3, 1))
        reports = [{"family": f, "variant": v, "dev_weighted_f1": score} for f, v, score in [("A", "char", 0.8), ("A", "word", 0.7), ("B", "camelbert", 0.6)]]
        for item in reports:
            run = f"{item['family']}_{item['variant']}_s42"
            metadata = {"run_id": run, "family": item["family"], "variant": item["variant"], "seed": 42,
                        "kind": "classical", "purpose": "benchmark", "offsets": [0, 0, 0], "config": {},
                        "checkpoint": f"models/{run}", "train_seconds": 0.0, "trainable_parameters": 0}
            save_json(ROOT / f"models/{run}/meta.json", metadata)
            evaluate.record_metrics(metadata, fake_frame, probabilities, "dev", raw=True)
            evaluate.record_metrics(metadata, fake_frame, probabilities, "dev")
        test_lock = {"artifacts": {}, "candidate": {"family": "A", "variant": "char"},
                     "selected_variants": [reports[0], reports[2]], "report_variants": reports,
                     "test_runs": [f"{r['family']}_{r['variant']}_s42" for r in reports],
                     "mcnemar_runs": ["A_char_s42", "A_word_s42"]}
        save_json(ROOT / "results/selection_lock.json", test_lock)
        def read_fake(split, **kwargs):
            return fake_frame.assign(text="train " + fake_frame.text) if split == "train" else fake_frame.copy()
        fake_predictor = lambda texts: np.tile([0.1, 0.8, 0.1], (len(texts), 1))
        with patch.object(evaluate, "load_split", side_effect=read_fake), patch.object(train, "predict_split", return_value=probabilities), patch.object(train, "load_predictor", return_value=fake_predictor), patch.object(evaluate, "bootstrap_ci", return_value={"ci_low": 0.0, "ci_high": 1.0, "bootstrap": 1000}), patch.object(evaluate, "error_review") as review_call, patch.object(evaluate, "save_confusion") as confusion:
            summary = evaluate.final_evaluation()
            assert len(summary) == 2
            assert len(pd.read_csv(ROOT / "results/comparison_all_variants.csv")) == 3
            assert review_call.call_count == 1
            assert sum(call.args[-1] == "modele_final" for call in confusion.call_args_list) == 1
        mcnemar_result = json.loads((ROOT / "results/f5_mcnemar.json").read_text(encoding="utf-8"))
        assert mcnemar_result["second"] == "A_word_s42"
        print("OK : 49 cellules non exécutées, plans A–G/P, garde-fous, fastText OOV, F4, F6 et diagnostic F1.")
        print("Aucun entraînement, téléchargement ou modèle préentraîné exécuté.")


if __name__ == "__main__":
    main()
