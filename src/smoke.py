"""Essai court A–E/P avec aller-retour disque ; jamais un résultat de benchmark."""
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import numpy as np
import torch


def smoke_check(device="cpu", real_pretrained=False, report_path=None):
    from . import train, evaluate, models
    from .preprocess import load_split, save_json, TEXT_MODES
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
    original_root = train.ROOT
    frames = {split: load_split(split).groupby("label", group_keys=False).head(2 if split == "train" else 1).copy()
              for split in ("train", "dev")}
    checks = []
    print("SMOKE TEST : sous-ensemble réduit, sorties temporaires, aucun score de benchmark.", flush=True)
    with TemporaryDirectory(prefix="darija_smoke_") as temporary, ExitStack() as stack:
        root = Path(temporary)
        project, output = root / "project", root / "output"
        for split, frame in frames.items():
            target = project / f"data/ary/{split}.tsv"
            target.parent.mkdir(parents=True, exist_ok=True)
            frame[["tweet", "label"]].to_csv(target, sep="\t", index=False)
        auxiliary = project / "data/arq/train.tsv"
        auxiliary.parent.mkdir(parents=True)
        frames["train"][["tweet", "label"]].to_csv(auxiliary, sep="\t", index=False)
        for module in (train, evaluate, models):
            stack.enter_context(patch.object(module, "ROOT", output))
        stack.enter_context(patch.object(train, "PROJECT_ROOT", project))
        stack.enter_context(patch.object(train, "load_split", side_effect=lambda split: frames[split].copy()))
        stack.enter_context(patch.dict(train.os.environ, {"DARIJA_MAX_LENGTH": "16", "DARIJA_EVAL_DEVICE": device}))
        if not real_pretrained:
            from transformers import BertConfig, BertModel, BertForSequenceClassification, BertTokenizerFast
            paths = {}
            for kind, labels in (("base", models.LABELS if hasattr(models, "LABELS") else train.LABELS),
                                 ("nli", ["contradiction", "neutral", "entailment"]), ("camel", train.LABELS)):
                folder = root / kind
                folder.mkdir()
                vocab = folder / "vocab.txt"
                vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ngood\nbad\nordinary\n", encoding="utf-8")
                BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(folder)
                config = BertConfig(vocab_size=8, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                    intermediate_size=32, max_position_embeddings=256,
                                    id2label=dict(enumerate(labels)), label2id={value: i for i, value in enumerate(labels)})
                (BertModel(config) if kind == "base" else BertForSequenceClassification(config)).save_pretrained(folder)
                paths[kind] = folder
            routes = {models.BERT_ID: paths["base"], "SI2M-Lab/DarijaBERT-arabizi": paths["base"],
                      models.NLI_ID: paths["nli"], models.CAMEL_ID: paths["camel"]}
            for factory in (AutoModel, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig):
                original = factory.from_pretrained
                def redirected(identifier, *args, _original=original, **kwargs):
                    if str(identifier) in routes:
                        identifier = str(routes[str(identifier)])
                        kwargs.pop("revision", None)
                    return _original(identifier, *args, **kwargs)
                stack.enter_context(patch.object(factory, "from_pretrained", side_effect=redirected))

        def check(action):
            row = action()
            folder = output / row["checkpoint"]
            loaded = train.load_predictor(folder, device=device)
            prediction = loaded(frames["dev"].text.tolist())
            recorded = __import__("pandas").read_csv(output / f"results/{row['run_id']}_dev_raw.csv")
            expected = recorded[[f"p_{label}" for label in train.LABELS]].to_numpy()
            assert prediction.shape == (len(frames["dev"]), 3)
            assert np.isfinite(prediction).all() and np.allclose(prediction.sum(1), 1, atol=1e-5)
            assert np.allclose(prediction, expected, atol=1e-5), row["run_id"]
            checks.append(row["run_id"])

        check(lambda: train.classical_run("union", preserve_symbols=True))
        for mode in (*TEXT_MODES, "emoji_only", "punctuation_only"):
            check(lambda mode=mode: train.classical_run("union", tag=f"_{mode}", cleaning=mode, family="P", preserve_symbols=True))
        for kind in ("nli", "camel"):
            check(lambda kind=kind: train.pretrained_run(kind, device=device))
        for kind in ("bilstm", "attention", "transformer", "charcnn"):
            check(lambda kind=kind: train.neural_run(kind, "random", epochs=1, max_length=16, device=device))
        check(lambda: train.neural_run("attention", "fasttext", epochs=1, max_length=16, device=device, open_fasttext=True))
        for pooling in ("cls", "mean", "states"):
            check(lambda pooling=pooling: train.frozen_run(pooling, epochs=1, device=device))
        for model_id in (models.BERT_ID, "SI2M-Lab/DarijaBERT-arabizi"):
            check(lambda model_id=model_id: train.finetune_run(model_id, epochs=3, micro_batch=2, max_length=16, device=device))
        check(lambda: train.finetune_run(models.BERT_ID, epochs=3, micro_batch=2, max_length=16, device=device,
                                        auxiliary=frames["train"], auxiliary_weight=0.1))
        check(lambda: train.finetune_run(models.BERT_ID, epochs=3, micro_batch=2, max_length=16, device=device,
                                        cleaning="without_both", family="P", tag="_without_both"))
    report = {"status": "passed", "device": device, "real_pretrained": real_pretrained, "checks": checks,
              "scope": "Petit sous-ensemble officiel ; vérification de fonctionnement, pas performance de benchmark."
                       + (" Encodeurs minuscules aléatoires : poids officiels et mémoire T4 non vérifiés." if not real_pretrained else " Poids officiels utilisés sur de petits lots.")}
    save_json(Path(report_path) if report_path else original_root / "results/smoke_report.json", report)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-pretrained", action="store_true")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    print(json.dumps(smoke_check(args.device, args.real_pretrained), indent=2, ensure_ascii=False))
