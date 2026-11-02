"""Un pas de gradient sur deux vrais tweets train pour chaque checkpoint E.

Ce contrôle technique n'est pas le protocole E (3 époques / batch effectif 32).
Il ne sauvegarde aucun checkpoint et aucune métrique de benchmark.
"""
import os
os.environ["USE_TF"] = "0"
import sys
import gc
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModel, AutoTokenizer
from src.models import BERT_ID, MODEL_REVISIONS, FineTunedSentiment
from src.preprocess import ROOT, LABELS, load_split, save_json
from src.train import seed_everything


def main():
    seed_everything(42)
    frame = load_split("train").iloc[:2]
    labels = torch.tensor([LABELS.index(x) for x in frame.label])
    rows = []
    for model_id in (BERT_ID, "SI2M-Lab/DarijaBERT-arabizi"):
        start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=MODEL_REVISIONS[model_id], local_files_only=True)
        model = FineTunedSentiment(AutoModel.from_pretrained(model_id, revision=MODEL_REVISIONS[model_id], add_pooling_layer=False, local_files_only=True))
        inputs = tokenizer(frame.text.tolist(), padding=True, truncation=True, max_length=32, return_tensors="pt")
        logits = model(**inputs)
        assert logits.shape == (2, 3)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.encoder.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        optimizer.step()
        rows.append({"model_id": model_id, "test": "one_forward_backward_adamw_step", "n_train_examples": 2, "shape": [2, 3], "seconds": time.perf_counter() - start, "benchmark": False, "checkpoint_saved": False})
        print(rows[-1], flush=True)
        del model, optimizer, inputs, logits, loss
        gc.collect()
    save_json(ROOT / "results/pretrained_smoke.json", rows)


if __name__ == "__main__":
    main()
