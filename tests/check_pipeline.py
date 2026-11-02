"""Vérifications de protocole et d'architectures ; aucun score de benchmark simulé."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from transformers import BertConfig, BertModel
from src.preprocess import ROOT, load_split, build_vocab, encode_texts, clean_text, script_type, LABELS
from src.models import TextNetwork, FrozenSequence, FineTunedSentiment, embedding_matrix
from src.evaluate import decide, tune_offsets, metric_scores, mcnemar
from src.train import seed_everything, fit_tensor


def main():
    seed_everything(42)
    frame = load_split("train").groupby("label", group_keys=False).head(4).reset_index(drop=True)
    assert clean_text("ma3jbnich 😡!!! https://t.co/x") == "ma3jbnich 😡!!! URL"
    assert script_type("@user زوين https://exemple.ma") == "arabic"
    assert script_type("زوين bzaf") == "mixed"
    assert "label" not in load_split("test").columns
    vocab = build_vocab(frame.text)
    ids = torch.tensor(encode_texts(frame.text.tolist(), vocab, 16))
    labels = torch.tensor([LABELS.index(x) for x in frame.label])
    for kind in ("bilstm", "attention", "transformer", "charcnn"):
        model = TextNetwork(len(vocab), kind)
        output = model(ids)
        assert output.shape == (len(frame), 3) and torch.isfinite(output).all()
        loss = torch.nn.functional.cross_entropy(output, labels)
        loss.backward()
        assert model.embedding.weight.grad is not None
        model.eval()
        if kind == "attention":
            _, weights = model(ids, return_attention=True)
            assert torch.allclose(weights.sum(1), torch.ones(len(frame)))
            assert weights[ids == 0].abs().max() < 1e-5
        print(kind, tuple(output.shape), "forward/backward OK")
    matrix, coverage = embedding_matrix(vocab, frame.text, "fasttext")
    assert matrix.shape == (len(vocab), 128) and np.isfinite(matrix).all()
    # Modèle BERT minuscule aléatoire : test de formes seulement, jamais enregistré comme résultat.
    bert = BertModel(BertConfig(vocab_size=100, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32))
    tuned = FineTunedSentiment(bert)
    fake_ids = ids.clamp(max=99)
    assert tuned(fake_ids, fake_ids.ne(0)).shape == (len(frame), 3)
    head = FrozenSequence(16)
    with torch.no_grad():
        states = bert(fake_ids, attention_mask=fake_ids.ne(0)).last_hidden_state
    out, weights = head(states, fake_ids.ne(0), return_attention=True)
    torch.nn.functional.cross_entropy(out, labels).backward()
    assert not any(p.grad is not None for p in bert.parameters())
    assert weights[fake_ids == 0].abs().max() < 1e-5
    truth = np.array(LABELS)[labels]
    scores = np.eye(3)[labels]
    assert metric_scores(truth, decide(scores))["weighted_f1"] == 1.0
    assert tune_offsets(truth, scores) == [0.0, 0.0, 0.0]
    assert mcnemar(truth, truth, truth)["p_exact"] == 1.0
    for invalid in ([.2, .3, .5], [[.2, .8]], [[float("nan"), 0, 1]], [[-.1, .2, .9]], [[.1, .1, .1]]):
        try:
            decide(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Scores invalides acceptés : {invalid}")
    try:
        decide([[0, 0, 1]], [0, float("inf"), 0])
    except ValueError:
        pass
    else:
        raise AssertionError("Décalages non finis acceptés")
    config = dict(device="cpu", micro_batch=2, effective_batch=4, epochs=1, lr=1e-3)
    _, history = fit_tensor(TextNetwork(len(vocab), "attention"), (ids,), labels.numpy(), (ids,), labels.numpy(), config)
    assert len(history) == 1 and np.isfinite(history[0]["loss"])
    _, tuned_history = fit_tensor(tuned, (fake_ids, fake_ids.ne(0)), labels.numpy(), (fake_ids, fake_ids.ne(0)), labels.numpy(), config)
    assert len(tuned_history) == 1 and np.isfinite(tuned_history[0]["loss"])
    print("Protocole, FastText, encodeur/tête, accumulation et masques OK. Ces vérifications ne sont pas des expériences scientifiques.")


if __name__ == "__main__":
    main()
