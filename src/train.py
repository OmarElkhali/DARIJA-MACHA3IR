"""Entraînements ary ; corpus arq auxiliaire uniquement pour E2, checkpoints réels."""
from pathlib import Path
import argparse
import copy
import json
import math
import os
import random
import time
import hashlib
from zipfile import BadZipFile

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.linear_model import LogisticRegression

from .preprocess import ROOT, PROJECT_ROOT, CLEANING, LABELS, load_split, build_vocab, encode_texts, save_json, sha256, transform_frame, clean_text
from .models import (classical_model, TextNetwork, FrozenSequence, FrozenEncoder,
                     FineTunedSentiment, MultiTaskSentiment, PretrainedSentiment, embedding_matrix, word_inputs,
                     BERT_ID, MODEL_REVISIONS)
from .evaluate import tune_offsets, record_metrics, metric_scores, decide, results_table

SEEDS = (42, 43, 44)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True, warn_only=True)


def runtime_info():
    import platform
    import psutil
    from importlib.metadata import version, PackageNotFoundError
    def installed(package):
        try:
            return version(package)
        except PackageNotFoundError:
            return "non installé (facultatif pour le notebook)"
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.cuda.is_available(), "cpu_threads": os.cpu_count(),
            "ram_gb": psutil.virtual_memory().total / 2**30, "available_ram_gb": psutil.virtual_memory().available / 2**30,
            "packages": {p: installed(p) for p in ("numpy", "pandas", "scikit-learn", "scipy", "torch", "transformers", "gensim", "streamlit")}}


def run_folder(run_id):
    if (ROOT / "results/selection_lock.json").exists():
        raise RuntimeError("étude verrouillée : aucun nouveau réglage. Pour une nouvelle campagne, copier le projet sans results/ et models/ et déclarer une nouvelle étude ; le test déjà observé ne redevient pas indépendant.")
    folder = ROOT / "models" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    if ROOT != PROJECT_ROOT:
        (folder / ".in_progress").touch()
    (ROOT / "results/figures").mkdir(parents=True, exist_ok=True)
    return folder


def finish_run(folder, family, variant, seed, config, elapsed, parameters, probabilities, dev, kind, purpose="benchmark", history=None):
    offsets = tune_offsets(dev.label, probabilities)
    meta = {"run_id": folder.name, "family": family, "variant": variant, "seed": seed, "kind": kind, "purpose": purpose,
            "config": config, "offsets": offsets, "train_seconds": elapsed, "trainable_parameters": int(parameters),
            "checkpoint": str(folder.relative_to(ROOT)), "labels": LABELS, "data_sha256": {s: sha256(PROJECT_ROOT / f"data/ary/{s}.tsv") for s in ("train", "dev")},
            "created_utc": pd.Timestamp.now(tz="UTC").isoformat(), "device": "cuda" if config.get("device") == "cuda" else "cpu"}
    if history is not None:
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
    record_metrics(meta, dev, probabilities, "dev", raw=True)
    row = record_metrics(meta, dev, probabilities, "dev")
    save_json(folder / "meta.json", meta)
    if ROOT != PROJECT_ROOT:
        save_json(folder / "artifact_sha256.json", {str(p.relative_to(folder)): sha256(p) for p in folder.rglob("*") if p.is_file() and p.name not in {"artifact_sha256.json", ".in_progress", "resume.pt"} and not p.name.endswith(".part")})
        (folder / ".in_progress").unlink(missing_ok=True)
        (folder / "resume.pt").unlink(missing_ok=True)  # Le checkpoint final remplace la reprise d'entraînement.
    print(f"{folder.name}: dev F1={row['weighted_f1']:.4f}, macro={row['macro_f1']:.4f}, {elapsed:.1f}s", flush=True)
    return row


def save_weights(model, folder):
    target = folder / "weights.pt"
    temporary = target.with_suffix(".pt.part")
    torch.save(model.cpu().state_dict(), temporary)
    temporary.replace(target)


def classical_run(kind="union", seed=42, train_script=None, C=1.0, preserve_symbols=False, tag="", cleaning=None, family="A", purpose="benchmark"):
    seed_everything(seed)
    variant = (kind if train_script is None else f"{kind}_train_{train_script}") + tag
    run_id = f"{family}_{variant}_s{seed}"
    folder = run_folder(run_id)
    train, dev = load_split("train"), load_split("dev")
    cleaning = cleaning or CLEANING
    train, dev = [transform_frame(frame, cleaning) for frame in (train, dev)]
    if train_script is not None:
        train = train.loc[train.script == train_script]
        if set(train.label.unique()) != set(LABELS):
            raise ValueError("Le sous-ensemble d'écriture ne contient pas trois classes")
    model = classical_model(kind, seed, C=C, preserve_symbols=preserve_symbols)
    started = time.perf_counter()
    model.fit(train.text, train.label)
    elapsed = time.perf_counter() - started
    probabilities = model.predict_proba(dev.text)[:, [list(model.classes_).index(l) for l in LABELS]]
    joblib.dump(model, folder / "classifier.joblib")
    classifier = model.named_steps["classifier"]
    parameters = classifier.coef_.size + classifier.intercept_.size if hasattr(classifier, "coef_") else 0
    return finish_run(folder, family, variant, seed, {"kind": kind, "train_script": train_script, "n_train": len(train), "class_weight": None if kind == "majority" else "balanced", "C": C, "preserve_symbols": preserve_symbols, "cleaning": cleaning}, elapsed, parameters, probabilities, dev, "classical", purpose="cross_script" if train_script else purpose)


def predict_tensor(model, inputs, device="cpu", batch_size=64):
    model.eval()
    loader = DataLoader(TensorDataset(*inputs), batch_size=batch_size)
    scores = []
    with torch.inference_mode():
        for batch in loader:
            scores.append(model(*[x.to(device) for x in batch]).softmax(-1).cpu().numpy())
    return np.concatenate(scores)


def fit_tensor(model, train_inputs, train_labels, dev_inputs, dev_labels, config, auxiliary_data=None, checkpoint_path=None):
    """CE pondérée normalisée sur le batch effectif ; E2 = L_ary + lambda * L_arq.

    L'auxiliaire est cyclé pour donner à chaque pas le poids lambda annoncé,
    indépendamment de la taille relative des deux corpus. La sélection reste ary/dev.
    """
    from itertools import islice
    device = config.get("device", "cpu")
    model.to(device)
    micro, effective = config["micro_batch"], config["effective_batch"]
    if effective % micro or micro < 1:
        raise ValueError("Le micro-batch positif doit diviser le batch effectif")
    if not len(train_labels) or not len(dev_labels):
        raise ValueError("Train et dev doivent être non vides")
    def data_loader(inputs, values, oversample=False):
        labels = torch.as_tensor(values, dtype=torch.long)
        counts = torch.bincount(labels, minlength=3).float().clamp_min(1)
        weights = len(labels) / (3 * counts)
        sampler = WeightedRandomSampler(weights[labels], len(labels), replacement=True) if oversample else None
        loader = DataLoader(TensorDataset(*inputs, labels), batch_size=micro, shuffle=sampler is None, sampler=sampler)
        return loader, weights.to(device) if config.get("weighted_loss") else torch.ones(3, device=device)
    loader, weights = data_loader(train_inputs, train_labels, config.get("oversampling", False))
    auxiliary_loader, auxiliary_weights = (None, None) if auxiliary_data is None else data_loader(*auxiliary_data)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config["lr"], weight_decay=0.01)
    accumulation = effective // micro
    steps = math.ceil(len(loader) / accumulation) * config["epochs"]
    warmup = max(1, math.ceil(steps * config.get("warmup_ratio", 0))) if config.get("warmup_ratio", 0) else 0
    def factor(step):
        return (step + 1) / warmup if step < warmup else max(0.0, (steps - step) / max(1, steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor) if warmup else None
    amp = bool(config.get("amp", False) and str(device).startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_score, patience, best_state, history = -1.0, 0, None, []
    start_epoch, stopped = 1, False
    signature = None
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        stable_config = {k: v for k, v in config.items() if k not in {"setup_seconds", "embedding_cache_used", "embedding_training_seconds", "extraction_cache_used", "resumed_epoch_seconds", "resumed_epochs"}}
        inputs = [*train_inputs, torch.as_tensor(train_labels), *dev_inputs, torch.as_tensor(dev_labels)]
        if auxiliary_data is not None:
            inputs += [*auxiliary_data[0], torch.as_tensor(auxiliary_data[1])]
        signature = {"config": stable_config, "inputs": [hashlib.sha256(memoryview(t.detach().cpu().contiguous().numpy()).cast("B")).hexdigest() for t in inputs]}
        if checkpoint_path.exists():
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if saved["signature"] != signature:
                raise RuntimeError("Checkpoint incompatible avec les données/configuration ; utiliser une nouvelle campagne.")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(saved["scheduler"])
            scaler.load_state_dict(saved["scaler"])
            best_state, best_score, patience, history = saved["best_model"], saved["best_score"], saved["patience"], saved["history"]
            start_epoch, stopped = saved["epoch"] + 1, saved["stopped"]
            config["resumed_epochs"] = len(history)
            config["resumed_epoch_seconds"] = sum(item["seconds"] for item in history)
            random.setstate(saved["python_rng"])
            nr = saved["numpy_rng"]
            np.random.set_state((nr[0], np.array(nr[1], dtype="uint32"), *nr[2:]))
            torch.set_rng_state(saved["torch_rng"])
            if torch.cuda.is_available() and saved["cuda_rng"]:
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
            print(f"Reprise après l'époque {saved['epoch']} : {checkpoint_path}", flush=True)
    for epoch in range(start_epoch, config["epochs"] + 1) if not stopped else ():
        started = time.perf_counter()
        model.train()
        auxiliary_iterator = iter(auxiliary_loader) if auxiliary_loader is not None else None
        total_loss, n_seen = 0.0, 0
        iterator = iter(loader)
        while True:
            group = list(islice(iterator, accumulation))
            if not group:
                break
            auxiliary_group = []
            if auxiliary_loader is not None:
                for _ in group:
                    try:
                        batch = next(auxiliary_iterator)
                    except StopIteration:
                        auxiliary_iterator = iter(auxiliary_loader)
                        batch = next(auxiliary_iterator)
                    auxiliary_group.append(batch)
            denominator = sum(weights[batch[-1].to(device)].sum() for batch in group)
            auxiliary_denominator = sum(auxiliary_weights[batch[-1].to(device)].sum() for batch in auxiliary_group) if auxiliary_group else None
            optimizer.zero_grad(set_to_none=True)
            for index, batch in enumerate(group):
                target = batch[-1].to(device)
                with torch.autocast(device_type="cuda" if amp else "cpu", enabled=amp):
                    logits = model(*[x.to(device) for x in batch[:-1]])
                    summed = nn.functional.cross_entropy(logits, target, weight=weights, reduction="sum")
                    loss = summed / denominator
                    if auxiliary_group:
                        aux = auxiliary_group[index]
                        aux_logits = model(*[x.to(device) for x in aux[:-1]], task="arq")
                        aux_loss = nn.functional.cross_entropy(aux_logits, aux[-1].to(device), weight=auxiliary_weights, reduction="sum")
                        loss = loss + config["auxiliary_weight"] * aux_loss / auxiliary_denominator
                scaler.scale(loss).backward()
                total_loss += float(summed.detach())
                n_seen += float(weights[target].sum())
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and scaler.get_scale() >= previous_scale:
                scheduler.step()
        scores = predict_tensor(model, dev_inputs, device, batch_size=micro)
        metrics = metric_scores(np.array(LABELS)[dev_labels], decide(scores))
        history.append({"epoch": epoch, "loss": total_loss / n_seen, "dev_weighted_f1": metrics["weighted_f1"], "dev_macro_f1": metrics["macro_f1"], "seconds": time.perf_counter() - started})
        print(f"  epoch {epoch}/{config['epochs']} loss={total_loss/n_seen:.4f} dev={metrics['weighted_f1']:.4f}", flush=True)
        if metrics["weighted_f1"] > best_score + 1e-8:
            best_score, patience = metrics["weighted_f1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        stopped = epoch >= config.get("min_epochs", 1) and patience >= config.get("patience", 2)
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            nr = np.random.get_state()
            temporary = checkpoint_path.with_suffix(".pt.part")
            torch.save({"signature": signature, "epoch": epoch, "stopped": stopped,
                        "model": model.state_dict(), "best_model": best_state, "best_score": best_score,
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "scaler": scaler.state_dict(), "patience": patience, "history": history,
                        "python_rng": random.getstate(), "numpy_rng": (nr[0], nr[1].tolist(), *nr[2:]),
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, temporary)
            temporary.replace(checkpoint_path)
        if stopped:
            break
    model.load_state_dict(best_state)
    return model, history


def neural_run(kind="attention", embedding="random", seed=42, positional=True, weighted_loss=False, oversampling=False, aravec_path=None, epochs=3, device="cpu", max_length=None, open_fasttext=False):
    seed_everything(seed)
    if kind == "charcnn" and embedding != "random":
        raise ValueError("AraVec/fastText sont des embeddings de mots ; non applicables au vocabulaire caractères.")
    variant = f"{kind}_{embedding}" + ("_nopos" if not positional else "") + ("_weighted" if weighted_loss else "") + ("_oversampling" if oversampling else "")
    folder = run_folder(f"C_{variant}_s{seed}")
    train, dev = load_split("train"), load_split("dev")
    character = kind == "charcnn"
    vocab = build_vocab(train.text, character=character)
    config = {"kind": kind, "embedding": embedding, "positional": positional, "weighted_loss": weighted_loss, "oversampling": oversampling, "max_length": max_length or (256 if character else 64),
              "min_freq": 2, "lr": 1e-3, "epochs": epochs, "micro_batch": 32, "effective_batch": 32, "patience": 3, "device": device, "n_train": len(train), "open_fasttext": open_fasttext, "cleaning": CLEANING}
    started = time.perf_counter()
    matrix, coverage = embedding_matrix(vocab, train.text, embedding, seed, aravec_path, ROOT / "results/cache" if open_fasttext else None)
    config["embedding_cache_used"] = coverage["embedding_cache_used"]
    config["embedding_training_seconds"] = coverage["embedding_training_seconds"]
    model = TextNetwork(len(vocab), kind, matrix, positional)
    if embedding == "fasttext" and open_fasttext:
        config["fasttext_path"] = str(Path(coverage["fasttext_path"]).relative_to(ROOT))
        train_inputs = word_inputs(train.text, vocab, config["max_length"], coverage["fasttext_path"])
        dev_inputs = word_inputs(dev.text, vocab, config["max_length"], coverage["fasttext_path"])
    else:
        train_inputs = (torch.tensor(encode_texts(train.text.tolist(), vocab, config["max_length"], character)),)
        dev_inputs = (torch.tensor(encode_texts(dev.text.tolist(), vocab, config["max_length"], character)),)
    y_train, y_dev = [np.array([LABELS.index(y) for y in f.label]) for f in (train, dev)]
    model, history = fit_tensor(model, train_inputs, y_train, dev_inputs, y_dev, config, checkpoint_path=folder / "resume.pt")
    elapsed = time.perf_counter() - started + config.get("resumed_epoch_seconds", 0.0)
    scores = predict_tensor(model, dev_inputs, device)
    save_weights(model, folder)
    save_json(folder / "vocab.json", vocab)
    save_json(folder / "embedding_coverage.json", coverage)
    return finish_run(folder, "C", variant, seed, config, elapsed, sum(p.numel() for p in model.parameters() if p.requires_grad), scores, dev, "neural", history=history)


def pretrained_run(kind="nli", language="fr", device="cpu"):
    # Modèles fixes : répéter trois graines ne créerait pas trois expériences indépendantes.
    seed_everything(42)
    variant = f"nli_{language}" if kind == "nli" else "camelbert"
    folder = run_folder(f"B_{variant}_s42")
    started = time.perf_counter()
    predictor = PretrainedSentiment(kind, language, device=device)
    dev = load_split("dev")
    scores = predictor.predict_proba(dev.text.tolist())
    config = {"kind": kind, "language": language, "model_id": predictor.model_id, "revision": MODEL_REVISIONS[predictor.model_id], "local_training": False, "device": device, "max_length": 128}
    # Checkpoint externe référencé à une révision immuable ; pas de poids inventés.
    save_json(folder / "pretrained.json", config)
    return finish_run(folder, "B", variant, 42, config, 0.0, 0, scores, dev, "pretrained", history=[{"inference_dev_seconds": time.perf_counter() - started}])


def feature_cache_path(split, pooling):
    max_length = int(os.environ.get("DARIJA_MAX_LENGTH", "64"))
    cache = ROOT / "results/cache"
    cache.mkdir(parents=True, exist_ok=True)
    prefix = "" if CLEANING == "historical" and max_length == 64 else f"{CLEANING}_{max_length}_"
    extension = ".npy" if pooling == "states" and CLEANING != "historical" else ".npz"
    return cache / f"{prefix}{split}_{pooling}_{MODEL_REVISIONS[BERT_ID]}_{sha256(PROJECT_ROOT / f'data/ary/{split}.tsv')[:12]}{extension}"


def valid_feature_cache(split, pooling):
    path = feature_cache_path(split, pooling)
    if not path.is_file():
        return False
    try:
        if path.suffix == ".npy":
            values = np.load(path, mmap_mode="r")
            try:
                mask = np.load(path.with_suffix(".mask.npy"))
                return (values.ndim == 3 and np.issubdtype(values.dtype, np.floating) and mask.shape == values.shape[:2]
                        and len(values) == len(load_split(split)) and values.shape[2] > 0
                        and values.shape[1] == int(os.environ.get("DARIJA_MAX_LENGTH", "64")) and np.isin(mask, [0, 1]).all())
            finally:
                values._mmap.close()
        with np.load(path) as saved:
            values, mask = saved["features"], saved["mask"]
            return (values.ndim == 2 and np.issubdtype(values.dtype, np.floating) and mask.ndim == 2 and len(values) == len(mask) == len(load_split(split))
                    and values.shape[1] > 0 and mask.shape[1] == int(os.environ.get("DARIJA_MAX_LENGTH", "64")) and np.isin(mask, [0, 1]).all())
    except (OSError, ValueError, KeyError, EOFError, BadZipFile):
        return False


def frozen_features(encoder, split, pooling):
    path = feature_cache_path(split, pooling)
    if valid_feature_cache(split, pooling):
        if path.suffix == ".npy":
            return np.load(path, mmap_mode="c"), np.load(path.with_suffix(".mask.npy"))
        with np.load(path) as saved:
            return saved["features"], saved["mask"]
    if encoder is None:
        raise RuntimeError(f"Cache absent ou incomplet : {path}. Préparer l'encodeur pour reconstruire l'extraction.")
    frame = load_split(split)
    started = time.perf_counter()
    temporary = path.with_suffix(path.suffix + ".part")
    options = {"output_path": temporary} if path.suffix == ".npy" else {}
    values, mask = encoder.extract(frame.text.tolist(), pooling, batch_size=4, max_length=int(os.environ.get("DARIJA_MAX_LENGTH", "64")), **options)
    elapsed = time.perf_counter() - started
    if path.suffix == ".npy":
        values.flush()
        values._mmap.close()
        del values
        mask_path = path.with_suffix(".mask.npy")
        mask_temporary = mask_path.with_suffix(".npy.part")
        with mask_temporary.open("wb") as output:
            np.save(output, mask)
        mask_temporary.replace(mask_path)
        temporary.replace(path)
        values = np.load(path, mmap_mode="c")
    else:
        with temporary.open("wb") as output:
            np.savez_compressed(output, features=values, mask=mask)
        temporary.replace(path)
    timing = ROOT / "results/frozen_extraction_time.json"
    records = json.loads(timing.read_text(encoding="utf-8")) if timing.exists() else []
    records.append({"split": split, "pooling": pooling, "seconds": elapsed})
    save_json(timing, records)
    return values, mask


def predict_split(folder, frame, split):
    """Réutilise les extractions D entre graines ; latence G mesurée sans ce cache."""
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    if meta["kind"] != "frozen":
        return load_predictor(folder, device=os.environ.get("DARIJA_EVAL_DEVICE", "cpu"))(frame.text.tolist())
    config = meta["config"]
    device = os.environ.get("DARIJA_EVAL_DEVICE", "cpu")
    encoder = None if valid_feature_cache(split, config["pooling"]) else FrozenEncoder(config["model_id"], device=device)
    features, mask = frozen_features(encoder, split, config["pooling"])
    if config["pooling"] != "states":
        model = joblib.load(folder / "classifier.joblib")
        return model.predict_proba(features)[:, [list(model.classes_).index(y) for y in LABELS]]
    model = FrozenSequence(config["hidden_size"])
    model.load_state_dict(torch.load(folder / "weights.pt", map_location="cpu", weights_only=True))
    model.to(device)
    return predict_tensor(model, (torch.from_numpy(features), torch.tensor(mask)), device)


def frozen_run(pooling="mean", seed=42, device="cpu", epochs=3):
    seed_everything(seed)
    folder = run_folder(f"D_{pooling}_s{seed}")
    started = time.perf_counter()
    cache_used = all(valid_feature_cache(s, pooling) for s in ("train", "dev"))
    encoder = None if cache_used else FrozenEncoder(device=device)
    train, dev = load_split("train"), load_split("dev")
    x_train, train_mask = frozen_features(encoder, "train", pooling)
    x_dev, dev_mask = frozen_features(encoder, "dev", pooling)
    if encoder is None:
        from transformers import AutoConfig
        hidden_size = AutoConfig.from_pretrained(BERT_ID, revision=MODEL_REVISIONS[BERT_ID]).hidden_size
    else:
        hidden_size = encoder.model.config.hidden_size
    del encoder
    config = {"model_id": BERT_ID, "revision": MODEL_REVISIONS[BERT_ID], "pooling": pooling, "encoder_frozen": True, "max_length": int(os.environ.get("DARIJA_MAX_LENGTH", "64")), "device": device, "hidden_size": hidden_size, "extraction_cache_used": cache_used, "n_train": len(train)}
    if pooling == "states":
        model = FrozenSequence(hidden_size)
        config.update(lr=1e-3, epochs=epochs, micro_batch=8, effective_batch=32, patience=3)
        y_train, y_dev = [np.array([LABELS.index(y) for y in f.label]) for f in (train, dev)]
        model, history = fit_tensor(model, (torch.from_numpy(x_train), torch.tensor(train_mask)), y_train, (torch.from_numpy(x_dev), torch.tensor(dev_mask)), y_dev, config, checkpoint_path=folder / "resume.pt")
        scores = predict_tensor(model, (torch.from_numpy(x_dev), torch.tensor(dev_mask)), device)
        save_weights(model, folder)
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
        model.fit(x_train, train.label)
        scores = model.predict_proba(x_dev)[:, [list(model.classes_).index(y) for y in LABELS]]
        joblib.dump(model, folder / "classifier.joblib")
        parameters, history = model.coef_.size + model.intercept_.size, None
    return finish_run(folder, "D", pooling, seed, config, time.perf_counter() - started + config.get("resumed_epoch_seconds", 0.0), parameters, scores, dev, "frozen", history=history)


def finetune_run(model_id=BERT_ID, seed=42, micro_batch=4, device="cpu", epochs=3, max_length=128, auxiliary=None, auxiliary_weight=0.0, cleaning=None, family="E", tag=""):
    from transformers import AutoModel, AutoTokenizer
    seed_everything(seed)
    if epochs < 3 or epochs > 5:
        raise ValueError("E : protocole 3 à 5 époques")
    variant = "arabizi" if model_id.endswith("arabizi") else "darijabert"
    if auxiliary_weight:
        variant += f"_mtl_l{auxiliary_weight:g}"
    variant += tag
    folder = run_folder(f"{family}_{variant}_s{seed}")
    train, dev = load_split("train"), load_split("dev")
    cleaning = cleaning or CLEANING
    train, dev = [transform_frame(frame, cleaning) for frame in (train, dev)]
    config = {"model_id": model_id, "revision": MODEL_REVISIONS[model_id], "device": device, "lr": 2e-5, "micro_batch": micro_batch,
              "effective_batch": 32, "epochs": epochs, "min_epochs": 3, "warmup_ratio": 0.1, "patience": 2, "max_length": max_length, "n_train": len(train), "optimizer": "AdamW", "task": "ary+arq" if auxiliary_weight else "ary_only", "auxiliary_weight": auxiliary_weight, "weighted_loss": True, "amp": device == "cuda", "cleaning": cleaning}
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=config["revision"])
    architecture = MultiTaskSentiment if auxiliary_weight else FineTunedSentiment
    model = architecture(AutoModel.from_pretrained(model_id, revision=config["revision"], add_pooling_layer=False))
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
    def make_inputs(frame):
        batch = tokenizer(frame.text.tolist(), padding="max_length", truncation=True, max_length=config["max_length"], return_tensors="pt")
        return batch.input_ids, batch.attention_mask
    train_inputs, dev_inputs = make_inputs(train), make_inputs(dev)
    y_train, y_dev = [np.array([LABELS.index(y) for y in f.label]) for f in (train, dev)]
    auxiliary_data = None
    if auxiliary_weight:
        if auxiliary is None or auxiliary.empty:
            raise ValueError("E2 demande un train arq non vide")
        config["n_arq_train"] = len(auxiliary)
        config["arq_train_sha256"] = sha256(PROJECT_ROOT / "data/arq/train.tsv")
        auxiliary_data = (make_inputs(auxiliary), np.array([LABELS.index(y) for y in auxiliary.label]))
    config["setup_seconds"] = time.perf_counter() - started
    fit_started = time.perf_counter()
    model, history = fit_tensor(model, train_inputs, y_train, dev_inputs, y_dev, config, auxiliary_data, checkpoint_path=folder / "resume.pt")
    training_seconds = time.perf_counter() - fit_started + config.get("resumed_epoch_seconds", 0.0)
    scores = predict_tensor(model, dev_inputs, device, batch_size=micro_batch)
    save_weights(model, folder)
    tokenizer.save_pretrained(folder / "tokenizer")
    model.encoder.config.save_pretrained(folder / "encoder_config")
    return finish_run(folder, family, variant, seed, config, training_seconds, sum(p.numel() for p in model.parameters() if p.requires_grad), scores, dev, "multitask" if auxiliary_weight else "finetuned", history=history)


def load_predictor(folder, device="cpu"):
    """Recharge un checkpoint local sans téléchargement ; les campagnes préparent le cache."""
    folder = Path(folder)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    config = meta["config"]
    if meta["kind"] == "classical":
        model = joblib.load(folder / "classifier.joblib")
        order = [list(model.classes_).index(y) for y in LABELS]
        return lambda texts: model.predict_proba([clean_text(t, config.get("cleaning", CLEANING)) or "<EMPTY>" for t in texts])[:, order]
    if meta["kind"] == "pretrained":
        model = PretrainedSentiment(config["kind"], config["language"], device, local_files_only=True, revision=config["revision"])
        return model.predict_proba
    if meta["kind"] == "neural":
        vocab = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
        model = TextNetwork(len(vocab), config["kind"], positional=config["positional"])
        model.load_state_dict(torch.load(folder / "weights.pt", map_location="cpu", weights_only=True))
        model.to(device).eval()
        def predict(texts):
            if config.get("fasttext_path"):
                inputs = word_inputs(texts, vocab, config["max_length"], ROOT / config["fasttext_path"])
            else:
                inputs = (torch.tensor(encode_texts(list(texts), vocab, config["max_length"], config["kind"] == "charcnn")),)
            return predict_tensor(model, inputs, device)
        return predict
    if meta["kind"] == "frozen":
        encoder = FrozenEncoder(config["model_id"], device, local_files_only=True, revision=config["revision"])
        if config["pooling"] != "states":
            classifier = joblib.load(folder / "classifier.joblib")
            order = [list(classifier.classes_).index(y) for y in LABELS]
            return lambda texts: classifier.predict_proba(encoder.extract(texts, config["pooling"], max_length=config["max_length"])[0])[:, order]
        model = FrozenSequence(config["hidden_size"])
        model.load_state_dict(torch.load(folder / "weights.pt", map_location="cpu", weights_only=True))
        model.to(device).eval()
        def predict(texts):
            states, mask = encoder.extract(texts, "states", max_length=config["max_length"])
            return predict_tensor(model, (torch.tensor(states, dtype=torch.float32), torch.tensor(mask)), device)
        return predict
    if meta["kind"] in {"finetuned", "multitask"}:
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        architecture = MultiTaskSentiment if meta["kind"] == "multitask" else FineTunedSentiment
        model = architecture(AutoModel.from_config(AutoConfig.from_pretrained(folder / "encoder_config"), add_pooling_layer=False))
        model.load_state_dict(torch.load(folder / "weights.pt", map_location="cpu", weights_only=True))
        model.to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(folder / "tokenizer")
        def predict(texts):
            inputs = tokenizer([clean_text(t, config.get("cleaning", CLEANING)) or "<EMPTY>" for t in texts], padding="max_length", truncation=True, max_length=config["max_length"], return_tensors="pt")
            return predict_tensor(model, (inputs.input_ids, inputs.attention_mask), device, batch_size=config["micro_batch"])
        return predict
    raise ValueError(meta["kind"])


def plan_table():
    rows = [{"family": "A", "variant": v, "seeds": list(SEEDS)} for v in ("majority", "word", "char", "union")]
    rows += [{"family": "B", "variant": v, "seeds": [42]} for v in ("nli_ar", "nli_fr", "nli_en", "camelbert")]
    for kind, positional in (("bilstm", True), ("attention", True), ("transformer", True), ("transformer", False)):
        for embedding in ("random", "fasttext", "aravec"):
            rows.append({"family": "C", "variant": f"{kind}_{embedding}" + ("_nopos" if not positional else ""), "seeds": list(SEEDS)})
    rows += [{"family": "C", "variant": v, "seeds": list(SEEDS)} for v in ("charcnn_random", "attention_random_weighted", "attention_random_oversampling")]
    rows += [{"family": "D", "variant": v, "seeds": list(SEEDS)} for v in ("cls", "mean", "states")]
    rows += [{"family": "E", "variant": v, "seeds": list(SEEDS)} for v in ("darijabert", "arabizi")]
    table = pd.DataFrame(rows)
    executed = results_table()
    def state(row):
        observed = executed.loc[(executed.family == row.family) & (executed.variant == row.variant) & (executed.split == "dev")]
        seeds = set(observed.seed.dropna().astype(int))
        return "exécuté" if set(row.seeds) <= seeds else f"non exécuté ({len(seeds)}/{len(row.seeds)} graines)"
    table["status"] = table.apply(state, axis=1)
    return table


def run_campaign(part="quick", aravec_path=None, device="cpu"):
    if part in {"quick", "all", "A"}:
        for kind in ("majority", "word", "char", "union"):
            for seed in SEEDS:
                classical_run(kind, seed)
        for script in ("arabic", "latin"):
            for seed in SEEDS:
                classical_run("union", seed, train_script=script)
    if part in {"quick", "C", "all"}:
        for seed in SEEDS:
            for kind, positional in (("bilstm", True), ("attention", True), ("transformer", True), ("transformer", False), ("charcnn", True)):
                neural_run(kind, "random", seed, positional=positional, device=device)
            neural_run("attention", "fasttext", seed, device=device)
            neural_run("attention", "random", seed, weighted_loss=True, device=device)
            neural_run("attention", "random", seed, oversampling=True, device=device)
    if part in {"C", "all"}:
        for seed in SEEDS:
            for kind, positional in (("bilstm", True), ("transformer", True), ("transformer", False)):
                neural_run(kind, "fasttext", seed, positional=positional, device=device)
            if aravec_path:
                for kind, positional in (("bilstm", True), ("attention", True), ("transformer", True), ("transformer", False)):
                    neural_run(kind, "aravec", seed, positional=positional, aravec_path=aravec_path, device=device)
    if part in {"B", "all"}:
        for language in ("ar", "fr", "en"):
            pretrained_run("nli", language)
        pretrained_run("camel")
    if part in {"D", "all"}:
        for pooling in ("cls", "mean", "states"):
            for seed in SEEDS:
                frozen_run(pooling, seed, device)
    if part in {"E", "all"}:
        for model_id in (BERT_ID, "SI2M-Lab/DarijaBERT-arabizi"):
            for seed in SEEDS:
                finetune_run(model_id, seed, device=device)
    plan_table().to_csv(ROOT / "results/execution_status.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Campagne ary : sélection dev puis évaluation séparée")
    parser.add_argument("--part", choices=["quick", "all", "A", "B", "C", "D", "E", "status", "finalize"], default="status")
    parser.add_argument("--aravec-path")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.part == "status":
        print(plan_table().to_string(index=False))
    elif args.part == "finalize":
        from .evaluate import lock_selection, final_evaluation
        lock_selection()
        print(final_evaluation().to_string(index=False))
    else:
        save_json(ROOT / "results/runtime.json", runtime_info())
        run_campaign(args.part, args.aravec_path, args.device)
