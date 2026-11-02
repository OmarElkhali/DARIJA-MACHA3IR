"""Mesures multiclasse, règles dev, verrouillage et robustesse après sélection."""
from itertools import product
from pathlib import Path
import json
import re
import time
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from .preprocess import ROOT, LABELS, load_split, save_json, sha256, annotation_audit, clean_text


def metric_scores(truth, predicted):
    return {"weighted_f1": f1_score(truth, predicted, labels=LABELS, average="weighted", zero_division=0),
            "macro_f1": f1_score(truth, predicted, labels=LABELS, average="macro", zero_division=0),
            "precision": precision_score(truth, predicted, labels=LABELS, average="weighted", zero_division=0),
            "recall": recall_score(truth, predicted, labels=LABELS, average="weighted", zero_division=0)}


def decide(probabilities, offsets=None):
    probabilities = np.asarray(probabilities, dtype=float)
    offsets = np.asarray(offsets if offsets is not None else [0, 0, 0], dtype=float)
    if (probabilities.ndim != 2 or probabilities.shape[1] != len(LABELS)
            or not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5, rtol=0)):
        raise ValueError("Scores invalides : trois probabilités finies et normalisées sont attendues par texte.")
    if offsets.shape != (len(LABELS),) or not np.isfinite(offsets).all():
        raise ValueError("Décalages invalides : trois valeurs finies sont attendues.")
    scores = np.log(np.clip(probabilities, 1e-12, 1)) + offsets
    return np.array(LABELS)[scores.argmax(axis=1)]


def tune_offsets(truth, probabilities):
    """Deux décalages relatifs ; premier = 0 pour lever l'indétermination."""
    best, best_score = [0.0, 0.0, 0.0], f1_score(truth, decide(probabilities), average="weighted", zero_division=0)
    for neutral, positive in product((-0.6, -0.3, 0.0, 0.3, 0.6), repeat=2):
        offsets = [0.0, neutral, positive]
        score = f1_score(truth, decide(probabilities, offsets), average="weighted", zero_division=0)
        if score > best_score + 1e-12:
            best, best_score = offsets, score
    return best


def results_table():
    path = ROOT / "results/results.csv"
    if not path.exists():
        return pd.DataFrame(columns=["run_id", "family", "variant", "seed", "split", "decision", "purpose", "weighted_f1", "macro_f1"])
    return pd.read_csv(path)


def record_metrics(meta, frame, probabilities, split, raw=False):
    offsets = [0, 0, 0] if raw else meta["offsets"]
    predicted = decide(probabilities, offsets)
    row = {k: meta.get(k) for k in ("run_id", "family", "variant", "seed", "train_seconds", "trainable_parameters", "checkpoint", "purpose")}
    row.update(split=split, decision="argmax" if raw else "dev_offsets", n=len(frame), **metric_scores(frame.label, predicted))
    table = results_table()
    if not table.empty:
        table = table.loc[~((table.run_id == row["run_id"]) & (table.split == split) & (table.decision == row["decision"]))]
    table = pd.DataFrame([row]) if table.empty else pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    result_path = ROOT / "results/results.csv"
    temporary = result_path.with_suffix(".csv.tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(result_path)
    prediction_frame = frame[["row_id", "script", "label"]].copy()
    prediction_frame["prediction"] = predicted
    for i, label in enumerate(LABELS):
        prediction_frame[f"p_{label}"] = probabilities[:, i]
    prediction_frame.to_csv(ROOT / "results" / f"{meta['run_id']}_{split}_{'raw' if raw else 'tuned'}.csv", index=False)
    if not raw and "text" in frame:
        from .validation import subgroup_scores
        subgroup_scores(frame, probabilities, offsets).to_csv(ROOT / "results" / f"{meta['run_id']}_{split}_subgroups.csv", index=False)
        audit = ROOT / "results/audit/near_duplicates.csv"
        if audit.exists():
            near = pd.read_csv(audit)
            excluded = near.loc[near.reference_split.eq("train") & near.query_split.eq(split), "row_id"]
            mask = ~frame.row_id.isin(excluded).to_numpy()
            pd.DataFrame([{"split": split, "n": int(mask.sum()), "scope": "sensibilité sans quasi-doublons train, sélection inchangée",
                           **(metric_scores(frame.label.to_numpy()[mask], predicted[mask]) if mask.any() else {})}]).to_csv(
                ROOT / "results" / f"{meta['run_id']}_{split}_without_near_train.csv", index=False)
    return row


def by_script(frame, probabilities, offsets=None):
    predicted = decide(probabilities, offsets)
    rows = []
    for script in ("arabic", "latin", "mixed", "other"):
        mask = frame.script.eq(script).to_numpy()
        if mask.any():
            rows.append({"script": script, "n": int(mask.sum()), **metric_scores(frame.label.to_numpy()[mask], predicted[mask])})
    return pd.DataFrame(rows)


def select_on_dev():
    table = results_table()
    if table.empty:
        raise RuntimeError("Aucun résultat de validation : terminer les entraînements avant la sélection.")
    table = table.loc[(table.split == "dev") & (table.decision == "dev_offsets") & (table.purpose == "benchmark")]
    if table.empty:
        raise RuntimeError("Aucun résultat dev de benchmark")
    summary = table.groupby(["family", "variant"], as_index=False).agg(dev_weighted_f1=("weighted_f1", "mean"), dev_macro_f1=("macro_f1", "mean"), seed_std=("weighted_f1", "std"), seeds=("seed", "nunique"))
    expected = summary.apply(lambda r: 1 if r.family in {"A", "B"} or (r.family == "D" and r.variant != "states") or (r.family == "P" and not r.variant.startswith("darijabert_")) else 3, axis=1)
    legacy_deterministic = expected.eq(1) & ~summary.family.eq("B") & summary.seeds.eq(3)
    summary = summary.loc[summary.seeds.eq(expected) | legacy_deterministic]
    if summary.empty:
        raise RuntimeError("Aucune variante complète : une exécution pour les modèles déterministes, trois graines pour les réseaux entraînés.")
    return summary.sort_values(["dev_weighted_f1", "family", "variant"], ascending=[False, True, True])


def lock_selection():
    path = ROOT / "results/selection_lock.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    selected = select_on_dev()
    rows = results_table()
    runs = rows.loc[(rows.split == "dev") & (rows.decision == "dev_offsets"), "run_id"].unique()
    artifacts = {}
    for run in runs:
        folder = ROOT / "models" / run
        artifacts[run] = {str(p.relative_to(ROOT)): sha256(p) for p in folder.rglob("*") if p.is_file()}
    # Le représentant est la graine 42 prédéfinie, jamais la meilleure graine sur test.
    winners = selected.drop_duplicates("family")
    lock = {"created_utc": pd.Timestamp.now(tz="UTC").isoformat(), "criterion": "mean dev weighted_f1, offsets dev, seed42 representative", "selected_variants": winners.astype(object).where(pd.notna(winners), None).to_dict("records"),
            "candidate": selected.iloc[0][["family", "variant"]].to_dict(), "artifacts": artifacts, "representative_seed": 42}
    save_json(path, lock)
    return lock


def assert_locked():
    path = ROOT / "results/selection_lock.json"
    if not path.exists():
        raise RuntimeError("Sélection non verrouillée")
    lock = json.loads(path.read_text(encoding="utf-8"))
    for files in lock["artifacts"].values():
        for file, digest in files.items():
            if sha256(ROOT / file) != digest:
                raise RuntimeError(f"Checkpoint ou règle modifiés après verrouillage : {file}")
    for file, digest in lock.get("shared_artifacts", {}).items():
        if sha256(ROOT / file) != digest:
            raise RuntimeError(f"Ressource partagée modifiée après verrouillage : {file}")
    return lock


def save_confusion(truth, predicted, name):
    matrix = confusion_matrix(truth, predicted, labels=LABELS)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    picture = ax.imshow(matrix, cmap="Blues")
    for (r, c), value in np.ndenumerate(matrix):
        ax.text(c, r, str(value), ha="center", va="center", color="white" if value > matrix.max() / 2 else "black")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS, xlabel="Prédit", ylabel="Réel", title=name)
    fig.colorbar(picture, ax=ax)
    fig.tight_layout()
    target = ROOT / "results/figures" / f"{name}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=140)
    plt.close(fig)
    return target


def bootstrap_ci(truth, prediction, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if len(truth) == 0 or len(truth) != len(prediction) or n_bootstrap < 1:
        raise ValueError("Bootstrap : deux tableaux non vides, de même longueur, et un nombre de tirages positif sont requis")
    scores = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(truth), len(truth))
        scores.append(f1_score(truth[indices], prediction[indices], labels=LABELS, average="weighted", zero_division=0))
    low, high = np.quantile(scores, [0.025, 0.975])
    return {"weighted_f1": float(f1_score(truth, prediction, average="weighted")), "ci_low": float(low), "ci_high": float(high), "bootstrap": n_bootstrap, "seed": seed}


def mcnemar(truth, first, second):
    truth, first, second = np.asarray(truth), np.asarray(first), np.asarray(second)
    b = int(((first == truth) & (second != truth)).sum())
    c = int(((first != truth) & (second == truth)).sum())
    return {"first_only_correct": b, "second_only_correct": c, "p_exact": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0,
            "scope": "différence de taux d'erreur apparié, pas test direct du F1"}


def strip_emojis_punctuation(text):
    return clean_text(text, "without_both")


def robustness_f4(frame, probabilities, predictor, offsets, filename="f4_perturbations.csv", transformation="without_both"):
    baseline = decide(probabilities, offsets)
    subset = frame.copy()  # Même population pour tous les modèles, incluant les erreurs initiales et neutres.
    subset["changed_text"] = subset.text.map(lambda text: clean_text(text, transformation) or "<EMPTY>")
    subset["baseline_prediction"] = baseline
    subset["altered_prediction"] = baseline
    subset["transformation"] = transformation
    changed = subset.text.ne(subset.changed_text)
    if changed.any():
        subset.loc[changed, "altered_prediction"] = decide(predictor(subset.loc[changed, "changed_text"].tolist()), offsets)
    subset.to_csv(ROOT / "results" / filename, index=False)
    before_metrics = metric_scores(subset.label, baseline)
    after = subset.altered_prediction.to_numpy()
    after_metrics = metric_scores(subset.label, after)
    accuracy_before = float(np.mean(baseline == subset.label.to_numpy()))
    accuracy_after = float(np.mean(after == subset.label.to_numpy()))
    return {"transformation": transformation, "eligible": len(subset), "changed": int(changed.sum()),
            "flip_rate_changed": float(np.mean(after[changed] != baseline[changed])) if changed.any() else None,
            "accuracy_before": accuracy_before, "accuracy_after": accuracy_after, "accuracy_drop": accuracy_before - accuracy_after,
            "weighted_f1_before": before_metrics["weighted_f1"], "weighted_f1_after": after_metrics["weighted_f1"],
            "weighted_f1_drop": before_metrics["weighted_f1"] - after_metrics["weighted_f1"],
            "caveat": "Diagnostic sur les labels originaux, invariance du sentiment à valider humainement ; une bascule n'est pas nécessairement une erreur."}


def error_review(frame, probabilities, offsets):
    prediction = decide(probabilities, offsets)
    errors = frame.loc[prediction != frame.label.to_numpy()].copy()
    errors["prediction"] = prediction[prediction != frame.label.to_numpy()]
    if errors.empty:
        for column in ("categorie_provisoire", "validation_humaine", "commentaire_humain", "categorie_humaine", "rang_categorie"):
            errors[column] = pd.Series(dtype="object")
        errors.to_csv(ROOT / "results/f6_30_erreurs.csv", index=False)
        return errors
    errors = annotation_audit(errors)
    def category(row):
        # ponytail: catégories provisoires ; valider humainement avant de conclure à une cause.
        if row.get("vote_disagreement") is True or row.get("vote_disagreement") == True:
            return "bruit_annotation_possible"
        if row.label == "neutral" or row.prediction == "neutral":
            return "confusion_neutre_polarise"
        if re.search(r"\b(mais|avec|pour|pas|c'est|merci|vraiment)\b", row.text, re.I):
            return "francais_possible"
        if row.script in {"latin", "mixed"}:
            return "tokenisation_Arabizi_possible"
        return "ironie_ou_autre_a_examiner"
    errors["categorie_provisoire"] = errors.apply(category, axis=1)
    errors["validation_humaine"] = "à faire"
    errors["commentaire_humain"] = ""
    errors["categorie_humaine"] = ""
    # Tour de rôle par catégorie, puis remplissage ; pas seulement les erreurs les plus faciles.
    errors["rang_categorie"] = errors.groupby("categorie_provisoire").cumcount()
    selected = errors.sort_values(["rang_categorie", "categorie_provisoire", "row_id"]).head(30)
    review_path = ROOT / "results/f6_30_erreurs.csv"
    if review_path.exists():
        existing = pd.read_csv(review_path, keep_default_na=False).set_index("row_id")
        for column in ("validation_humaine", "commentaire_humain", "categorie_humaine"):
            if column in existing:
                selected[column] = selected.row_id.map(existing[column]).fillna(selected[column])
    selected.to_csv(review_path, index=False)
    return selected


def final_evaluation():
    """Unique frontière F/G : aucune sélection ne dépend de ce qui suit."""
    from .train import load_predictor, predict_split, valid_feature_cache, frozen_features
    from .validation import subgroup_scores, robustness_review, evaluate_human_sets
    lock = assert_locked()
    test, dev = load_split("test", labels=True), load_split("dev")
    distributions = pd.concat([dev.label.value_counts(normalize=True).rename("dev"), test.label.value_counts(normalize=True).rename("test")], axis=1).reindex(LABELS).fillna(0)
    distributions.to_csv(ROOT / "results/f1_label_shift.csv")
    train = load_split("train")
    joint = pd.concat([train.assign(split="train"), dev.assign(split="dev"), test.assign(split="test")])
    contradictory = joint.groupby("text").label.nunique()
    final_conflicts = joint.loc[joint.text.isin(contradictory[contradictory > 1].index), ["row_id", "split", "text", "label"]]
    final_conflicts.to_csv(ROOT / "results/audit/conflicts_including_test_final.csv", index=False)
    test_predictions, comparison = {}, []
    dev_rows = results_table().query("split == 'dev' and decision == 'dev_offsets'")
    selected_pairs = {(item["family"], item["variant"]) for item in lock["selected_variants"]}
    # Les variantes sont comparées sur dev ; test porte sur les représentants fixés et F2.
    declared = set(lock.get("test_runs", []))
    chosen = [row.run_id in declared if declared else ((row.family, row.variant) in selected_pairs or row.purpose == "cross_script") for row in dev_rows.itertuples()]
    dev_rows = dev_rows.loc[chosen]
    required_poolings = dev_rows.loc[dev_rows.family.eq("D"), "variant"].unique()
    if any(not valid_feature_cache("test", p) for p in required_poolings):
        from .models import FrozenEncoder
        print("F/G : extraction test D après verrouillage, variante retenue sur dev.", flush=True)
        encoder = FrozenEncoder(device=__import__("os").environ.get("DARIJA_EVAL_DEVICE", "cpu"))
        for pooling in required_poolings:
            array, mask = frozen_features(encoder, "test", pooling)
        del encoder, mask, array
    for run_id in dev_rows.run_id.unique():
        print(f"F/G : évaluation finale {run_id}", flush=True)
        meta = json.loads((ROOT / "models" / run_id / "meta.json").read_text(encoding="utf-8"))
        probabilities = predict_split(ROOT / meta["checkpoint"], test, "test")
        record_metrics(meta, test, probabilities, "test", raw=True)
        record_metrics(meta, test, probabilities, "test")
        by_script(test, probabilities, meta["offsets"]).to_csv(ROOT / "results" / f"{run_id}_scripts.csv", index=False)
        subgroup_scores(test, probabilities, meta["offsets"]).to_csv(ROOT / "results" / f"{run_id}_subgroups.csv", index=False)
        test_predictions[run_id] = (meta, probabilities)
        if meta["purpose"] == "cross_script":
            target = "latin" if meta["config"]["train_script"] == "arabic" else "arabic"
            part = test.script.eq(target).to_numpy()
            if part.any():
                comparison.append({"run_id": run_id, "train_script": meta["config"]["train_script"], "test_script": target, "n": int(part.sum()), **metric_scores(test.label.to_numpy()[part], decide(probabilities, meta["offsets"])[part])})
    near_file = ROOT / "results/audit/near_duplicates.csv"
    near_train_test = set()
    if near_file.exists():
        near = pd.read_csv(near_file)
        near_train_test = set(near.loc[near.reference_split.eq("train") & near.query_split.eq("test"), "row_id"])
    pd.DataFrame(comparison).to_csv(ROOT / "results/f2_cross_script.csv", index=False)
    final_rows = []
    reports = lock.get("report_variants", [next((w for w in lock["selected_variants"] if w["family"] == family), {"family": family}) for family in "ABCDE"])
    for winner in reports:
        family = winner["family"]
        if "variant" not in winner:
            final_rows.append({"family": family, "status": "non exécuté — aucun résultat test"})
            continue
        selected = [(m, p) for m, p in test_predictions.values() if m["family"] == family and m["variant"] == winner["variant"] and m["purpose"] == "benchmark"]
        meta, probabilities = next((m, p) for m, p in selected if m["seed"] == 42)
        prediction = decide(probabilities, meta["offsets"])
        predictor = load_predictor(ROOT / meta["checkpoint"], device="cpu")
        predictor([test.text.iloc[0]])
        start = time.perf_counter()
        for text in test.text.head(30):
            predictor([text])
        latency = (time.perf_counter() - start) / min(30, len(test)) * 1000
        scores = [metric_scores(test.label, decide(p, m["offsets"]))["weighted_f1"] for m, p in selected]
        folder = ROOT / meta["checkpoint"]
        weight_bytes = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file() and p.suffix in {".pt", ".joblib", ".safetensors", ".bin"})
        external_bytes = 0
        if meta["kind"] in {"pretrained", "frozen"}:
            from huggingface_hub import try_to_load_from_cache
            config = meta["config"]
            for filename in ("model.safetensors", "pytorch_model.bin"):
                cached = try_to_load_from_cache(config["model_id"], filename, revision=config["revision"])
                if isinstance(cached, str):
                    external_bytes = Path(cached).stat().st_size
                    break
        if meta["config"].get("fasttext_path"):
            fasttext_file = ROOT / meta["config"]["fasttext_path"]
            weight_bytes += sum(p.stat().st_size for p in fasttext_file.parent.glob(fasttext_file.name + "*") if p.suffix in {".model", ".npy"})
        weight_bytes += external_bytes
        feature_seconds = 0.0
        extraction = ROOT / "results/frozen_extraction_time.json"
        if meta["kind"] == "frozen" and meta["config"].get("extraction_cache_used") and extraction.exists():
            feature_seconds = sum(item["seconds"] for item in json.loads(extraction.read_text(encoding="utf-8")) if item.get("pooling", meta["config"]["pooling"]) == meta["config"]["pooling"] and item.get("split", "train") in {"train", "dev"})
        embedding_seconds = meta["config"].get("embedding_training_seconds", 0.0) if meta["config"].get("embedding_cache_used") else 0.0
        row = {"family": family, "status": "exécuté", "variant": winner["variant"], "run_id": meta["run_id"], **metric_scores(test.label, prediction), "dev_weighted_f1_mean": winner["dev_weighted_f1"], "setup_seconds": meta["config"].get("setup_seconds", 0.0),
               "train_seconds": meta["train_seconds"] + feature_seconds + embedding_seconds, "run_seconds": meta["train_seconds"], "feature_extraction_seconds_shared": feature_seconds, "embedding_training_seconds_shared": embedding_seconds, "cpu_ms_per_tweet": latency, "weight_mb": weight_bytes / 2**20, "external_encoder_mb": external_bytes / 2**20, "trainable_parameters": meta["trainable_parameters"],
               "test_seed_mean": np.mean(scores), "test_seed_std": np.std(scores, ddof=1) if len(scores) > 1 else None, "n_seeds": len(scores), "scripts": by_script(test, probabilities, meta["offsets"]).to_json(orient="records")}
        row.update(bootstrap_ci(test.label, prediction))
        unseen = ~test.text.isin(train.text).to_numpy()
        row["sensitivity_n_without_train_collisions"] = int(unseen.sum())
        row["sensitivity_f1_without_train_collisions"] = metric_scores(test.label.to_numpy()[unseen], prediction[unseen])["weighted_f1"] if unseen.any() else None
        independent_mask = ~test.row_id.isin(near_train_test).to_numpy()
        row["sensitivity_n_without_near_train_collisions"] = int(independent_mask.sum()) if near_file.exists() else None
        row["sensitivity_f1_without_near_train_collisions"] = metric_scores(test.label.to_numpy()[independent_mask], prediction[independent_mask])["weighted_f1"] if near_file.exists() and independent_mask.any() else None
        evaluation_device = __import__("os").environ.get("DARIJA_EVAL_DEVICE", "cpu")
        if evaluation_device != "cpu" and meta["kind"] != "classical":
            del predictor  # La latence CPU est déjà mesurée ; les perturbations utilisent le GPU demandé.
            predictor = load_predictor(ROOT / meta["checkpoint"], device=evaluation_device)
        perturbation_name = f"f4_{meta['run_id']}_perturbations.csv" if "report_variants" in lock else f"f4_{family}_perturbations.csv"
        robustness = robustness_f4(test, probabilities, predictor, meta["offsets"], filename=perturbation_name)
        row["robustness_f4"] = json.dumps(robustness, ensure_ascii=False)
        separate = [robustness]
        for transformation in ("without_emojis", "without_punctuation"):
            separate.append(robustness_f4(test, probabilities, predictor, meta["offsets"],
                            filename=f"f4_{meta['run_id']}_{transformation}.csv", transformation=transformation))
        pd.DataFrame(separate).to_csv(ROOT / f"results/f4_{meta['run_id']}_summary.csv", index=False)
        save_confusion(test.label, prediction, f"test_{meta['run_id']}")
        if (family, winner["variant"]) in selected_pairs:
            save_confusion(test.label, prediction, f"test_{family}")
        if family == lock["candidate"]["family"] and winner["variant"] == lock["candidate"]["variant"]:
            detail = ROOT / "results" / perturbation_name
            if detail.exists():
                (ROOT / "results/f4_perturbations.csv").write_bytes(detail.read_bytes())
            error_review(test, probabilities, meta["offsets"])
            robustness_review(test, predictor, meta["offsets"])
            evaluate_human_sets(predictor, meta["offsets"])
            save_confusion(test.label, prediction, "modele_final")
        final_rows.append(row)
        del predictor
    final = pd.DataFrame(final_rows)
    if "report_variants" in lock:
        final.to_csv(ROOT / "results/comparison_all_variants.csv", index=False)
        final = final.loc[[(r.family, r.variant) in selected_pairs for r in final.itertuples()]].reset_index(drop=True)
    final.to_csv(ROOT / "results/comparison.csv", index=False)
    measurements = results_table()
    measurements.loc[measurements.family.eq("P") | (measurements.family.eq("E") & measurements.variant.eq("darijabert"))].to_csv(ROOT / "results/cleaning_study.csv", index=False)
    # Test apparié candidat contre le meilleur autre candidat de famille (choisi sur dev).
    available = final.dropna(subset=["run_id"])
    if len(available) >= 2:
        ordered = available.sort_values("dev_weighted_f1_mean", ascending=False)
        a, b = lock.get("mcnemar_runs", ordered.run_id.iloc[:2].tolist())
        ma, pa = test_predictions[a]
        mb, pb = test_predictions[b]
        save_json(ROOT / "results/f5_mcnemar.json", {"first": a, "second": b, **mcnemar(test.label, decide(pa, ma["offsets"]), decide(pb, mb["offsets"]))})
    return final


def interpret(table):
    if table.empty:
        return "Aucune métrique disponible : expérience non exécutée. Aucune conclusion de performance."
    if "weighted_f1" not in table:
        return "Affichage descriptif ; aucune performance prédictive n'est déduite de ce tableau."
    best = table.sort_values("weighted_f1", ascending=False).iloc[0]
    name = best.get("variant", best.get("run_id", "modèle"))
    return f"Résultat observé : {name}, Weighted-F1={best.weighted_f1:.4f}. Interprétation possible : ce score décrit cette partition et cette règle de décision. Limite : comparer les graines, le Macro-F1 et les écritures ; aucun gain causal n'est démontré par ce seul maximum."


def attention_examples(family="C", n=5):
    """Cinq vrais textes dev ; l'attention n'est pas une explication causale."""
    import torch
    from .models import TextNetwork, FrozenSequence, FrozenEncoder
    from .preprocess import encode_texts, tokens
    table = results_table()
    variant = "attention_random" if family == "C" else "states"
    if family == "C" and (ROOT / "campaign.json").exists():
        attention = table.loc[table.family.eq("C") & table.variant.str.startswith("attention_") & table.split.eq("dev") & table.decision.eq("dev_offsets")]
        if not attention.empty:
            variant = attention.groupby("variant").weighted_f1.mean().idxmax()
    rows = table.loc[(table.family == family) & (table.variant == variant) & (table.seed == 42) & (table.split == "dev")]
    if rows.empty:
        return pd.DataFrame(columns=["example", "token", "attention"])
    folder = ROOT / rows.iloc[0].checkpoint
    config = json.loads((folder / "meta.json").read_text(encoding="utf-8"))["config"]
    frame = load_split("dev").head(n)
    if family == "C":
        vocab = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
        model = TextNetwork(len(vocab), "attention")
        from .models import word_inputs
        args = word_inputs(frame.text, vocab, config["max_length"], ROOT / config["fasttext_path"] if config.get("fasttext_path") else None)
        token_lists = [tokens(t)[:config["max_length"]] for t in frame.text]
    else:
        encoder = FrozenEncoder()
        states, mask = encoder.extract(frame.text.tolist(), "states", max_length=config["max_length"])
        model = FrozenSequence(config["hidden_size"])
        token_lists = [encoder.tokenizer.convert_ids_to_tokens(encoder.tokenizer(t, max_length=config["max_length"], truncation=True).input_ids) for t in frame.text]
        args = (torch.tensor(states, dtype=torch.float32), torch.tensor(mask))
    model.load_state_dict(torch.load(folder / "weights.pt", map_location="cpu", weights_only=True))
    model.eval()
    with torch.inference_mode():
        _, weights = model(*args, return_attention=True)
    output = []
    for i, words in enumerate(token_lists):
        values = weights[i, :len(words)].numpy()
        output += [{"example": frame.row_id.iloc[i], "token": word, "attention": float(value)} for word, value in zip(words, values)]
    result = pd.DataFrame(output)
    result.to_csv(ROOT / f"results/attention_{family}_5_examples.csv", index=False)
    fig, axes = plt.subplots(len(frame), 1, figsize=(12, 2.2 * len(frame)), squeeze=False)
    for ax, (example, group) in zip(axes[:, 0], result.groupby("example", sort=False)):
        # Les étiquettes complètes et RTL sont aussi disponibles dans le tableau HTML du notebook.
        top = group.nlargest(12, "attention")
        ax.bar(range(len(top)), top.attention, color="#168a82")
        ax.set(xticks=range(len(top)), xticklabels=top.token, ylabel="Poids", title=example)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle(f"Attention {family} — visualisation descriptive sur cinq exemples dev")
    fig.tight_layout()
    fig.savefig(ROOT / f"results/figures/attention_{family}.png", dpi=140)
    plt.close(fig)
    return result


def plot_histories(family="C"):
    files = sorted((ROOT / "models").glob(f"{family}_*/history.csv"))
    histories = []
    for file in files:
        frame = pd.read_csv(file)
        if {"epoch", "loss", "dev_weighted_f1"} <= set(frame):
            histories.append((file.parent.name, frame))
    if not histories:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, frame in histories:
        if name.endswith("s42"):
            axes[0].plot(frame.epoch, frame.loss, label=name)
            axes[1].plot(frame.epoch, frame.dev_weighted_f1, label=name)
    axes[0].set(xlabel="Époque", ylabel="Loss train", title=f"{family} : apprentissage")
    axes[1].set(xlabel="Époque", ylabel="Weighted-F1 dev", title=f"{family} : validation")
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    path = ROOT / f"results/figures/curves_{family}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def posthoc_diagnostics():
    """Décrit la baisse test après verrouillage ; ne modifie aucun choix ni fichier brut."""
    from .preprocess import build_vocab, oov_table
    assert_locked()
    frames = {s: load_split(s, labels=s == "test") for s in ("train", "dev", "test")}
    rows, markers = [], []
    for split, frame in frames.items():
        for script, group in frame.groupby("script"):
            for label in LABELS:
                rows.append({"split": split, "script": script, "label": label, "n_script": len(group), "count": int(group.label.eq(label).sum()), "proportion": float(group.label.eq(label).mean())})
        markers.append({"split": split, "mentions_or_urls": int(frame.tweet.str.contains(r"@\w+|https?://", regex=True).sum()),
                        "literal_positiveHOTO": int(frame.tweet.str.contains("[[positiveHOTO]]", regex=False).sum()),
                        "literal_COpositive22": int(frame.tweet.str.contains("COpositive22", regex=False).sum())})
    conditional = pd.DataFrame(rows)
    conditional.to_csv(ROOT / "results/f1_conditional_labels.csv", index=False)
    markers = pd.DataFrame(markers)
    markers.to_csv(ROOT / "results/corpus_markers.csv", index=False)
    oov = oov_table(frames, build_vocab(frames["train"].text))
    oov.to_csv(ROOT / "results/oov.csv", index=False)
    neutral = conditional.loc[conditional.label.eq("neutral")].pivot(index="script", columns="split", values="proportion")
    ax = neutral.reindex(columns=["train", "dev", "test"]).plot.bar(color=["#168a82", "#6b7b90", "#c95b5b"])
    ax.set(title="F1 — proportion neutre conditionnelle à l'écriture", ylabel="Proportion neutre", xlabel="Écriture")
    plt.tight_layout()
    plt.savefig(ROOT / "results/figures/F1_conditional_shift.png", dpi=140)
    plt.close()
    lock = assert_locked()
    candidate = lock["candidate"]
    table = results_table()
    candidate_rows = table.loc[table.family.eq(candidate["family"]) & table.variant.eq(candidate["variant"]) & table.seed.eq(42)]
    raw_test = candidate_rows.loc[candidate_rows.split.eq("test") & candidate_rows.decision.eq("argmax"), "weighted_f1"].iloc[0]
    tuned_test = candidate_rows.loc[candidate_rows.split.eq("test") & candidate_rows.decision.eq("dev_offsets"), "weighted_f1"].iloc[0]
    latin_neutral = conditional.loc[conditional.script.eq("latin") & conditional.label.eq("neutral")].set_index("split").proportion
    run = f"{candidate['family']}_{candidate['variant']}_s42"
    predictions = pd.read_csv(ROOT / f"results/{run}_test_tuned.csv")
    predicted_neutral = predictions.loc[predictions.script.eq("latin"), "prediction"].eq("neutral").mean()
    marker_counts = markers.set_index("split").mentions_or_urls
    note = f"""# Analyse a posteriori de la généralisation

Les choix ont été verrouillés avant ces observations. Aucun réglage ne change.

- Pour les textes latins, la classe neutre représente {latin_neutral['train']:.1%} du train, {latin_neutral['dev']:.1%} de dev et {latin_neutral['test']:.1%} du test. La relation écriture/sentiment change davantage que la distribution globale des classes.
- Le candidat classe {predicted_neutral:.1%} des textes latins test comme neutres. Cette prédominance est compatible avec un raccourci appris sur train/dev ; elle ne prouve pas à elle seule son mécanisme causal.
- Le Weighted-F1 test de la règle brute vaut {raw_test:.4f}, celui des décalages choisis sur dev {tuned_test:.4f}. Malgré cela, le résultat principal conserve la règle fixée sur dev. Choisir la règle brute parce qu'elle gagne sur test serait une nouvelle sélection sur test.
- Les taux OOV sont enregistrés dans oov.csv : ils décrivent le vocabulaire de mots min_freq=2, pas directement la couverture des n-grammes du candidat TF-IDF caractères. Ils ne suffisent pas à expliquer causalement sa baisse.
- Les mentions/URLs concernent {marker_counts['train']} tweet train, {marker_counts['dev']} tweet dev et {marker_counts['test']} tweets test. Le remplacement conservateur par USER/URL crée donc des formes presque absentes de l'apprentissage. Une influence sur les prédictions est plausible, mais aucune ablation causale de ces marqueurs n'a été exécutée ; la règle de nettoyage reste celle fixée avant test.
- Des marqueurs littéraux anormaux, dont [[positiveHOTO]] et COpositive22, apparaissent dans les fichiers bruts officiels (effectifs dans corpus_markers.csv). Leur origine n'est pas démontrée par cet audit et leur présence ne prouve pas une fuite exploitable. Les données officielles restent intactes.
- Les taux de perturbation F4 de familles différentes sont conditionnés à des sous-ensembles initiaux différents. Éviter de les interpréter comme un classement causal de robustesse sur une population commune.

Hypothèses pour une future étude indépendante : contrôler les raccourcis d'écriture, annoter les domaines/cibles, examiner la normalisation des mentions et les marqueurs de corpus, puis valider ces interventions sans réutiliser ce test pour les choisir. Aucune de ces interventions n'est déclarée efficace ici.
"""
    (ROOT / "results/generalization_analysis.md").write_text(note, encoding="utf-8")
    return conditional, markers, oov
