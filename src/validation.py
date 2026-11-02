"""Audits de proximité, groupes et évaluations humaines séparées du benchmark."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_recall_fscore_support
from .preprocess import ROOT, PROJECT_ROOT, LABELS, clean_text, emoji_parts, load_split, sha256


def near_duplicates(reference, queries, threshold=0.9):
    """Plus proche voisin lexical ; seuil fixé avant évaluation, pas preuve de paraphrase."""
    if not 0 < threshold <= 1:
        raise ValueError("Le seuil de proximité doit appartenir à ]0,1].")
    columns = ["row_id", "reference_row_id", "similarity", "text", "reference_text"]
    if reference.empty or queries.empty:
        return pd.DataFrame(columns=columns)
    normalize = lambda text: clean_text(text, "without_both").lower() or "<EMPTY>"
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=100000)
    matrix = vectorizer.fit_transform(reference.text.map(normalize))
    neighbors = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute").fit(matrix)
    rows = []
    for start in range(0, len(queries), 256):
        part = queries.iloc[start:start + 256]
        distance, index = neighbors.kneighbors(vectorizer.transform(part.text.map(normalize)))
        for query, similarity, position in zip(part.itertuples(), 1 - distance[:, 0], index[:, 0]):
            if similarity + 1e-10 >= threshold:
                other = reference.iloc[position]
                rows.append(dict(zip(columns, [query.row_id, other.row_id, float(similarity), query.text, other.text])))
    return pd.DataFrame(rows, columns=columns)


def audit_near_duplicates():
    frames = {split: load_split(split) for split in ("train", "dev", "test")}
    parts = []
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        parts.append(near_duplicates(frames[a], frames[b]).assign(reference_split=a, query_split=b))
    result = pd.concat(parts, ignore_index=True)
    folder = ROOT / "results/audit"
    folder.mkdir(parents=True, exist_ok=True)
    result.to_csv(folder / "near_duplicates.csv", index=False)
    return result


def subgroup_scores(frame, probabilities, offsets):
    from .evaluate import decide, metric_scores
    prediction = decide(probabilities, offsets)
    has_emoji = frame.text.map(lambda text: bool(emoji_parts(text))).to_numpy()
    groups = [("global", "all", np.ones(len(frame), dtype=bool))]
    groups += [("script", script, frame.script.eq(script).to_numpy()) for script in frame.script.unique()]
    groups += [("emoji", value, has_emoji if value == "present" else ~has_emoji) for value in ("present", "absent")]
    rows = [{"group": group, "value": value, "n": int(mask.sum()),
             **metric_scores(frame.label.to_numpy()[mask], prediction[mask])}
            for group, value, mask in groups if mask.any()]
    precision, recall, f1, support = precision_recall_fscore_support(frame.label, prediction, labels=LABELS, zero_division=0)
    rows += [{"group": "class", "value": label, "n": int(n), "precision": float(p), "recall": float(r), "f1": float(f)}
             for label, p, r, f, n in zip(LABELS, precision, recall, f1, support)]
    return pd.DataFrame(rows)


def robustness_review(frame, predictor, offsets):
    from .evaluate import decide, metric_scores, assert_locked
    assert_locked()
    path = ROOT / "results/f4_human_review.csv"
    rows = []
    baseline = decide(predictor(frame.text.tolist()), offsets)
    for transformation in ("without_emojis", "without_punctuation", "without_both"):
        changed = frame.text.map(lambda text: clean_text(text, transformation) or "<EMPTY>")
        mask = changed.ne(frame.text)
        predictions = decide(predictor(changed.loc[mask].tolist()), offsets) if mask.any() else []
        for item, text, before, after in zip(frame.loc[mask].itertuples(), changed.loc[mask], baseline[mask], predictions):
            rows.append({"row_id": item.row_id, "transformation": transformation, "text": item.text,
                         "changed_text": text, "original_label": item.label, "baseline_prediction": before,
                         "altered_prediction": after, "sentiment_preserved": "", "human_label": "",
                         "validation_humaine": "à faire", "reviewer": "", "comment": ""})
    columns = ["row_id", "transformation", "text", "changed_text", "original_label", "baseline_prediction", "altered_prediction",
               "sentiment_preserved", "human_label", "validation_humaine", "reviewer", "comment"]
    review = pd.DataFrame(rows, columns=columns)
    keys = ["row_id", "transformation", "text", "changed_text"]
    if path.exists():
        existing = pd.read_csv(path, keep_default_na=False, dtype=str).set_index(keys)
        if not existing.index.is_unique:
            raise ValueError("Lignes dupliquées dans la revue F4 ; corriger le CSV.")
        for column in ("sentiment_preserved", "human_label", "validation_humaine", "reviewer", "comment"):
            saved = pd.Series(pd.MultiIndex.from_frame(review[keys]).map(existing[column]), index=review.index)
            review[column] = saved.where(saved.notna(), review[column])
    review.to_csv(path, index=False)
    reports = []
    for transformation, part in review.groupby("transformation"):
        valid = (part.validation_humaine.eq("validé") & part.reviewer.str.strip().ne("") & part.human_label.isin(LABELS)
                 & part.sentiment_preserved.isin(["oui", "non", "incertain"]))
        if (valid & part.sentiment_preserved.eq("oui") & part.human_label.ne(part.original_label)).any():
            raise ValueError("Revue contradictoire : sentiment déclaré conservé mais label humain différent.")
        selected = part.loc[valid]
        reports.append({"transformation": transformation, "n_reviewed": len(selected), "n_pending": int((~valid).sum()),
                        **(metric_scores(selected.human_label, selected.altered_prediction) if len(selected) else {}),
                        "status": "évalué sur labels humains" if len(selected) else "en attente de validation humaine"})
    report = pd.DataFrame(reports) if reports else pd.DataFrame(columns=["transformation", "n_reviewed", "n_pending", "status"])
    report.to_csv(ROOT / "results/f4_human_scores.csv", index=False)
    return report


def evaluate_human_sets(predictor, offsets, review_dir=None):
    from .evaluate import assert_locked, decide, metric_scores
    assert_locked()
    directory = Path(review_dir) if review_dir else ROOT / "human_review"
    reports = []
    for name in ("behavioral_cases", "independent"):
        path = directory / f"{name}.csv"
        if not path.exists():
            reports.append({"set": name, "n": 0, "status": "fichier absent"})
            continue
        frame = pd.read_csv(path, keep_default_na=False, dtype=str)
        required = {"text", "label", "reviewer", "validation_humaine", "source", "collected_at"}
        if not required <= set(frame):
            raise ValueError(f"Colonnes manquantes dans {path}: {required - set(frame)}")
        valid = frame.validation_humaine.eq("validé") & frame.label.isin(LABELS) & frame.reviewer.str.strip().ne("")
        selected = frame.loc[valid].copy()
        if selected.empty:
            reports.append({"set": name, "n": 0, "status": "en attente de collecte/annotation humaine", "pending": len(frame)})
            continue
        if selected.text.str.strip().eq("").any():
            raise ValueError(f"Texte vide dans {path}")
        selected["text"] = selected.text.map(lambda text: clean_text(text, "conservative_v2") or "<EMPTY>")
        if name == "independent":
            if selected.source.str.strip().eq("").any() or pd.to_datetime(selected.collected_at, errors="coerce", utc=True).isna().any():
                raise ValueError("L'ensemble indépendant exige une source et une date de collecte valides par exemple.")
            official = pd.concat([load_split(s) for s in ("train", "dev", "test")], ignore_index=True)
            selected["row_id"] = [f"independent:{i}" for i in selected.index]
            overlaps = near_duplicates(official, selected)
            overlaps.to_csv(ROOT / "results/independent_overlap_audit.csv", index=False)
            if len(overlaps) or selected.text.duplicated().any():
                raise ValueError("Exemples indépendants dupliqués/proches du benchmark : examiner l'audit, ne pas publier ce score.")
        prediction = decide(predictor(selected.text.tolist()), offsets)
        selected["prediction"] = prediction
        selected.to_csv(ROOT / f"results/{name}_predictions.csv", index=False)
        reports.append({"set": name, "n": len(selected), "status": "évalué après verrouillage", **metric_scores(selected.label, prediction)})
        if "category" in selected:
            for category, part in selected.groupby("category"):
                reports.append({"set": name, "category": category, "n": len(part), "status": "évalué après verrouillage", **metric_scores(part.label, part.prediction)})
    report = pd.DataFrame(reports)
    report["source_sha256"] = report["set"].map({name: sha256(directory / f"{name}.csv") for name in ("behavioral_cases", "independent") if (directory / f"{name}.csv").exists()})
    report.to_csv(ROOT / "results/human_test_scores.csv", index=False)
    return report
