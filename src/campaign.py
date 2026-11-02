"""Campagne A–G v2 : aucune expérience n'est lancée à l'import."""
import gc
import json
import time
import urllib.request
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .preprocess import (ROOT, PROJECT_ROOT, CLEANING, LABELS, COMMIT, BASE_URL, TEXT_MODES,
                         load_split, clean_text, script_type, tokens, build_vocab,
                         oov_table, save_json, sha256)
from .evaluate import (results_table, metric_scores, decide, select_on_dev,
                       lock_selection, assert_locked, final_evaluation)
from .train import (SEEDS, classical_run, neural_run, pretrained_run, frozen_run,
                    finetune_run, load_predictor)
from .models import BERT_ID
from .resources import download_atomic

ARQ_TRAIN_SHA256 = "edeaea0e5e57564428b1b4a17a4429381b3f2fa933000d2a6dddf83573e57745"

def default_config():
    return {"seeds": list(SEEDS), "device": "cuda" if torch.cuda.is_available() else "cpu",
            "max_length": 128, "char_length": 512, "c_epochs": 15, "d_epochs": 10,
            "e_epochs": 5, "micro_batch_e": 2, "c_grid_a": [0.25, 1.0, 4.0],
            "lambdas": [0.1, 0.3, 0.7], "include_e2": True,
            "include_cleaning_study": True,
            "aravec_path": str(PROJECT_ROOT / "models/resources/aravec/full_uni_cbow_100_twitter.mdl")}


def prepare(config, dry_run=False):
    """Fige code/config ; les sorties historiques ne sont jamais réutilisées."""
    if ROOT == PROJECT_ROOT or CLEANING != "conservative_v2":
        raise RuntimeError("Configurer DARIJA_OUTPUT_ROOT et DARIJA_CLEANING avant d'importer src ; redémarrer le kernel.")
    unknown = set(config) - set(default_config())
    if unknown:
        raise ValueError("Options CONFIG inconnues : " + ", ".join(sorted(unknown)))
    if config["device"] not in {"cpu", "cuda"}:
        raise ValueError("DEVICE doit être 'auto', 'cpu' ou 'cuda' dans la première cellule")
    for key in ("max_length", "char_length", "c_epochs", "d_epochs", "e_epochs", "micro_batch_e"):
        if isinstance(config[key], bool) or not isinstance(config[key], int):
            raise ValueError(f"CONFIG[{key!r}] doit être un entier")
    if not config["c_grid_a"] or len(set(config["c_grid_a"])) != len(config["c_grid_a"]) or any(not np.isfinite(value) or value <= 0 for value in config["c_grid_a"]):
        raise ValueError("Les régularisations C doivent être positives, finies et distinctes")
    if config["include_e2"] and not config["lambdas"]:
        raise ValueError("E2 nécessite au moins un lambda auxiliaire positif")
    if config["seeds"] != list(SEEDS) or not 3 <= config["e_epochs"] <= 5:
        raise ValueError("Trois graines 42/43/44 et 3 à 5 époques E sont requises")
    if config["micro_batch_e"] < 1 or 32 % config["micro_batch_e"]:
        raise ValueError("micro_batch_e doit diviser 32")
    if not 1 <= config["max_length"] <= 512 or config["char_length"] < 7 or min(config["c_epochs"], config["d_epochs"]) < 1:
        raise ValueError("Longueurs ou budgets d'époques invalides")
    if len(set(config["lambdas"])) != len(config["lambdas"]) or any(value <= 0 for value in config["lambdas"]):
        raise ValueError("Les lambdas E2 doivent être positifs et distincts ; lambda=0 est la référence E1")
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA demandé mais indisponible dans ce kernel")
    if int(__import__("os").environ.get("DARIJA_MAX_LENGTH", "64")) != config["max_length"]:
        raise ValueError("DARIJA_MAX_LENGTH et config.max_length doivent correspondre")
    manifest = {"config": config, "cleaning": CLEANING, "data_commit": COMMIT,
                "code_sha256": {f.name: sha256(f) for f in (PROJECT_ROOT / "src").glob("*.py")},
                "test_status": "test ary déjà observé lors de la campagne historique ; réévaluation exploratoire"}
    path = ROOT / "campaign.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("Code/config modifiés depuis le début : choisir un nouveau CAMPAIGN_NAME, sans écraser cette campagne.")
    if dry_run:
        return manifest
    for folder in ("models", "results/figures", "results/audit", "results/cache"):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    (ROOT / "human_review").mkdir(exist_ok=True)
    for source in (PROJECT_ROOT / "data/review").glob("*"):
        target = ROOT / "human_review" / source.name
        if source.is_file() and not target.exists():
            shutil.copy2(source, target)
    save_json(path, manifest)
    return manifest


def auxiliary_train():
    """Télécharge uniquement arq/train à la révision officielle. Aucun test arq."""
    path = PROJECT_ROOT / "data/arq/train.tsv"
    provenance = PROJECT_ROOT / "data/arq/provenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    url = BASE_URL + "data/arq/train.tsv"
    if not path.exists() or not provenance.exists():
        if path.exists() and sha256(path) != ARQ_TRAIN_SHA256:
            raise ValueError("Le train arq local diffère de la révision figée ; aucune substitution automatique")
        if not path.exists():
            download_atomic(url, path, ARQ_TRAIN_SHA256)
        save_json(provenance, {"url": url, "commit": COMMIT, "sha256": ARQ_TRAIN_SHA256})
    origin = json.loads(provenance.read_text(encoding="utf-8"))
    if origin["commit"] != COMMIT or sha256(path) != origin["sha256"] or sha256(path) != ARQ_TRAIN_SHA256:
        raise ValueError("Intégrité arq incorrecte")
    frame = pd.read_csv(path, sep="\t", keep_default_na=False)
    if not {"tweet", "label"} <= set(frame) or not frame.label.isin(LABELS).all() or frame.tweet.str.strip().eq("").any():
        raise ValueError("Format/labels arq invalides")
    frame["text"] = frame.tweet.map(clean_text).replace("", "<EMPTY>")
    frame["row_id"] = [f"arq:train:{i}" for i in range(len(frame))]
    frame["script"] = frame.tweet.map(script_type)
    # Textes seulement, jamais labels dev/test ary : retire les collisions inter-tâches.
    reserved = set(pd.concat([load_split(s).text for s in ("train", "dev", "test")]))
    original = len(frame)
    conflicts = frame.groupby("text").label.nunique()
    ambiguous = set(conflicts[conflicts > 1].index)
    frame = frame.loc[~frame.text.isin(reserved | ambiguous)].drop_duplicates("text").reset_index(drop=True)
    if set(frame.label) != set(LABELS):
        raise ValueError("arq filtré doit conserver les trois classes")
    save_json(ROOT / "results/arq_audit.json", {**origin, "n_original": original, "n_used": len(frame),
              "removed_collisions_duplicates_or_conflicts": original - len(frame), "conflicting_texts_removed": len(ambiguous), "test_arq_used": False})
    return frame


def jobs(part, config):
    """Liste prévisible avant exécution ; variantes dépendantes choisies sur dev seulement."""
    out = []
    def add(run, function, *args, **kwargs):
        out.append((run, function, args, kwargs))
    if part == "A":
        for seed in (42,):  # LR et majorité déterministes : une exécution suffit.
            for kind in ("majority", "word", "char", "union"):
                add(f"A_{kind}_s{seed}", classical_run, kind, seed, preserve_symbols=True)
            for value in config["c_grid_a"]:
                if value != 1:
                    tag = f"_C{value:g}"
                    add(f"A_union{tag}_s{seed}", classical_run, "union", seed, C=value, preserve_symbols=True, tag=tag)
            for script in ("arabic", "latin"):
                add(f"A_union_train_{script}_s{seed}", classical_run, "union", seed, train_script=script, preserve_symbols=True)
    elif part == "B":
        for language in ("ar", "fr", "en"):
            add(f"B_nli_{language}_s42", pretrained_run, "nli", language, device=config["device"])
        add("B_camelbert_s42", pretrained_run, "camel", device=config["device"])
    elif part == "C":
        for seed in SEEDS:
            for kind in ("bilstm", "attention", "transformer"):
                for embedding in ("random", "fasttext", "aravec"):
                    for positional in ((True, False) if kind == "transformer" else (True,)):
                        variant = f"{kind}_{embedding}" + ("_nopos" if not positional else "")
                        add(f"C_{variant}_s{seed}", neural_run, kind, embedding, seed, positional=positional,
                            aravec_path=config["aravec_path"], epochs=config["c_epochs"], device=config["device"],
                            max_length=config["max_length"], open_fasttext=True)
            add(f"C_charcnn_random_s{seed}", neural_run, "charcnn", "random", seed,
                epochs=config["c_epochs"], device=config["device"], max_length=config["char_length"])
    elif part == "C_balance":
        path = ROOT / "results/c_imbalance_selection.json"
        if not path.exists():
            raise RuntimeError("Terminer C puis appeler choose_c_for_balance()")
        selected = json.loads(path.read_text(encoding="utf-8"))
        for seed in SEEDS:
            for mode in ("weighted", "oversampling"):
                add(f"C_{selected['variant']}_{mode}_s{seed}", neural_run,
                    selected["config"]["kind"], selected["config"]["embedding"], seed,
                    positional=selected["config"]["positional"], weighted_loss=mode == "weighted",
                    oversampling=mode == "oversampling", aravec_path=config["aravec_path"],
                    epochs=config["c_epochs"], device=config["device"], max_length=selected["config"]["max_length"],
                    open_fasttext=selected["config"].get("open_fasttext", False))
    elif part == "D":
        for pooling in ("cls", "mean", "states"):
            for seed in (SEEDS if pooling == "states" else (42,)):
                add(f"D_{pooling}_s{seed}", frozen_run, pooling, seed, device=config["device"], epochs=config["d_epochs"])
    elif part in {"E1", "E2"}:
        if part == "E1":
            models = [(BERT_ID, "darijabert"), ("SI2M-Lab/DarijaBERT-arabizi", "arabizi")]
            weights = [0.0]
        else:
            path = ROOT / "results/e2_encoder_selection.json"
            if not path.exists():
                raise RuntimeError("Terminer E1 puis appeler choose_e2_encoder()")
            selected = json.loads(path.read_text(encoding="utf-8"))
            models = [(selected["model_id"], selected["variant"])]
            weights = config["lambdas"]
        for model_id, variant in models:
            for value in weights:
                for seed in SEEDS:
                    suffix = f"_mtl_l{value:g}" if value else ""
                    add(f"E_{variant}{suffix}_s{seed}", finetune_run, model_id, seed,
                        micro_batch=config["micro_batch_e"], device=config["device"], epochs=config["e_epochs"],
                        max_length=config["max_length"], auxiliary_weight=value)
    elif part == "P":
        if not config["include_cleaning_study"]:
            return out
        for mode in (*TEXT_MODES, "emoji_only", "punctuation_only"):
            add(f"P_union_{mode}_s42", classical_run, "union", 42,
                family="P", tag=f"_{mode}", cleaning=mode, preserve_symbols=True,
                purpose="shortcut" if mode.endswith("_only") else "benchmark")
        # La référence conservatrice est E_darijabert_s42/43/44 : pas de réentraînement doublon.
        for mode in TEXT_MODES[1:]:
            for seed in SEEDS:
                add(f"P_darijabert_{mode}_s{seed}", finetune_run, BERT_ID, seed,
                    family="P", tag=f"_{mode}", cleaning=mode, micro_batch=config["micro_batch_e"],
                    device=config["device"], epochs=config["e_epochs"], max_length=config["max_length"])
    else:
        raise ValueError(part)
    return out


def complete(run):
    meta_file = ROOT / "models" / run / "meta.json"
    if not meta_file.exists() or (meta_file.parent / ".in_progress").exists():
        return False
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    folder = meta_file.parent
    required = "classifier.joblib" if meta["kind"] == "classical" or (meta["kind"] == "frozen" and meta["config"]["pooling"] != "states") else "pretrained.json" if meta["kind"] == "pretrained" else "weights.pt"
    signatures = folder / "artifact_sha256.json"
    if signatures.exists():
        for filename, digest in json.loads(signatures.read_text(encoding="utf-8")).items():
            if not (folder / filename).exists() or sha256(folder / filename) != digest:
                raise RuntimeError(f"Artefact de run modifié : {run}/{filename}")
    rows = results_table()
    if not (folder / required).exists():
        raise RuntimeError(f"Checkpoint manquant : {run}. Restaurer ses fichiers ou utiliser une nouvelle campagne.")
    return (not rows.empty and len(rows.loc[rows.run_id.eq(run) & rows.split.eq("dev")]) == 2
            and all((ROOT / f"results/{run}_dev_{decision}.csv").exists() for decision in ("raw", "tuned")))


def run_part(part, config):
    prepare(config)
    planned = jobs(part, config)
    if (ROOT / "results/selection_lock.json").exists():
        if all(complete(run) for run, *_ in planned):
            print(f"{part} déjà terminé ; campagne verrouillée")
            return
        raise RuntimeError("Campagne verrouillée : entraînement interdit")
    if any(kw.get("aravec_path") and args[1:2] == ("aravec",) for _, _, args, kw in planned) and not Path(config["aravec_path"]).exists():
        raise FileNotFoundError("AraVec requis pour C : renseigner CONFIG['aravec_path'] avant de démarrer la campagne")
    auxiliary = auxiliary_train() if part == "E2" else None
    for index, (run, function, args, kwargs) in enumerate(planned, 1):
        if complete(run):
            print(f"[{index}/{len(planned)}] {run} déjà terminé")
            continue
        print(f"[{index}/{len(planned)}] {run}", flush=True)
        if part == "E2":
            kwargs["auxiliary"] = auxiliary
        function(*args, **kwargs)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    status(config).to_csv(ROOT / "results/execution_status.csv", index=False)


def choose_c_for_balance():
    ranking = select_on_dev().query("family == 'C'")
    ranking = ranking.loc[~ranking.variant.str.endswith(("_weighted", "_oversampling"))]
    winner = ranking.iloc[0]
    meta = json.loads((ROOT / f"models/C_{winner.variant}_s42/meta.json").read_text(encoding="utf-8"))
    selected = {"variant": winner.variant, "config": meta["config"], "criterion": "mean dev weighted F1 over three seeds"}
    save_json(ROOT / "results/c_imbalance_selection.json", selected)
    return ranking


def choose_e2_encoder():
    ranking = select_on_dev().query("family == 'E'")
    ranking = ranking.loc[~ranking.variant.str.contains("_mtl_")]
    winner = ranking.iloc[0]
    meta = json.loads((ROOT / f"models/E_{winner.variant}_s42/meta.json").read_text(encoding="utf-8"))
    selected = {"variant": winner.variant, "model_id": meta["config"]["model_id"], "lambda_zero_reference": winner.variant}
    save_json(ROOT / "results/e2_encoder_selection.json", selected)
    return ranking


def status(config):
    rows = []
    for part in ("A", "B", "C", "C_balance", "D", "E1", "E2", "P"):
        if part == "E2" and not config["include_e2"]:
            continue
        if part == "P" and not config["include_cleaning_study"]:
            continue
        try:
            planned = jobs(part, config)
        except RuntimeError:
            rows.append({"part": part, "run_id": "choix dev en attente", "status": "non exécuté"})
            continue
        rows.extend({"part": part, "run_id": run, "status": "exécuté" if complete(run) else "non exécuté"} for run, *_ in planned)
    return pd.DataFrame(rows)


def finalize(config):
    prepare(config)
    table = status(config)
    if not table.status.eq("exécuté").all():
        raise RuntimeError("Terminer toutes les parties avant d'ouvrir les labels test : " + ", ".join(table.loc[table.status.ne("exécuté"), "run_id"].head(8)))
    lock = lock_selection()
    if "test_runs" not in lock:
        # Toutes les comparaisons demandées sont déclarées avant toute nouvelle métrique test.
        lock["test_runs"] = table.run_id.tolist()
        ranking = select_on_dev()
        lock["report_variants"] = ranking.astype(object).where(pd.notna(ranking), None).to_dict("records")
        lock["mcnemar_runs"] = [f"{row.family}_{row.variant}_s42" for row in select_on_dev().head(2).itertuples()]
        cache = ROOT / "results/cache"
        lock["shared_artifacts"] = {str(p.relative_to(ROOT)): sha256(p) for p in cache.glob("fasttext*") if p.is_file()}
        save_json(ROOT / "results/selection_lock.json", lock)
    for path, digest in lock.get("shared_artifacts", {}).items():
        if sha256(ROOT / path) != digest:
            raise RuntimeError(f"Ressource fastText modifiée : {path}")
    return final_evaluation()


def uncertainty_table():
    """IC de chaque variante représentative ; variabilité des graines séparée."""
    from .evaluate import bootstrap_ci
    assert_locked()
    complete_table = ROOT / "results/comparison_all_variants.csv"
    if complete_table.exists():
        result = pd.read_csv(complete_table)[["family", "variant", "n_seeds", "test_seed_mean", "test_seed_std", "weighted_f1", "ci_low", "ci_high", "bootstrap"]]
        result.to_csv(ROOT / "results/uncertainty_all_variants.csv", index=False)
        return result
    metrics = results_table()
    metrics = metrics.loc[metrics.split.eq("test") & metrics.decision.eq("dev_offsets") & metrics.purpose.eq("benchmark")]
    rows = []
    for (family, variant), group in metrics.groupby(["family", "variant"]):
        run = f"{family}_{variant}_s42"
        frame = pd.read_csv(ROOT / f"results/{run}_test_tuned.csv")
        rows.append({"family": family, "variant": variant, "n_seeds": len(group), "seed_mean": group.weighted_f1.mean(),
                     "seed_std": group.weighted_f1.std(), **bootstrap_ci(frame.label, frame.prediction, n_bootstrap=1000)})
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "results/uncertainty_all_variants.csv", index=False)
    return result


def distributions(include_test=False):
    rows = []
    for split in (("train", "dev", "test") if include_test else ("train", "dev")):
        frame = load_split(split, labels=split == "test")
        for script, part in [("global", frame), *frame.groupby("script")]:
            for label in LABELS:
                rows.append({"split": split, "script": script, "label": label, "n": int(part.label.eq(label).sum()), "proportion": float(part.label.eq(label).mean())})
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "results/label_distributions.csv", index=False)
    return result


def shortcuts(include_test=False):
    train = load_split("train")
    rules = train.groupby("script").label.agg(lambda s: s.value_counts().idxmax()).to_dict()
    majority = train.label.mode().iloc[0]
    rows = []
    for split in (("dev", "test") if include_test else ("dev",)):
        frame = load_split(split, labels=split == "test")
        for name, prediction in [("majority", np.repeat(majority, len(frame))), ("script_only", frame.script.map(rules).fillna(majority))]:
            rows.append({"split": split, "rule": name, **metric_scores(frame.label, prediction)})
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "results/shortcut_baselines.csv", index=False)
    return result


def embedding_oov(config, include_test=False):
    """Sépare vocabulaire du réseau, lexique appris et composition sous-mots."""
    from gensim.models import FastText, Word2Vec
    train = load_split("train")
    vocab = build_vocab(train.text, min_freq=2)
    frames = {s: load_split(s) for s in (("train", "dev", "test") if include_test else ("train", "dev"))}
    oov_table(frames, vocab).to_csv(ROOT / "results/oov.csv", index=False)
    aravec = Word2Vec.load(config["aravec_path"], mmap="r").wv
    cache_files = sorted((ROOT / "results/cache").glob("fasttext_s42_*.model"))
    if len(cache_files) != 1:
        raise RuntimeError("Terminer C/fastText graine42 avant la comparaison de couverture")
    fasttext = FastText.load(str(cache_files[0])).wv
    rows = []
    for split, frame in frames.items():
        for script, group in frame.groupby("script"):
            words = [w for text in group.text for w in tokens(text)]
            for method, vectors in (("AraVec", aravec), ("fastText", fasttext)):
                lexical = sum(w not in vectors.key_to_index for w in words)
                composed = sum(w not in vectors.key_to_index and np.isfinite(vectors[w]).all() and np.linalg.norm(vectors[w]) > 0 for w in set(words)) if method == "fastText" else 0
                usable_missing = sum(not np.isfinite(vectors[w]).all() or np.linalg.norm(vectors[w]) == 0 for w in words) if method == "fastText" else lexical
                rows.append({"split": split, "script": script, "embedding": method, "tokens": len(words),
                             "lexical_oov_rate": lexical / max(1, len(words)), "unusable_vector_rate": usable_missing / max(1, len(words)),
                             "distinct_oov_with_subword_vector": composed, "network_vocab_oov_rate": sum(w not in vocab for w in words) / max(1, len(words))})
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "results/embedding_oov_by_script.csv", index=False)
    return result


def threshold_diagnostic():
    """Balayage post hoc, sauvegardé séparément ; aucune règle n'est réécrite."""
    from itertools import product
    lock = assert_locked()
    candidate = lock["candidate"]
    run = f"{candidate['family']}_{candidate['variant']}_s42"
    meta_path = ROOT / f"models/{run}/meta.json"
    before = sha256(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frames = {s: pd.read_csv(ROOT / f"results/{run}_{s}_raw.csv") for s in ("dev", "test")}
    rows = []
    for neutral, positive in product(np.linspace(-1.5, 1.5, 21), repeat=2):
        offset = [0.0, float(neutral), float(positive)]
        row = {"neutral_offset": neutral, "positive_offset": positive}
        for split, frame in frames.items():
            row[f"{split}_weighted_f1"] = metric_scores(frame.label, decide(frame[[f"p_{y}" for y in LABELS]].to_numpy(), offset))["weighted_f1"]
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "results/f1_posthoc_grid.csv", index=False)
    optimum = result.loc[result.test_weighted_f1.idxmax()].to_dict()
    frozen = {s: metric_scores(f.label, decide(f[[f"p_{y}" for y in LABELS]].to_numpy(), meta["offsets"]))["weighted_f1"] for s, f in frames.items()}
    save_json(ROOT / "results/f1_posthoc_oracle.json", {"run": run, "frozen_offsets": meta["offsets"], "frozen_scores": frozen,
              "posthoc_gain_test_points": 100 * (optimum["test_weighted_f1"] - frozen["test"]), "grid_test_optimum": optimum,
              "scope": "diagnostic optimiste sur grille finie ; pas un optimum continu ni une règle déployable"})
    assert sha256(meta_path) == before
    return result


def error_review_summary():
    """La classification interprétative reste explicitement à valider par le lecteur."""
    path = ROOT / "results/f6_30_erreurs.csv"
    frame = pd.read_csv(path, keep_default_na=False)
    valid = frame.validation_humaine.eq("validé")
    return pd.DataFrame({"total": [len(frame)], "validées": [int(valid.sum())], "à_revoir": [int((~valid).sum())],
                         "note": ["Désaccord de votes ≠ preuve d'erreur d'annotation ; cet échantillon diversifié ne donne pas la fréquence globale des catégories."]})
