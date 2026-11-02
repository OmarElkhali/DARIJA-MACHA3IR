"""Affichage des temps cumulés et vérifications réutilisables, sans modifier la campagne."""
from pathlib import Path
import json


def install_storage_writes():
    """Un temporaire unique par écriture ; conserver les octets des fichiers inchangés."""
    import sys
    import tempfile
    import time
    from src import preprocess
    original = preprocess.save_json
    if getattr(original, "_storage_safe", False):
        return

    def save_json(path, value):
        path = Path(path)
        payload = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        if path.is_file() and path.read_text(encoding="utf-8") == payload:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        try:
            for attempt in range(5):
                try:
                    temporary.replace(path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.02 * (attempt + 1))  # Windows : attendre la fin d'une lecture concurrente.
        finally:
            temporary.unlink(missing_ok=True)

    save_json._storage_safe = True
    for name, module in list(sys.modules.items()):
        if (name == "src" or name.startswith("src.")) and getattr(module, "save_json", None) is original:
            module.save_json = save_json


def install_resume_checks(campaign):
    """Relire les poids une fois ; invalider dès qu'un fichier surveillé change."""
    if getattr(campaign.complete, "_resume_cache", False):
        return
    from src import preprocess
    original_hash = preprocess.file_sha256
    hashes = {}

    def cached_hash(path):
        path = Path(path)
        stat = path.stat()
        key = str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
        if key not in hashes:
            hashes[key] = original_hash(path)
        return hashes[key]

    # Toutes les vérifications SHA-256 de campagne passent par cette fonction commune.
    preprocess.file_sha256 = cached_hash
    original = campaign.complete
    cache = {}

    def complete(run):
        folder = Path(campaign.ROOT) / "models" / run
        manifest = folder / "artifact_sha256.json"
        if not manifest.exists() or (folder / ".in_progress").exists():
            return original(run)
        paths = [manifest, folder / "meta.json", folder / "weights.pt",
                 folder / "classifier.joblib", folder / "pretrained.json",
                 Path(campaign.ROOT) / "results/results.csv"]
        paths += [folder / name for name in json.loads(manifest.read_text(encoding="utf-8"))]
        paths += [Path(campaign.ROOT) / f"results/{run}_dev_{decision}.csv" for decision in ("raw", "tuned")]
        def stamp(path):
            try:
                stat = path.stat()
                return str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
            except FileNotFoundError:
                return str(path), None
        key = tuple(stamp(path) for path in paths)
        if cache.get(run, (None,))[0] == key:
            return cache[run][1]
        print("Vérification des sauvegardes :", run, flush=True)
        result = original(run)
        if result:
            cache[run] = key, result
        return result

    complete._resume_cache = True
    campaign.complete = complete


def training_times(root, table):
    """Une seule ligne par run : les lignes raw/tuned/test ne sont pas additionnées."""
    import pandas as pd
    root = Path(root)
    rows = []
    if table.empty:
        return pd.DataFrame()
    completed = table.loc[table.split.eq("dev") & table.decision.eq("dev_offsets")].drop_duplicates("run_id")
    for row in completed.itertuples():
        path = root / "models" / row.run_id / "meta.json"
        if not path.exists():
            continue
        meta = json.loads(path.read_text(encoding="utf-8"))
        total = float(meta["train_seconds"])
        previous = float(meta["config"].get("resumed_epoch_seconds", 0.0))
        current = total - previous
        if current < -0.01:
            raise ValueError(f"Temps de reprise incohérents : {row.run_id}")
        def duration(seconds):
            return str(pd.Timedelta(seconds=round(seconds)))
        rows.append({"run_id": row.run_id, "family": row.family, "variant": row.variant,
                     "avant_reprise_s": previous, "derniere_session_s": current,
                     "total_entrainement_s": total,
                     "duree_cumulee": f"{duration(previous)} + {duration(current)} = {duration(total)}"})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.to_csv(root / "results/training_times_cumulative.csv", index=False)
    return frame


def show_training_times(namespace):
    frame = training_times(namespace["ROOT"], namespace["results_table"]())
    if frame.empty:
        print("Aucun entraînement terminé à afficher.")
        return frame
    display = namespace["display"]
    display(frame.loc[frame.family.isin(["E", "P"])])
    totals = frame.groupby("family", as_index=False).agg(
        runs=("run_id", "nunique"), total_entrainement_s=("total_entrainement_s", "sum"))
    totals["total_minutes"] = totals.total_entrainement_s / 60
    display(totals)
    print("Temps mesurés cumulés, sans compter deux fois les résultats raw/tuned ou les sessions reprises.")
    print("Les époques perdues avant un checkpoint et les coupures réseau ne sont pas mesurables ici.")
    return frame
