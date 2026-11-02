"""Bloc de configuration inséré dans le notebook après l'amorçage stdlib."""
import json, html, importlib

os.environ["DARIJA_OUTPUT_ROOT"] = str(OUTPUT)
os.environ["DARIJA_CLEANING"] = "conservative_v2"
os.environ["DARIJA_MAX_LENGTH"] = str(CONFIG_OVERRIDES.get("max_length", 128))
current_sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (PROJECT / "src").glob("*.py")}
loaded = sys.modules.get("src.preprocess")
project_modules = [name for name in sys.modules if name == "src" or name.startswith("src.")]
if project_modules and (loaded is None
        or Path(loaded.__file__).resolve().parent.parent != PROJECT
        or loaded.ROOT != OUTPUT or loaded.CLEANING != "conservative_v2"
        or globals().get("_DARIJA_IMPORTED_SOURCE_HASHES") != current_sources):
    # Recharger ensemble les modules Python du projet pour éviter des références croisées périmées.
    for name in project_modules:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    print("Anciens imports du projet retirés ; chargement du projet et de la sortie actuels.")
# Le projet courant doit précéder les anciens dossiers runtime, même s'il était déjà dans sys.path.
sys.path[:] = [str(PROJECT), *(entry for entry in sys.path if entry != str(PROJECT))]
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch
    from IPython.display import display, Markdown, HTML, Image
    from src import campaign
    from src.preprocess import ROOT, LABELS, download_data, load_split, audit_data, tokens, build_vocab, encode_texts
    from src.evaluate import results_table, select_on_dev, attention_examples, plot_histories, by_script, decide
    from src.train import runtime_info, load_predictor
    from src.resources import preflight
except Exception as error:
    raise RuntimeError(f"Import impossible : {error}. Installer requirements-notebook.txt dans ce kernel, redémarrer la session et relancer depuis le début.") from error
_DARIJA_IMPORTED_SOURCE_HASHES = current_sources
CONFIG = campaign.default_config()
CONFIG.update(CONFIG_OVERRIDES)
if DEVICE != "auto":
    CONFIG["device"] = DEVICE
if ARAVEC_PATH:
    CONFIG["aravec_path"] = ARAVEC_PATH
CONFIG["aravec_path"] = str(Path(CONFIG["aravec_path"]).expanduser().resolve())
os.environ["DARIJA_EVAL_DEVICE"] = CONFIG["device"]
campaign.prepare(CONFIG, dry_run=True)
if RUN_TRAINING:
    resource_report = preflight(CONFIG, check_models=CHECK_MODEL_ACCESS, download_aravec=AUTO_DOWNLOAD_ARAVEC)
    display(pd.Series(resource_report).to_frame("Ressources vérifiées avant A"))
    if CONFIG["include_e2"]:
        campaign.auxiliary_train()  # Vérifier arq dès le départ ; aucune tâche n'est entraînée ici.
manifest = campaign.prepare(CONFIG)
plt.rcParams.update({"figure.figsize": (9, 4), "axes.spines.top": False, "axes.spines.right": False})
COLORS = ["#b65b54", "#87958b", "#1f6755"]


def show_file(name, limit=30):
    path = ROOT / "results" / name
    if not path.exists():
        print("Non exécuté :", name)
        return pd.DataFrame()
    frame = pd.read_csv(path)
    display(frame.head(limit))
    return frame


def show_attention(frame):
    if frame.empty:
        print("Pas de checkpoint attention disponible.")
        return
    for example, group in frame.groupby("example", sort=False):
        maximum = max(group.attention.max(), 1e-12)
        line = " ".join(f'<span title="{r.attention:.4f}" style="background:rgba(31,103,85,{0.08+0.65*r.attention/maximum:.3f});padding:3px">{html.escape(r.token)}</span>' for r in group.itertuples())
        display(HTML(f'<p><b>{html.escape(str(example))}</b></p><p dir="auto" style="line-height:2.5">{line}</p>'))


def dev_summary(family=None):
    table = results_table()
    if table.empty:
        print("Aucun résultat : exécuter les entraînements.")
        return pd.DataFrame()
    table = table.loc[table.split.eq("dev") & table.decision.eq("dev_offsets") & table.purpose.eq("benchmark")]
    if family:
        table = table.loc[table.family.eq(family)]
    summary = table.groupby(["family", "variant"]).agg(weighted_f1=("weighted_f1", "mean"), macro_f1=("macro_f1", "mean"), std=("weighted_f1", "std"), graines=("seed", "nunique"), secondes=("train_seconds", "mean")).reset_index()
    return summary.sort_values("weighted_f1", ascending=False)


def after_test():
    if not (ROOT / "results/comparison.csv").exists():
        raise RuntimeError("Exécuter la cellule de verrouillage et d'évaluation finale avant cette analyse.")


display(pd.Series(CONFIG, name="Configuration figée après contrôle").to_frame())
display(pd.Series(runtime_info()).to_frame("Runtime"))
print("Projet :", PROJECT, "\nSorties :", ROOT)
print("GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
