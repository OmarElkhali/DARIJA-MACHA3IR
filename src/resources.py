"""Ressources officielles : téléchargements atomiques et vérification avant entraînement."""
import hashlib
import json
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from zipfile import ZipFile

ARAVEC_URL = "https://bakrianoo.ewr1.vultrobjects.com/aravec/full_uni_cbow_100_twitter.zip"
ARAVEC_FILES = {
    "full_uni_cbow_100_twitter.mdl": "f4ec79343e3389113e6e9157a6a0bc007d0fd42199f67cbed2925d150e2c848c",
    "full_uni_cbow_100_twitter.mdl.trainables.syn1neg.npy": "e015e1ab24c6af9cb1481bbbc99e1f84dbc49fc92ff338ff00b8167ea89860eb",
    "full_uni_cbow_100_twitter.mdl.wv.vectors.npy": "8c4febdc7cb2dc4f115fb75f1b80530b92d7d4c9ca4ed8a34eee4fc15f6167d7",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_atomic(url, target, expected_sha256=None, attempts=3):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=45) as response, partial.open("wb") as output:
                received = 0
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
                    received += len(block)
                    if received % (64 * 1024 * 1024) < len(block):
                        print(f"Téléchargement {target.name} : {received / 2**20:.0f} Mio", flush=True)
                length = response.headers.get("Content-Length")
            if length is not None and received != int(length):
                raise ConnectionError("Téléchargement incomplet")
            if expected_sha256 and file_sha256(partial) != expected_sha256:
                raise ValueError(f"Empreinte incorrecte : {target.name}, aucun fichier valide remplacé")
            partial.replace(target)
            return target
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt == attempts - 1:
                raise RuntimeError(f"Téléchargement impossible : {url}. Vérifier Internet ou fournir le fichier local officiel.") from error
            time.sleep(attempt + 1)
        finally:
            if partial.exists():
                partial.unlink()


def ensure_aravec(path, allow_download=False):
    path = Path(path).expanduser().resolve()
    if path.name != "full_uni_cbow_100_twitter.mdl":
        raise ValueError("Cette campagne utilise AraVec CBOW Twitter 100D : full_uni_cbow_100_twitter.mdl")
    missing = []
    for name, digest in ARAVEC_FILES.items():
        candidate = path.parent / name
        if not candidate.exists():
            missing.append(name)
        elif file_sha256(candidate) != digest:
            raise ValueError(f"Poids AraVec incorrects : {candidate}. Utiliser les fichiers officiels vérifiés.")
    if not missing:
        return path
    if not allow_download:
        raise FileNotFoundError("AraVec incomplet : " + ", ".join(missing) + ". Renseigner ARAVEC_PATH ou activer AUTO_DOWNLOAD_ARAVEC avant les entraînements.")
    path.parent.mkdir(parents=True, exist_ok=True)
    archive_path = path.parent / "full_uni_cbow_100_twitter.zip"
    print("AraVec absent : téléchargement officiel avant la campagne (~1 Gio une fois extrait).", flush=True)
    download_atomic(ARAVEC_URL, archive_path)
    try:
        with ZipFile(archive_path) as archive:
            for name in missing:
                members = [m for m in archive.infolist() if Path(m.filename).name == name and not m.is_dir()]
                if len(members) != 1:
                    raise ValueError(f"Archive AraVec inattendue : {name}")
                target = path.parent / name
                temporary = target.with_suffix(target.suffix + ".part")
                try:
                    with archive.open(members[0]) as source, temporary.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    if file_sha256(temporary) != ARAVEC_FILES[name]:
                        raise ValueError(f"Poids AraVec corrompus : {name}")
                    temporary.replace(target)
                finally:
                    if temporary.exists():
                        temporary.unlink()
    finally:
        if archive_path.exists():
            archive_path.unlink()
    return path


def preflight(config, check_models=True, download_aravec=False):
    """Aucun forward ni apprentissage ; valider toutes les dépendances avant A."""
    import importlib
    import torch
    from .preprocess import ROOT, download_data, save_json
    from .models import MODEL_REVISIONS
    for name in ("numpy", "scipy", "pandas", "sklearn", "gensim", "transformers", "sentencepiece", "google.protobuf", "matplotlib"):
        try:
            importlib.import_module(name)
        except Exception as error:
            raise RuntimeError(f"Bibliothèque {name} inutilisable : {error}. Installer requirements-notebook.txt puis redémarrer le kernel.") from error
    ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=ROOT):
        pass
    if config["device"] == "cuda":
        try:
            probe = torch.empty(1, device="cuda")
            del probe
        except RuntimeError as error:
            raise RuntimeError("CUDA annoncé mais allocation impossible ; vérifier le runtime GPU/PyTorch ou choisir DEVICE='cpu'.") from error
    data = download_data()
    config["aravec_path"] = str(ensure_aravec(config["aravec_path"], allow_download=download_aravec))
    from gensim.models import Word2Vec
    model = Word2Vec.load(config["aravec_path"], mmap="r")
    if model.wv.vector_size != 100:
        raise ValueError("Dimension AraVec inattendue")
    del model
    if check_models:
        from huggingface_hub import hf_hub_download
        for model_id, revision in MODEL_REVISIONS.items():
            try:
                hf_hub_download(model_id, "config.json", revision=revision)
            except Exception as error:
                raise RuntimeError(f"Accès au modèle {model_id} impossible. Vérifier Internet/le cache avant de lancer A–G.") from error
    report = {"data_files": len(data), "aravec": config["aravec_path"], "device": config["device"],
              "free_disk_gib": shutil.disk_usage(ROOT).free / 2**30, "model_configs_checked": check_models,
              "note": "Configurations accessibles ; poids complets, entraînements et mémoire de pic non vérifiés."}
    save_json(ROOT / "preflight.json", report)
    return report
