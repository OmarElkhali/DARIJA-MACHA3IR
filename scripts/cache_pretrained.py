"""Prépare les poids publics B/D sans lancer d'expérience ni lire de labels test."""
import os
import sys
from pathlib import Path
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huggingface_hub import snapshot_download
from src.models import MODEL_REVISIONS, BERT_ID, NLI_ID, CAMEL_ID
from src.preprocess import ROOT, save_json

status = []
for repo in (BERT_ID, CAMEL_ID, NLI_ID):
    print("Téléchargement :", repo, flush=True)
    try:
        path = snapshot_download(repo, revision=MODEL_REVISIONS[repo], allow_patterns=["*.json", "vocab.txt", "spm.model", "README.md", "pytorch_model.bin" if repo == CAMEL_ID else "model.safetensors"], max_workers=2, etag_timeout=15)
        status.append({"model": repo, "status": "cached", "path": path})
        print("Disponible :", repo, flush=True)
    except Exception as error:
        status.append({"model": repo, "status": "failed", "error": f"{type(error).__name__}: {error}"})
        print("Échec :", repo, type(error).__name__, str(error)[:200], flush=True)
    save_json(ROOT / "results/download_status.json", status)
