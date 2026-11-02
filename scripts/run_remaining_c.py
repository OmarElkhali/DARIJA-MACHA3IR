"""Complète seulement les variantes C absentes ; aucune lecture des labels test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess import ROOT, save_json, sha256
from src.train import neural_run, classical_run, plan_table

aravec = ROOT / "models/resources/aravec/full_uni_cbow_100_twitter.mdl"
save_json(aravec.parent / "provenance.json", {
    "source": "https://github.com/bakrianoo/aravec",
    "download_url": "https://bakrianoo.ewr1.vultrobjects.com/aravec/full_uni_cbow_100_twitter.zip",
    "origin": "Copie de poids officiels disponibles dans le cache local ; aucun code ni donnée d'un autre projet n'est utilisé.",
    "files": {p.name: sha256(p) for p in aravec.parent.glob("*.mdl*")},
})
# Le détecteur de script ignore désormais @user, URLs et [[PHOTO]]. Recalculer F2 avant verrouillage.
for script in ("arabic", "latin"):
    classical_run("union", 42, train_script=script)
for seed in (42, 43, 44):
    for kind, positional in (("bilstm", True), ("attention", True), ("transformer", True), ("transformer", False)):
        for embedding in ("fasttext", "aravec"):
            variant = f"{kind}_{embedding}" + ("_nopos" if not positional else "")
            if (ROOT / f"models/C_{variant}_s{seed}/meta.json").exists():
                print("Déjà exécuté :", variant, seed, flush=True)
                continue
            neural_run(kind, embedding, seed, positional=positional, aravec_path=aravec if embedding == "aravec" else None)
plan_table().to_csv(ROOT / "results/execution_status.csv", index=False)
