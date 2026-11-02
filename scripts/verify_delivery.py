"""Vérifie les livrables, exporte le notebook et rédige un bilan à partir des fichiers réels."""
import sys
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import nbformat
from src.preprocess import ROOT, load_split, save_json
from src.evaluate import assert_locked, results_table
from src.train import plan_table
from src.research import write_research, solutions_table, oral_report


def main():
    checks = {}
    assert_locked()
    assert not (ROOT / "data/arq").exists()
    for name, script in [("pipeline", "tests/check_pipeline.py"), ("streamlit", "tests/check_app.py"), ("checkpoints", "tests/check_checkpoints.py")]:
        completed = subprocess.run([sys.executable, "-X", "utf8", script], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8")
        checks[name] = {"passed": True, "stdout": completed.stdout, "stderr": completed.stderr}
        print(name, "OK", flush=True)
    import json
    ui_check = json.loads((ROOT / "results/ui_verification.json").read_text(encoding="utf-8"))
    checks["streamlit"].update(artifact="results/ui_verification.json", verified_utc=ui_check["verified_utc"])
    checkpoint_check = json.loads((ROOT / "results/checkpoint_verification.json").read_text(encoding="utf-8"))
    checks["checkpoints"].update(count=checkpoint_check["count"], artifact="results/checkpoint_verification.json", max_absolute_score_difference=max(row["max_absolute_score_difference"] for row in checkpoint_check["checks"]))
    write_research()
    solutions_table()
    oral_report()
    subprocess.run([sys.executable, "-X", "utf8", "scripts/build_notebook.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-X", "utf8", "scripts/execute_notebook.py"], cwd=ROOT, check=True)
    notebook = nbformat.read(ROOT / "notebooks/TP2_Darija.ipynb", as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    errors = [o for cell in code for o in cell.outputs if o.output_type == "error"]
    assert len(code) == 40 and all(c.execution_count is not None for c in code) and not errors
    assert all(len(c.outputs) > 0 for c in code), "Une cellule de code ne produit aucun affichage"
    checks["notebook"] = {"code_cells": len(code), "executed": len(code), "errors": len(errors), "all_have_output": True}
    table = results_table()
    test = load_split("test", labels=True)
    assert table.query("split == 'test'").n.eq(len(test)).all()
    assert set(table.family) <= set("ABCDE")
    status = plan_table()
    status.to_csv(ROOT / "results/execution_status.csv", index=False)
    counts = table.drop_duplicates("run_id").groupby("family").size()
    final = pd.read_csv(ROOT / "results/comparison.csv")
    assert len(final) == 5
    checks["benchmark"] = {"distinct_runs": int(table.run_id.nunique()), "results_rows": len(table), "by_family": counts.to_dict(), "test_n": len(test), "ary_only": True}
    checks["figures"] = [p.name for p in sorted((ROOT / "results/figures").glob("*.png"))]
    smoke = ROOT / "results/pretrained_smoke.json"
    if smoke.exists():
        import json
        checks["pretrained_technical_steps"] = json.loads(smoke.read_text(encoding="utf-8"))
    save_json(ROOT / "results/verification.json", checks)
    selected_columns = ["family", "status", "variant", "weighted_f1", "macro_f1", "n_seeds", "cpu_ms_per_tweet"]
    report = "# Bilan réel d'exécution — 29 octobre 2026\n\nPérimètre : **ary exclusivement**. Exécution Conda locale, CPU, sans CUDA. Les énoncés officiels Q1–Q12 sont absents ; les douze thèmes d'oral ne sont pas leur remplacement certifié.\n\n"
    report += f"## Expériences exécutées\n\n{table.run_id.nunique()} runs distincts ; {len(table)} lignes de métriques (dev/test × argmax/règle dev), sans scores inventés. Répartition : {counts.to_dict()}. Les modèles B sont fixes : un run par variante, pas trois répétitions artificielles. Les familles entraînées disposent de trois graines par variante. F2 possède ses propres sous-ensembles train, conformément à sa définition.\n\n"
    report += "```text\n" + final[[c for c in selected_columns if c in final]].to_string(index=False) + "\n```\n\n"
    report += "## Variantes non exécutées\n\n```text\n" + status.loc[status.status != "exécuté"].to_string(index=False) + "\n```\n\nLa campagne E complète n'a pas été exécutée : pas de GPU, budget RAM réduit et coût CPU élevé. Le modèle standard comporte environ 147 millions de paramètres avec sa tête ; poids FP32 + gradients + deux moments AdamW + copie du meilleur état demandent au moins 2,74 Gio, avant activations et temporaires. Voir resource_estimate_E.json. Un pas technique sur deux exemples, lorsqu'il est présent dans pretrained_smoke.json, n'est pas un entraînement complet et ne crée aucune métrique E dans results.csv.\n\n"
    report += "## Robustesse\n\nF1 : proportions dev/test et comparaison a posteriori des décisions sans réoptimiser. F2 : arabe→latin et latin→arabe avec graines 42/43/44. F3 : F1 par écriture, OOV train/dev/test ; translittération facultative non exécutée. F4 : perturbations emojis/ponctuation sur les tweets polarisés initialement corrects de chaque famille sélectionnée. F5 : 1 000 bootstrap, IC 95 %, variabilité entre graines pour A/C/D, McNemar exact apparié. F6 : 30 erreurs avec catégories **provisoires**, votes disponibles et colonnes de validation humaine ; la validation interprétative reste à faire.\n\n"
    report += f"## Vérifications\n\n- Notebook : 40 cellules de code exécutées, aucune erreur, un affichage par cellule ; export HTML avec sorties.\n- Pipeline : formes, gradients, masques, vocabulaire, FastText, tête séquentielle et boucle de fine-tuning sur petit encodeur de test.\n- Streamlit : six espaces, corpus sans labels test, prédictions réelles A/B/C/D, filtres, erreurs, historique et mode soutenance. Voir ui_verification.json ; contrôle visuel navigateur automatisé non réalisé.\n- Checkpoints : {checkpoint_check['count']} rechargements réels sur un exemple dev de chaque écriture, comparés aux scores sauvegardés ; détails dans checkpoint_verification.json.\n- Sélection : empreintes des checkpoints verrouillés vérifiées avant test ; aucune nouvelle sélection à partir du test.\n\n## Limites scientifiques\n\n20 doublons train, 2 collisions entre partitions officielles conservées ; absence de contamination externe non certifiée ; petit effectif mixte ; couleurs d'attention descriptives et non causales ; catégories F6 et indices de français à valider ; E complet absent. Temps D hors extraction lorsque le cache est utilisé : coût d'extraction partagé dans frozen_extraction_time.json.\n"
    (ROOT / "results/EXECUTION.md").write_text(report, encoding="utf-8")
    print("Livrables vérifiés et bilan enregistré.")


if __name__ == "__main__":
    main()
