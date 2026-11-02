"""Réduire uniquement E2/P avant leur démarrage, sans redémarrer le kernel."""
from pathlib import Path
import json
import hashlib


def activate_reduced_plan(config, namespace=None):
    from src import campaign, evaluate
    import pandas as pd

    root = Path(campaign.ROOT)
    plan_path = root / "reduced_remaining_plan.json"
    if getattr(campaign, "_reduced_remaining", None) is not None:
        state = campaign._reduced_remaining
        if json.loads(plan_path.read_text(encoding="utf-8")) != state["plan"]:
            raise RuntimeError("Le plan réduit a été modifié sur disque.")
        if namespace is not None:
            namespace["select_on_dev"] = evaluate.select_on_dev
        if state["plan"]["E2"]["lambdas"] != [0.1, 0.3]:
            raise RuntimeError("Ancien plan encore chargé : réexécuter la configuration avant d'activer les deux λ.")
        print("Plan réduit déjà actif : E2 = 2 runs ; P = 2 runs, sans emojis ni ponctuation.")
        return state["plan"]

    lock_path = root / "results/selection_lock.json"
    if lock_path.exists() and not plan_path.exists():
        raise RuntimeError("Campagne déjà verrouillée : aucun changement de plan autorisé.")
    original_jobs = campaign.jobs
    original_select = evaluate.select_on_dev
    original_lock = evaluate.lock_selection
    campaign.prepare(config)  # Conserver les contrôles du manifeste et les anciens résultats.
    manifest_hash = hashlib.sha256((root / "campaign.json").read_bytes()).hexdigest()
    plan = {"version": 1, "base_manifest_sha256": manifest_hash,
            "E2": {"lambdas": [0.1, 0.3], "seeds": [42], "epochs": 3},
            "P": {"cleanings": ["without_both"], "seeds": [42], "epochs": 3,
                  "models": ["TF-IDF union", "DarijaBERT"]},
            "references": {"E2": "encodeur choisi sur E1, graine 42", "P": ["A_union_s42", "E_darijabert_s42"]},
            "scope": "Réduction avant entraînement E2/P, décidée pour le temps de calcul. E1 et A-D inchangés. Comparaisons exploratoires : une graine et budget réduit ; pas d'estimation de variabilité E2/P."}
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise RuntimeError("Un autre plan réduit existe déjà ; ne pas mélanger les expériences.")
        if lock_path.exists():
            locked = json.loads(lock_path.read_text(encoding="utf-8"))
            if locked.get("reduced_plan") != plan:
                raise RuntimeError("Le verrou existant ne correspond pas au plan réduit.")
    else:
        # Ne pas rebaptiser/réutiliser un run E2/P commencé avec un budget différent.
        started = [p for p in (root / "models").glob("*")
                   if (p.name.startswith("P_") or "_mtl_l" in p.name) and any(p.iterdir())]
        if started:
            raise RuntimeError("E2/P déjà commencés : conserver leur protocole ou créer une autre campagne. "
                               + ", ".join(p.name for p in started))
        temporary = plan_path.with_suffix(".json.part")
        temporary.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(plan_path)

    def reduced_jobs(part, settings):
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise RuntimeError("Le plan réduit a été modifié sur disque.")
        if (part == "E2" and not settings["include_e2"]) or (part == "P" and not settings["include_cleaning_study"]):
            return []
        planned = original_jobs(part, settings)
        if part not in {"E2", "P"}:
            return planned
        reduced = []
        for run, function, args, kwargs in planned:
            if not run.endswith("_s42"):
                continue
            if part == "E2" and kwargs.get("auxiliary_weight") not in plan["E2"]["lambdas"]:
                continue
            if part == "P" and kwargs.get("cleaning") != "without_both":
                continue
            kwargs = dict(kwargs)
            if "epochs" in kwargs:
                kwargs["epochs"] = 3
            reduced.append((run, function, args, kwargs))
        expected = 2 if settings.get("include_e2" if part == "E2" else "include_cleaning_study", True) else 0
        if len(reduced) != expected:
            raise RuntimeError(f"Plan {part} inattendu : {len(reduced)} runs, {expected} attendus.")
        return reduced

    def reduced_selection():
        base = original_select()
        declared = set()
        for part in ("E2", "P"):
            try:
                declared.update(run for run, *_ in reduced_jobs(part, config))
            except RuntimeError as error:
                if part != "E2" or not str(error).startswith("Terminer E1"):
                    raise
        table = evaluate.results_table()
        table = table.loc[table.run_id.isin(declared) & table.split.eq("dev")
                          & table.decision.eq("dev_offsets") & table.purpose.eq("benchmark")]
        if table.empty:
            return base
        extra = table.groupby(["family", "variant"], as_index=False).agg(
            dev_weighted_f1=("weighted_f1", "mean"), dev_macro_f1=("macro_f1", "mean"),
            seed_std=("weighted_f1", "std"), seeds=("seed", "nunique"))
        extra = extra.loc[extra.seeds.eq(1)]
        return pd.concat([base, extra], ignore_index=True).drop_duplicates(["family", "variant"], keep="last").sort_values(
            ["dev_weighted_f1", "family", "variant"], ascending=[False, True, True])

    def reduced_lock():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise RuntimeError("Le plan réduit a été modifié sur disque.")
        locked = original_lock()
        digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        locked["artifacts"]["__reduced_remaining_plan__"] = {plan_path.name: digest}
        locked["reduced_plan"] = plan
        evaluate.save_json(root / "results/selection_lock.json", locked)
        return locked

    campaign.jobs = reduced_jobs
    campaign.select_on_dev = evaluate.select_on_dev = reduced_selection
    campaign.lock_selection = evaluate.lock_selection = reduced_lock
    campaign._reduced_remaining = {"plan": plan}
    if namespace is not None:
        namespace["select_on_dev"] = reduced_selection
    print("PLAN RÉDUIT : E2 = λ 0.1 et 0.3, graine 42, 3 époques maximum (2 runs au lieu de 9).")
    print("P = sans emojis ET ponctuation : 1 TF-IDF + 1 DarijaBERT, graine 42, 3 époques maximum.")
    print("Les références conservatrices A/E1 sont réutilisées. Résultats exploratoires à une graine.")
    print("Plan sauvegardé :", plan_path)
    return plan
