"""Plan réduit réel, sans entraînement : manifeste préservé, sélection et verrou compatibles."""
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import sys
import runpy
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    with TemporaryDirectory(prefix="darija_reduced_plan_") as temporary:
        os.environ.update(DARIJA_OUTPUT_ROOT=temporary, DARIJA_CLEANING="conservative_v2", DARIJA_MAX_LENGTH="128")
        import pandas as pd
        from src import campaign, evaluate
        from src.preprocess import ROOT, save_json
        activate = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/reduce_remaining.py"))["activate_reduced_plan"]
        config = campaign.default_config()
        config["device"] = "cpu"
        campaign.prepare(config)
        old_manifest = (ROOT / "campaign.json").read_bytes()
        save_json(ROOT / "results/e2_encoder_selection.json", {"model_id": "SI2M-Lab/DarijaBERT-arabizi", "variant": "arabizi"})
        original_jobs, original_select, original_lock = campaign.jobs, evaluate.select_on_dev, evaluate.lock_selection
        reference_e1 = original_jobs("E1", config)
        state = {}
        plan = activate(config, state)
        assert (ROOT / "campaign.json").read_bytes() == old_manifest
        assert campaign.jobs("E1", config) == reference_e1
        e2, p = campaign.jobs("E2", config), campaign.jobs("P", config)
        assert {job[0] for job in e2} == {"E_arabizi_mtl_l0.1_s42", "E_arabizi_mtl_l0.3_s42"}
        assert all(job[3]["epochs"] == 3 for job in e2)
        assert {job[3]["auxiliary_weight"] for job in e2} == {0.1, 0.3}
        assert {job[0] for job in p} == {"P_union_without_both_s42", "P_darijabert_without_both_s42"}
        assert all(job[3]["cleaning"] == "without_both" for job in p)
        assert p[-1][3]["epochs"] == 3
        assert not campaign.jobs("E2", {**config, "include_e2": False})
        assert activate(config) == plan
        rows = []
        for family, variant, seeds, score in [("A", "union", [42], .6), ("E", "darijabert", [42, 43, 44], .7),
                                             ("E", "arabizi_mtl_l0.1", [42], .71),
                                             ("E", "arabizi_mtl_l0.3", [42], .73), ("P", "union_without_both", [42], .61),
                                             ("P", "darijabert_without_both", [42], .72)]:
            for seed in seeds:
                rows.append(dict(run_id=f"{family}_{variant}_s{seed}", family=family, variant=variant, seed=seed,
                                 split="dev", decision="dev_offsets", purpose="benchmark", weighted_f1=score, macro_f1=score))
        pd.DataFrame(rows).to_csv(ROOT / "results/results.csv", index=False)
        selected = evaluate.select_on_dev()
        assert set(selected.variant) == {"union", "darijabert", "arabizi_mtl_l0.1", "arabizi_mtl_l0.3", "union_without_both", "darijabert_without_both"}
        assert selected.loc[selected.variant.eq("arabizi_mtl_l0.3"), "seeds"].iloc[0] == 1
        assert state["select_on_dev"] is evaluate.select_on_dev
        locked = campaign.lock_selection()
        assert locked["reduced_plan"] == plan
        evaluate.assert_locked()
        assert (ROOT / "campaign.json").read_bytes() == old_manifest
        # Nouveau kernel : réinstaller le même plan, même après le verrouillage.
        del campaign._reduced_remaining
        campaign.jobs = original_jobs
        campaign.select_on_dev = evaluate.select_on_dev = original_select
        campaign.lock_selection = evaluate.lock_selection = original_lock
        assert activate(config) == plan
        assert len(campaign.jobs("E2", config)) == 2
        plan_path = ROOT / "reduced_remaining_plan.json"
        changed = dict(plan, version=99)
        plan_path.write_text(json.dumps(changed), encoding="utf-8")
        try:
            campaign.jobs("P", config)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Plan modifié accepté")
        try:
            evaluate.assert_locked()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Verrou du plan ignoré")
        print("OK : E2 9→2, P 15→2, E1 inchangé, manifeste préservé, sélection à une graine et plan verrouillé.")


if __name__ == "__main__":
    main()
