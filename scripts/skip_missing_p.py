"""Exclusion F/G demandée par l'utilisateur ; conserver les résultats dev de P."""
from pathlib import Path
import json


def skip_missing_p(namespace):
    from src import evaluate
    from src.preprocess import save_json, sha256

    campaign = namespace['campaign']
    root = Path(campaign.ROOT)
    run = 'P_darijabert_without_both_s42'
    path = root / 'results/evaluation_exclusions.json'
    declaration = {'excluded_test_runs': [run], 'reason': 'weights.pt absent ; exclusion demandée par l’utilisateur avant F/G',
                   'dev_results_preserved': True, 'campaign_sha256': sha256(root / 'campaign.json')}
    lock_path = root / 'results/selection_lock.json'
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding='utf-8'))
        if lock.get('evaluation_exclusions') != declaration:
            raise RuntimeError('Évaluation déjà verrouillée avec un autre périmètre ; aucun changement automatique.')
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != declaration:
        raise RuntimeError('Une autre déclaration d’exclusion existe.')
    save_json(path, declaration)
    if getattr(campaign, '_skip_missing_p', False):
        namespace['select_on_dev'] = evaluate.select_on_dev
        return declaration
    original_jobs, original_selection, original_lock = campaign.jobs, evaluate.select_on_dev, campaign.lock_selection

    def jobs(part, config):
        return [job for job in original_jobs(part, config) if job[0] != run]

    def selection():
        table = original_selection()
        return table.loc[~(table.family.eq('P') & table.variant.eq('darijabert_without_both'))]

    def lock_selection():
        lock = original_lock()
        lock['evaluation_exclusions'] = declaration
        lock['artifacts']['__evaluation_exclusions__'] = {str(path.relative_to(root)): sha256(path)}
        save_json(lock_path, lock)
        return lock

    campaign.jobs = jobs
    campaign.select_on_dev = evaluate.select_on_dev = selection
    campaign.lock_selection = evaluate.lock_selection = lock_selection
    campaign._skip_missing_p = True
    namespace['select_on_dev'] = selection
    print('Exclu de F/G uniquement :', run, '— scores dev conservés.', flush=True)
    return declaration
