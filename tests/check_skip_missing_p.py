"""Vérifier une seule exclusion, sa traçabilité et la protection du verrou."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import pandas as pd

install = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/skip_missing_p.py'))['skip_missing_p']
with TemporaryDirectory() as directory:
    root = Path(directory)
    (root / 'results').mkdir()
    (root / 'campaign.json').write_text('{}')
    dev = root / 'results/results.csv'
    dev.write_text('Résultats dev à conserver', encoding='utf-8')
    missing = 'P_darijabert_without_both_s42'
    evaluate = ModuleType('src.evaluate')
    evaluate.select_on_dev = lambda: pd.DataFrame([
        {'family': 'P', 'variant': 'darijabert_without_both'},
        {'family': 'P', 'variant': 'union_without_both'}, {'family': 'E', 'variant': 'darijabert'}])
    campaign = SimpleNamespace(ROOT=root, jobs=lambda part, cfg: [(missing,), ('P_union_without_both_s42',)],
        lock_selection=lambda: {'artifacts': {}, 'candidate': evaluate.select_on_dev().iloc[0].to_dict()})
    preprocess = ModuleType('src.preprocess')
    preprocess.save_json = lambda p, value: p.write_text(json.dumps(value), encoding='utf-8')
    preprocess.sha256 = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    src = ModuleType('src')
    src.evaluate = evaluate
    with patch.dict(sys.modules, {'src': src, 'src.evaluate': evaluate, 'src.preprocess': preprocess}):
        namespace = {'campaign': campaign}
        install(namespace)
        install(namespace)  # Relance sans empiler les wrappers.
        assert campaign.jobs('P', {}) == [('P_union_without_both_s42',)]
        assert len(evaluate.select_on_dev()) == 2
        lock = campaign.lock_selection()
        assert lock['candidate']['variant'] == 'union_without_both'
        assert lock['evaluation_exclusions']['excluded_test_runs'] == [missing]
        assert lock['artifacts']['__evaluation_exclusions__']
        assert dev.read_text(encoding='utf-8') == 'Résultats dev à conserver'
        (root / 'results/selection_lock.json').write_text('{"artifacts": {}}')
        try:
            install(namespace)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Un verrou existant incompatible doit bloquer le changement.')
print('Exclusion unique, scores dev préservés et verrou protégés : OK.')
