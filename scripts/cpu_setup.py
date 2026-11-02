"""Reprendre les mesures sur CPU sans changer le protocole des entraînements."""
from pathlib import Path
import copy
import hashlib
import json
import os
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / 'experiments/campaign_v4'
frozen = json.loads((OUTPUT / 'campaign.json').read_text(encoding='utf-8'))
CONFIG_OVERRIDES = dict(frozen['config'])
CONFIG_OVERRIDES.update(device='cpu', aravec_path=str(PROJECT / 'models/resources/aravec/full_uni_cbow_100_twitter.mdl'))
DEVICE, ARAVEC_PATH = 'cpu', CONFIG_OVERRIDES['aravec_path']
RUN_TRAINING, RUN_FINAL_EVALUATION = False, True
CHECK_MODEL_ACCESS = AUTO_DOWNLOAD_ARAVEC = False
os.environ.update(DARIJA_OUTPUT_ROOT=str(OUTPUT), DARIJA_CLEANING='conservative_v2',
                  DARIJA_MAX_LENGTH=str(CONFIG_OVERRIDES['max_length']), DARIJA_EVAL_DEVICE='cpu')
sys.path.insert(0, str(PROJECT))
from src import campaign

_prepare = getattr(campaign, '_cpu_original_prepare', campaign.prepare)
campaign._cpu_original_prepare = _prepare


def prepare_cpu(config, dry_run=False):
    if json.loads((OUTPUT / 'campaign.json').read_text(encoding='utf-8')) != frozen:
        raise RuntimeError('Le manifeste sauvegardé a changé pendant la reprise.')
    expected = copy.deepcopy(frozen)
    expected['config'].update(device='cpu', aravec_path=ARAVEC_PATH)
    original_root = campaign.ROOT
    # Réutiliser toutes les validations, dans un dossier sans manifeste historique.
    with tempfile.TemporaryDirectory(prefix='darija_cpu_check_') as directory:
        try:
            campaign.ROOT = Path(directory)
            actual = _prepare(config, dry_run=True)
        finally:
            campaign.ROOT = original_root
    if actual != expected:
        raise RuntimeError('Le code ou le protocole diffère de la campagne sauvegardée.')
    return frozen


campaign.prepare = prepare_cpu
_DARIJA_IMPORTED_SOURCE_HASHES = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (PROJECT / 'src').glob('*.py')}
exec(compile((PROJECT / 'scripts/notebook_setup.py').read_text(encoding='utf-8'),
             str(PROJECT / 'scripts/notebook_setup.py'), 'exec'), globals())
exec(compile((PROJECT / 'scripts/continue_campaign.py').read_text(encoding='utf-8'),
             str(PROJECT / 'scripts/continue_campaign.py'), 'exec'), globals())
install_storage_writes()
install_resume_checks(campaign)
exec(compile((PROJECT / 'scripts/reduce_remaining.py').read_text(encoding='utf-8'),
             str(PROJECT / 'scripts/reduce_remaining.py'), 'exec'), globals())
activate_reduced_plan(CONFIG, globals())
(ROOT / 'results/figures').mkdir(parents=True, exist_ok=True)
provenance = {'device': 'cpu', 'torch': torch.__version__, 'python': sys.version,
              'campaign_sha256': hashlib.sha256((ROOT / 'campaign.json').read_bytes()).hexdigest(),
              'training_restarted': False}
(ROOT / 'results/cpu_execution.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
display(dev_summary())
show_training_times(globals())
print('CPU local : évaluations uniquement, entraînements conservés.')
