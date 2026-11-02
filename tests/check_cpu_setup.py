"""Contrôler le passage CPU et le refus de changer le protocole."""
from pathlib import Path
import contextlib
import io
import runpy

project = Path(__file__).resolve().parents[1]
manifest = project / 'experiments/campaign_v4/campaign.json'
before = manifest.read_bytes()
with contextlib.redirect_stdout(io.StringIO()):
    namespace = runpy.run_path(str(project / 'scripts/cpu_setup.py'))
    namespace = runpy.run_path(str(project / 'scripts/cpu_setup.py'))
assert namespace['CONFIG']['device'] == 'cpu'
assert namespace['RUN_TRAINING'] is False
assert namespace['manifest']['config']['device'] == 'cuda'
changed = dict(namespace['CONFIG'], max_length=64)
try:
    namespace['campaign'].prepare(changed)
except (ValueError, RuntimeError):
    pass
else:
    raise AssertionError('Un changement du protocole doit être refusé')
assert manifest.read_bytes() == before
print('CPU : protocole historique conservé, configuration modifiée refusée, reprise répétable.')
