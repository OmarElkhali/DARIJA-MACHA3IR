"""Vérifier la reprise sans écraser un notebook ouvert ni lancer le réseau."""
from pathlib import Path
import runpy
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
generator = runpy.run_path(str(PROJECT / 'scripts/create_colab_continuation.py'))
with patch('nbformat.write') as write:
    generator['main']()
notebook = write.call_args.args[0]
assert 'shortcut-targets-by-id' in notebook.cells[2].source
assert 'RUN_TRAINING = False' in notebook.cells[2].source
code = [c for c in notebook.cells if c.cell_type == 'code']
assert all(c.execution_count is None and not c.outputs for c in code)
assert all('campaign.run_part(' not in c.source for c in code)
for cell in code:
    compile(cell.source, '<colab-continuation-check>', 'exec')
print(f'Reprise vérifiée : {len(code)} cellules compilées ; aucun fichier ni entraînement modifié.')
