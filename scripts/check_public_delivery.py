"""Vérifier le paquet public sans exécuter les cellules du notebook."""
from pathlib import Path
import ast
import json
import re
import zipfile
import nbformat

PROJECT = Path(__file__).resolve().parents[1]
patterns = [r'gh[pousr]_[A-Za-z0-9]{30,}', r'github_pat_[A-Za-z0-9_]{40,}',
            r'hf_[A-Za-z0-9]{30,}', r'sk-[A-Za-z0-9_-]{30,}']
hits = []
for folder in ('src', 'site', 'scripts', 'notebooks', 'research'):
    for path in (PROJECT/folder).rglob('*'):
        if path.is_file() and '.openai' not in path.parts and path.suffix in {'.py','.json','.ipynb','.js','.md','.txt','.yml'}:
            text = path.read_text(encoding='utf-8', errors='ignore')
            if any(re.search(pattern,text) for pattern in patterns):
                hits.append(str(path.relative_to(PROJECT)))
assert not hits, hits
notebook = nbformat.read(PROJECT/'notebooks/TP2_Darija_complet_resultats.ipynb', as_version=4)
nbformat.validate(notebook)
for index, cell in enumerate(notebook.cells):
    if cell.cell_type == 'code':
        assert cell.execution_count is None
        ast.parse(cell.source, filename=f'cell_{index}')
        assert not any(output.output_type == 'error' for output in cell.outputs)
deck = PROJECT/'presentation/Darija_Soutenance_38_Slides_QR/Darija_Soutenance_38_Slides_QR_Finale.pptx'
with zipfile.ZipFile(deck) as archive:
    slides = [name for name in archive.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',name)]
    assert len(slides) == 38
data = json.loads((PROJECT/'site/dist/data.json').read_text(encoding='utf-8'))
assert len(data['comparison']) == 6 and len(data['variants']) == 33
assert data['meta']['candidate_run'] == 'E_arabizi_s42'
assert next(row for row in data['comparison'] if row['family']=='E')['weighted_f1'] == 0.608138
print(f'Publication vérifiée : {len(notebook.cells)} cellules, 38 diapositives, six familles, 33 variantes ; aucun motif de secret détecté.')
