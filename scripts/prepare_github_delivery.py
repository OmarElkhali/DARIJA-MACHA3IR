"""Copier les livrables vérifiés dans le clone GitHub, sans exécution de notebook."""
from pathlib import Path
import json
import shutil

PROJECT = Path(__file__).resolve().parents[1]
TARGET = PROJECT.parent / 'publish/DARIJA-MACHA3IR'
assert (TARGET / '.git').is_dir(), 'Cloner le dépôt de destination avant la préparation.'

html = PROJECT/'site/dist/index.html'
content = html.read_text(encoding='utf-8').replace(
    'CSV, figures, prédictions, manifeste et exclusions. ↓',
    'CSV transcrits, figures, provenance et exclusions. ↓').replace(
    'Qualité, latence, taille, graines et incertitude. ↓',
    'Qualité, graines et incertitude mesurées. ↓')
deck_link = '<p class="caption"><a href="downloads/Darija_Sentiment_Resultats_2026.pptx" download>Télécharger la présentation PowerPoint — 31 diapositives ↓</a> · <a href="https://github.com/OmarElkhali/DARIJA-MACHA3IR">Code et livrables sur GitHub ↗</a></p>'
if 'Télécharger la présentation PowerPoint' not in content:
    content = content.replace('<div class="references">', deck_link+'<div class="references">')
html.write_text(content, encoding='utf-8')

def copy(source, target=None):
    source = PROJECT / source
    target = TARGET / (target or source.relative_to(PROJECT))
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.ipynb_checkpoints', '.openai'))
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

for folder in ('src', 'scripts', 'tests', 'data', 'site'):
    copy(folder)
for file in ('README.md', 'requirements.txt', 'requirements-notebook.txt', 'requirements-presentation.txt'):
    copy(file)
for name in ('TP2_Darija.ipynb', 'TP2_Darija_colab_FG.ipynb',
             'TP2_Darija_complet_resultats.ipynb', 'TP2_Darija_complet_resultats.html'):
    copy('notebooks/' + name)
copy('presentation/Resultats_2026')
for name in ('notebook_tables_snapshot.json', 'notebook_figures', 'site_methodology.md',
             'extract_notebook_outputs.py', 'web_delivery_checks.json', 'site_desktop.png', 'site_mobile.png'):
    copy('research/' + name)
for name in ('campaign.json', 'preflight.json', 'reduced_remaining_plan.json'):
    copy('experiments/campaign_v4/' + name)

selected = {row['run_id'] for row in json.loads((PROJECT/'site/dist/data.json').read_text(encoding='utf-8'))['comparison']}
root = PROJECT / 'experiments/campaign_v4/models'
for folder in root.iterdir():
    if not folder.is_dir():
        continue
    for file in folder.rglob('*'):
        if file.is_file() and ('__pycache__' not in file.parts):
            if file.name in {'resume.pt', '.in_progress'}:
                continue
            if file.suffix in {'.pt', '.joblib', '.npy', '.npz'} and folder.name not in selected:
                continue
            if file.stat().st_size < 90 * 1024**2:
                copy(str(file.relative_to(PROJECT)))

# GitHub Pages sert exactement la même interface statique ; le moteur reste local.
copy('site/dist', 'docs')
(TARGET/'docs/.nojekyll').touch()
(TARGET/'.gitignore').write_text('''__pycache__/
*.pyc
.venv*/
.ipynb_checkpoints/
.env
.env.*
.openai/
*.log
experiments/**/cache/
experiments/**/resume.pt
experiments/**/.in_progress
''', encoding='utf-8')
files = [p for p in TARGET.rglob('*') if p.is_file() and '.git' not in p.parts]
assert max(p.stat().st_size for p in files) < 100 * 1024**2
print(f'{len(files)} fichiers préparés, {sum(p.stat().st_size for p in files)/1024**2:.1f} Mio ; aucun notebook exécuté.')
