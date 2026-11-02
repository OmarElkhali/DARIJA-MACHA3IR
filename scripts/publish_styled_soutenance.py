"""Mettre à disposition la soutenance vérifiée dans le site et le clone GitHub."""
from pathlib import Path
import json
import shutil

PROJECT=Path(__file__).resolve().parents[1]
DECK=PROJECT/'presentation/Darija_Sentiment_Soutenance_Resultats'
REPO=PROJECT.parent/'publish/DARIJA-MACHA3IR'
qa=json.loads((DECK/'qa_report.json').read_text(encoding='utf-8'))
assert qa['slides']==26 and qa['notes_embedded']==26
name=DECK.name+'.pptx'
dist=PROJECT/'site/dist'
shutil.copy2(DECK/name,dist/'downloads'/name)
shutil.copy2(DECK/'speech.md',dist/'downloads/notes_soutenance.md')
shutil.copy2(DECK/'planche_apercus.png',dist/'downloads/planche_soutenance.png')
index=dist/'index.html'
html=index.read_text(encoding='utf-8').replace('Darija_Sentiment_Resultats_2026.pptx',name).replace(
 'Télécharger la présentation PowerPoint — 31 diapositives','Télécharger la soutenance illustrée — 26 diapositives')
index.write_text(html,encoding='utf-8')
readme=PROJECT/'README.md'
text=readme.read_text(encoding='utf-8').replace('Présentation PowerPoint — 31 diapositives](presentation/Resultats_2026/Darija_Sentiment_Resultats_2026.pptx)',
 'Soutenance illustrée — 26 diapositives](presentation/Darija_Sentiment_Soutenance_Resultats/Darija_Sentiment_Soutenance_Resultats.pptx)').replace(
 'Notes pour l’oral](presentation/Resultats_2026/notes_orales.md)',
 'Notes pour l’oral](presentation/Darija_Sentiment_Soutenance_Resultats/speech.md)')
text=text.replace('- `presentation/Resultats_2026/` : PowerPoint éditable, notes et aperçus. Les aperçus sont des rendus de contrôle générés ; le diaporama n’a pas été contrôlé dans PowerPoint.',
 '- `presentation/Darija_Sentiment_Soutenance_Resultats/` : nouvelle soutenance en 26 pages-images, générée dans le style de la soutenance originale, notes françaises intégrées et traçabilité codex-ppt. Les images ont été inspectées ; aucune inspection native PowerPoint n’est revendiquée. La version éditable antérieure reste archivée dans `presentation/Resultats_2026/`.')
text=text.replace('31 diapositives dans les limites de page','26 nouvelles diapositives illustrées inspectées, provenance de génération et notes intégrées vérifiées')
readme.write_text(text,encoding='utf-8')
shutil.copy2(readme,REPO/'README.md')
shutil.copytree(DECK,REPO/'presentation'/DECK.name,dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('*.draft.*','.*.lock'))
for filename in ('prepare_styled_soutenance.py','write_styled_soutenance_notes.py','check_styled_soutenance.py','publish_styled_soutenance.py','refine_styled_soutenance_state.py'):
    shutil.copy2(PROJECT/'scripts'/filename,REPO/'scripts'/filename)
shutil.copytree(dist,REPO/'site/dist',dirs_exist_ok=True)
shutil.copytree(dist,REPO/'docs',dirs_exist_ok=True)
print('Nouvelle soutenance, notes et aperçu copiés dans le site et le dépôt ; anciennes versions préservées.')
