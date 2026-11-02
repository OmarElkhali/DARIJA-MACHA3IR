"""Publier la soutenance finale de 38 pages dans les deux copies du site."""
from pathlib import Path
import hashlib
import re
import shutil
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
DECK = PROJECT / 'presentation/Darija_Soutenance_38_Slides_QR'
NAME = 'Darija_Soutenance_38_Slides_QR_Finale.pptx'

def publish():
    source = DECK / NAME
    with zipfile.ZipFile(source) as archive:
        assert archive.testzip() is None
        assert len([n for n in archive.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml', n)]) == 38
        assert len([n for n in archive.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml', n)]) == 38
    for folder in ('site/dist', 'docs'):
        base = PROJECT / folder
        downloads = base / 'downloads'
        shutil.copy2(source, downloads / NAME)
        shutil.copy2(DECK / 'speech.md', downloads / 'notes_soutenance.md')
        shutil.copy2(DECK / 'script_oral_simple.md', downloads / 'script_oral_simple.md')
        if (DECK / 'planche_apercus.png').exists():
            shutil.copy2(DECK / 'planche_apercus.png', downloads / 'planche_soutenance.png')
        index = base / 'index.html'
        text = index.read_text(encoding='utf-8')
        for previous in ('Darija_Sentiment_Resultats_2026.pptx', 'Darija_Sentiment_Soutenance_Resultats.pptx'):
            text = text.replace(previous, NAME)
            obsolete = downloads / previous
            if obsolete.exists():
                obsolete.unlink()
        text = text.replace('26 diapositives', '38 diapositives').replace('31 diapositives', '38 diapositives')
        if 'downloads/script_oral_simple.md' not in text:
            text = text.replace('38 diapositives ↓</a>', '38 diapositives ↓</a> · <a href="downloads/script_oral_simple.md" download>Script oral simple — 38 pages ↓</a>')
        index.write_text(text, encoding='utf-8')
        assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256((downloads / NAME).read_bytes()).digest()
    for name in ('README.md', 'README_LIVRAISON.md'):
        file = PROJECT / name
        text = file.read_text(encoding='utf-8')
        text = re.sub(r'\*\*\[(?:Soutenance illustrée|Présentation PowerPoint) — \d+ diapositives\]\(presentation/[^)]+\)\*\*',
                      f'**[Soutenance illustrée — 38 diapositives](presentation/{DECK.name}/{NAME})**', text)
        text = re.sub(r'\[Notes pour l’oral\]\(presentation/[^)]+\)', f'[Notes pour l’oral](presentation/{DECK.name}/speech.md)', text)
        text = re.sub(r'^- `presentation/(?:Darija_Sentiment_Soutenance_Resultats|Resultats_2026)/`.*$',
                      f'- `presentation/{DECK.name}/` : soutenance finale en 38 pages, notes intégrées, comparaison scientifique et 12 questions-réponses avec renvois. Anciennes versions supprimées.', text, flags=re.M)
        text = text.replace('26 nouvelles diapositives illustrées', '38 diapositives illustrées')
        file.write_text(text, encoding='utf-8')
    print('38 pages et 38 notes vérifiées ; téléchargements et liens actualisés dans site/dist et docs.')

if __name__ == '__main__':
    publish()
