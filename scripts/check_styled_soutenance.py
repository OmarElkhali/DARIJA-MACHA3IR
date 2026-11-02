"""QA des pages générées et du PPTX assemblé, sans rendu de nouvelles diapositives."""
from pathlib import Path
import hashlib
import json
import re
from zipfile import ZipFile
from PIL import Image, ImageDraw

PROJECT=Path(__file__).resolve().parents[1]
DECK=PROJECT/'presentation/Darija_Sentiment_Soutenance_Resultats'
jobs=json.loads((DECK/'slide_jobs.json').read_text(encoding='utf-8'))
slides=jobs['slides']
assert len(slides)==26
assert all(s['status'] in {'accepted','recorded'} for s in slides)
images=sorted((DECK/'origin_image').glob('slide_*.png'))
assert len(images)==26
assert [p.name for p in images]==[f'slide_{n:02d}.png' for n in range(1,27)]
for path in images:
    with Image.open(path) as image:
        assert image.width>=1500 and abs(image.width/image.height-16/9)<.02,path
    assert path.stat().st_size>10000
    state=next(s for s in slides if s['slide_id']==path.stem)
    if state['status']=='recorded':
        assert hashlib.sha256(path.read_bytes()).hexdigest()==state['result']['final_image_sha256']
with ZipFile(DECK/(DECK.name+'.pptx')) as archive:
    slide_xml=[name for name in archive.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',name)]
    note_xml=[name for name in archive.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml',name)]
    assert len(slide_xml)==26 and len(note_xml)==26
    for name in slide_xml:
        assert archive.read(name).count(b'<p:pic>')==1,name
    packaged={hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist() if name.startswith('ppt/media/')}
    assert {hashlib.sha256(p.read_bytes()).hexdigest() for p in images} <= packaged
    assert all(len(archive.read(name))>400 for name in note_xml)

# Une planche de contrôle n'est pas une diapositive finale et ne modifie pas les images.
sheet=Image.new('RGB',(1600,7*245),'#e5e8df')
draw=ImageDraw.Draw(sheet)
for index,path in enumerate(images):
    with Image.open(path) as image:
        thumb=image.convert('RGB')
        thumb.thumbnail((400,225))
        x=(index%4)*400
        y=(index//4)*245
        sheet.paste(thumb,(x,y))
        draw.text((x+10,y+227),str(index+1),fill='#173d36')
sheet.save(DECK/'planche_apercus.png')
report={'slides':26,'format':'16:9','backend':'built-in image tool',
        'generated_slides_recorded':sum(s['status']=='recorded' for s in slides),
        'reference_samples':sum(s['status']=='accepted' for s in slides),
        'notes_embedded':len(note_xml),'visual_review':'Every full slide inspected before recording',
        'native_powerpoint_review':False,'notebook_executed':False,
        'measurements':'Saved campaign_v4 outputs, not historical soutenance scores',
        'images':[{'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in images]}
(DECK/'presentation_evidence.json').write_text((PROJECT/'site/dist/data.json').read_text(encoding='utf-8'),encoding='utf-8')
(DECK/'qa_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('26 pages générées, enregistrées et inspectées ; 26 notes intégrées ; PPTX en pages-images vérifié.')
