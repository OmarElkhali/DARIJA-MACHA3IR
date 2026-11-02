"""Synchroniser les jobs après QA en conservant l'historique avant réparation."""
from pathlib import Path
import json
import subprocess
import sys

PROJECT=Path(__file__).resolve().parents[1]
DECK=PROJECT/'presentation/Darija_Sentiment_Soutenance_Resultats'
SKILL=Path.home()/'.agents/skills/codex-ppt/scripts'
jobs=json.loads((DECK/'slide_jobs.json').read_text(encoding='utf-8'))
assert all(s['status'] in {'recorded','accepted'} for s in jobs['slides'])
(DECK/'qa_jobs_before_refinement.json').write_text(json.dumps(jobs,ensure_ascii=False,indent=2),encoding='utf-8')

def run(script,*args):
    result=subprocess.run([sys.executable,str(script),*map(str,args)],capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)

run(PROJECT/'scripts/prepare_styled_soutenance.py','--finalize')
run(SKILL/'prepare_slide_prompts.py','--spec',DECK/'deck_spec.json','--out-dir',DECK,
    '--selected-backend','built-in image tool','--max-concurrent-slides','3','--force')
for slide in jobs['slides']:
    if slide['number'] in {16,19}:
        continue
    agent=slide['dispatch']['agent_id']
    run(SKILL/'record_slide_dispatch.py',DECK,'--slide',slide['slide_id'],'--agent-id',agent,
        '--prompt-file',slide['job'])
    result=slide['result']
    run(SKILL/'record_slide_result.py',DECK,'--slide',slide['slide_id'],'--agent-id',agent,
        '--backend-used',result['backend_used'],'--selected-source',result['selected_source'],
        '--qa-note',result['qa_note']+' Provenance antérieure conservée dans qa_jobs_before_refinement.json.')
(DECK/'qa_repairs.json').write_text(json.dumps({'repairs':[
    {'slide':7,'reason':'Schéma TF-IDF explicitement illustratif sans poids inventés.'},
    {'slide':11,'reason':'Tokens ordinaires et attention optionnelle, pas de marqueurs BERT.'},
    {'slide':14,'reason':'Encodeur et budget E2 corrigés.'},
    {'slide':19,'reason':'Séparer les deux sens de transfert en rangées non croisées.'},
    {'slide':20,'reason':'Tableau exact préféré à une géométrie synthétique approximative.'}],
    'method':'Même backend intégré ; aucune surimpression ni correction locale des images.'},ensure_ascii=False,indent=2),encoding='utf-8')
print('Jobs synchronisés avec les corrections QA ; seule la diapositive19 attend une nouvelle génération.')
