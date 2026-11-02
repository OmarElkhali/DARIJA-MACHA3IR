"""Consolider les vrais résultats Colab en site et notebook, sans apprentissage."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import base64
import copy
import hashlib
import io
import json
import re
import shutil
import textwrap
import urllib.request
import zlib

import nbformat
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / 'experiments/campaign_v4'
DIST = PROJECT / 'site/dist'
DOWNLOADS = DIST / 'downloads'
EXPECTED_EXPORT = 'd1802dab0b66a9d3383585e2ddc1fd459a4273e3168b48387249e07cc2b71440'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def import_colab():
    archive = PROJECT / 'research/darija_livraison.zip'
    if not archive.exists():
        raise FileNotFoundError('Télécharger research/darija_livraison.zip depuis le runtime Colab.')
    if digest(archive) != EXPECTED_EXPORT:
        raise ValueError('L’archive téléchargée diffère de celle exportée par Colab.')
    backup = PROJECT.parent / 'backups/before_web_delivery'
    with ZipFile(archive) as zipped:
        for entry in zipped.infolist():
            relative = Path(entry.filename)
            target = (ROOT.parent / relative).resolve()
            if not target.is_relative_to(ROOT.resolve()) or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f'Chemin interdit dans l’archive : {entry.filename}')
            if entry.is_dir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # Conserver l’état local précédent avant d’importer les résultats courants.
            if target.exists():
                old = backup / target.relative_to(ROOT)
                if not old.exists():
                    old.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target, old)
            with zipped.open(entry) as incoming, target.open('wb') as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=4 * 1024 * 1024)
    print('Archive Colab vérifiée et importée.', flush=True)


def records(path):
    frame = pd.read_csv(path, keep_default_na=False)
    return json.loads(frame.to_json(orient='records', force_ascii=False))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def notebook_delivery(data):
    main_path = PROJECT / 'notebooks/TP2_Darija.ipynb'
    final_path = PROJECT / 'notebooks/TP2_Darija_colab_FG.ipynb'
    main = nbformat.read(main_path, as_version=4)
    final = nbformat.read(final_path, as_version=4)
    setup_index = next(i for i, c in enumerate(final.cells) if c.cell_type == 'code' and "frozen = json.loads((OUTPUT / 'campaign.json')" in c.source)
    completed_index = next(i for i, c in enumerate(final.cells) if c.cell_type == 'code' and 'def skip_missing_p(namespace)' in c.source and 'final = campaign.finalize(CONFIG)' in c.source)
    analysis_start = next(i for i,c in enumerate(final.cells) if c.cell_type=='markdown' and 'F1.1' in c.source)
    analysis_end = next((i for i,c in enumerate(final.cells) if c.cell_type=='code' and ('LIVRAISON PRETE' in c.source or 'EXPORT TERMINE' in c.source)), len(final.cells))
    cells = copy.deepcopy(final.cells[:setup_index + 1])
    cells += copy.deepcopy(main.cells[15:69])
    cells += [nbformat.v4.new_markdown_cell('## Complétude, exclusion P et évaluation finale\n\nLe verrou inclut la déclaration d’exclusion du seul checkpoint P DarijaBERT absent. Les autres fichiers sont contrôlés. Les mesures test et les choix restent ceux de l’exécution enregistrée.')]
    cells += [copy.deepcopy(final.cells[completed_index])]
    cells += [copy.deepcopy(c) for c in final.cells[analysis_start:analysis_end]
              if not (c.cell_type == 'markdown' and 'Conclusion à rédiger' in c.source)]
    for cell in cells:
        if cell.cell_type == 'code' and 'checklist = ' in cell.source:
            methodology = (PROJECT / 'research/site_methodology.md').read_text(encoding='utf-8')
            cell.source = 'from IPython.display import Markdown, display\ndisplay(Markdown(' + repr(methodology) + '))'
            cell.outputs = [nbformat.v4.new_output('display_data', data={'text/markdown':methodology})]
    candidate = next(r for r in data['comparison'] if r['run_id'] == data['meta']['candidate_run'])
    cells[0].source = f'''# Darija-Sentiment · code complet et résultats mesurés

Ce notebook consolide les sorties enregistrées de la campagne **campaign_v4**. Les résultats A-E/P proviennent du notebook principal ; les résultats test et les analyses F/G proviennent du nouveau Colab T4. Les originaux sont conservés.

**Candidat fixé sur validation : {candidate['variant']}**, graine 42. Weighted-F1 test : **{candidate['weighted_f1']:.6f}** ; Macro-F1 test : **{candidate['macro_f1']:.6f}**. Le test avait déjà été observé dans l’étude historique : cette campagne est exploratoire.

**Exclusion :** `P_darijabert_without_both_s42` n’a pas de fichier de poids disponible. Ses scores dev restent présents ; il n’a aucun score test. Les six familles restent représentées par A, B, C, D, E et P (TF-IDF).

**Lecture / replay par défaut :** exécuter toutes les cellules réaffiche les sorties sauvegardées, sans installer de dépendances, monter Drive, entraîner ou réévaluer un modèle. Ce replay n’est pas une nouvelle mesure. Pour relancer le pipeline original, activer `REEXECUTER_PIPELINE=True` ci-dessous ; les checkpoints et ressources de la campagne restent alors nécessaires. Le code de tous les modules est aussi lisible en annexe.
'''
    saved = {}
    removed_errors = []
    for index, cell in enumerate(cells):
        if cell.cell_type != 'code':
            continue
        outputs = []
        for output in cell.get('outputs', []):
            if output.output_type == 'error':
                removed_errors.append({'cell': index, 'name': output.get('ename', 'interruption enregistrée')})
            else:
                outputs.append(output)
        cell.outputs = outputs
        saved[str(index)] = json.loads(json.dumps(outputs))
        original = cell.source
        cell.source = 'if REEXECUTER_PIPELINE:\n' + textwrap.indent(original, '    ') + f'\nelse:\n    rejouer_sorties({str(index)!r})'
        cell.execution_count = None
        cell.metadata['provenance'] = 'Sorties réelles assemblées ; replay sans nouvelle mesure par défaut.'
        compile(cell.source, f'<cell-{index}>', 'exec')
    payload = base64.b64encode(zlib.compress(json.dumps(saved, ensure_ascii=False).encode('utf-8'), level=9)).decode('ascii')
    replay = f'''# False : lecture des résultats existants. True : réexécution du pipeline et accès aux sauvegardes.
REEXECUTER_PIPELINE = False
import base64, json, zlib
from IPython.display import display
_SORTIES_MESUREES = json.loads(zlib.decompress(base64.b64decode({payload!r})))

def rejouer_sorties(cle):
    for sortie in _SORTIES_MESUREES.get(cle, []):
        if sortie['output_type'] == 'stream':
            print(sortie.get('text', ''), end='')
        elif sortie['output_type'] in ('display_data', 'execute_result'):
            display(sortie['data'], raw=True, metadata=sortie.get('metadata', {{}}))
print('Replay des sorties mesurées ; aucun nouvel entraînement ou score.')
'''
    cells.insert(1, nbformat.v4.new_code_cell(replay))
    cells.append(nbformat.v4.new_markdown_cell('## Conclusion issue des mesures\n\n' +
        f"Le candidat choisi sur validation est **{candidate['variant']}**. Son F1 pondéré test vaut **{candidate['weighted_f1']:.4f}**, avec un IC bootstrap 95 % de **[{candidate['ci_low']:.4f}, {candidate['ci_high']:.4f}]** conditionnel au checkpoint 42. Le résultat de validation et celui du test restent distincts. Les diagnostics d’écriture et les perturbations décrivent des limites ; ils n’en démontrent pas une cause unique. Les revues humaines et un nouvel ensemble indépendant demandent encore une annotation documentée.\n\nLe plan E2 réduit ne teste que λ=0,1 et 0,3. La mention λ=0,7 dans certaines questions historiques n’est pas une mesure de cette campagne. Les interruptions conservées dans les notebooks originaux ne sont pas présentées comme des expériences test supplémentaires."))
    cells.append(nbformat.v4.new_markdown_cell('## Annexe · code source intégral\n\nLes modules ci-dessous sont ceux du projet contrôlé par le manifeste de campagne. Les utilitaires de reprise et d’exclusion sont présentés séparément ; ils ne modifient pas les fichiers src figés.'))
    source_paths = sorted((PROJECT / 'src').glob('*.py'))
    source_paths += [PROJECT / 'scripts' / name for name in ['notebook_bootstrap.py','notebook_setup.py','continue_campaign.py','reduce_remaining.py','skip_missing_p.py']]
    for path in source_paths:
        cells.append(nbformat.v4.new_markdown_cell(f"### {path.relative_to(PROJECT).as_posix()}\n\n```python\n{path.read_text(encoding='utf-8')}\n```"))
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'name':'python3','display_name':'Python 3','language':'python'},
        'darija_delivery': {'campaign':'campaign_v4','candidate':data['meta']['candidate_run'],
            'source_notebooks':{p.name:digest(p) for p in [main_path, final_path]},
            'archive_sha256':EXPECTED_EXPORT,'saved_output_cells':len(saved),
            'historical_interruptions_retained_in_originals':removed_errors,
            'default_mode':'replay_saved_outputs_not_new_measurements'}})
    nbformat.validate(notebook)
    target = PROJECT / 'notebooks/TP2_Darija_complet_resultats.ipynb'
    nbformat.write(notebook, target)
    shutil.copy2(target, DOWNLOADS / target.name)
    print(f'Notebook consolidé : {len(cells)} cellules, {len(saved)} blocs de sorties.', flush=True)


def build():
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    lock = read_json(ROOT / 'results/selection_lock.json')
    comparison = records(ROOT / 'results/comparison.csv')
    assert {r['family'] for r in comparison} == set('ABCDEP')
    candidate_run = f"{lock['candidate']['family']}_{lock['candidate']['variant']}_s42"
    metrics = pd.read_csv(ROOT / 'results/results.csv')
    assert 'P_darijabert_without_both_s42' not in lock['test_runs']
    assert len(lock['test_runs']) == 71
    for row in comparison:
        measured = metrics.loc[metrics.run_id.eq(row['run_id']) & metrics.split.eq('test') & metrics.decision.eq('dev_offsets')]
        assert len(measured) == 1 and abs(measured.weighted_f1.iloc[0] - row['weighted_f1']) < 1e-12
    dev = metrics.loc[metrics.split.eq('dev') & metrics.decision.eq('dev_offsets') & metrics.purpose.eq('benchmark')]
    summary = dev.groupby(['family','variant']).agg(weighted_f1=('weighted_f1','mean'),macro_f1=('macro_f1','mean'),std=('weighted_f1','std'),graines=('seed','nunique')).reset_index()
    corpus = []
    for split,name in [('train','Train'),('dev','Validation'),('test','Test')]:
        frame=pd.read_csv(PROJECT / f'data/ary/{split}.tsv', sep='\t')
        corpus.append({'split':split,'name':name,'n':len(frame),**frame.label.value_counts().to_dict()})
    csv_names=['f1_label_shift','f1_conditional_labels','f2_cross_script','embedding_oov_by_script','f6_30_erreurs',
               'f4_human_scores','human_test_scores','training_times_cumulative','cleaning_study','oov','execution_status']
    tables={n:records(ROOT / f'results/{n}.csv') for n in csv_names if (ROOT / f'results/{n}.csv').exists()}
    json_names=['f5_mcnemar','f1_posthoc_oracle','evaluation_exclusions','c_imbalance_selection','e2_encoder_selection','arq_audit']
    docs={n:read_json(ROOT / f'results/{n}.json') for n in json_names if (ROOT / f'results/{n}.json').exists()}
    models,robustness={},[]
    for row in comparison:
        models[row['family']]=read_json(ROOT / f"models/{row['run_id']}/meta.json")
        file=ROOT / f"results/f4_{row['run_id']}_summary.csv"
        if file.exists():
            robustness += [{**r,'family':row['family'],'run_id':row['run_id']} for r in records(file)]
    figures=[]
    figure_dir=DIST / 'figures'
    figure_dir.mkdir(exist_ok=True)
    titles={'curves_C.png':'C : courbes d’apprentissage et de validation','curves_D.png':'D : courbes de la tête gelée',
            'curves_E.png':'E : courbes du fine-tuning','curves_P.png':'P : courbes de l’ablation',
            'modele_final.png':'Matrice de confusion du candidat fixé sur validation',
            'attention_C.png':'Attention C sur cinq exemples de validation','attention_D.png':'Attention D sur cinq exemples de validation',
            'F1_threshold_grid.png':'Diagnostic post hoc : grille des décalages','F1_threshold_curve.png':'Diagnostic post hoc : courbe des décalages',
            'G_quality_cost.png':'Compromis qualité et coûts mesurés'}
    for name,title in titles.items():
        file=ROOT / 'results/figures' / name
        if file.exists():
            shutil.copy2(file, figure_dir / name)
            figures.append({'file':name,'title':title})
    data={'meta':{'campaign':'campaign_v4','candidate_run':candidate_run,'test_runs':len(lock['test_runs']),
                   'archive_sha256':EXPECTED_EXPORT,'criterion':lock['criterion'],'exploratory':True},
          'comparison':comparison,'variants':records(ROOT / 'results/comparison_all_variants.csv'),
          'dev_summary':json.loads(summary.to_json(orient='records',force_ascii=False)),
          'corpus':corpus,'tables':tables,'json':docs,'models':models,'robustness':robustness,'figures':figures}
    (DIST / 'data.js').write_text('window.DARIJA_DATA = '+json.dumps(data,ensure_ascii=False,allow_nan=False)+';\n',encoding='utf-8')
    for name in ['comparison.csv','comparison_all_variants.csv','f6_30_erreurs.csv']:
        shutil.copy2(ROOT / 'results' / name, DOWNLOADS / name)
    with ZipFile(DOWNLOADS / 'resultats_campaign_v4.zip','w',ZIP_DEFLATED,compresslevel=6) as zipped:
        for file in (ROOT / 'results').rglob('*'):
            if file.is_file() and 'cache' not in file.relative_to(ROOT / 'results').parts:
                zipped.write(file,str(Path('campaign_v4') / file.relative_to(ROOT)))
        for name in ['campaign.json','reduced_remaining_plan.json']:
            zipped.write(ROOT / name,str(Path('campaign_v4') / name))
    notebook_delivery(data)
    fonts=DIST / 'fonts'
    fonts.mkdir(exist_ok=True)
    if not (fonts / 'Manrope-latin.woff2').exists():
        request=urllib.request.Request('https://fonts.googleapis.com/css2?family=Manrope:wght@200..800&display=swap',headers={'User-Agent':'Mozilla/5.0'})
        css=urllib.request.urlopen(request,timeout=30).read().decode()
        urls=re.findall(r'url\((https://[^)]+)\)',css)
        if urls:
            urllib.request.urlretrieve(urls[-1],fonts / 'Manrope-latin.woff2')
    (PROJECT / 'research/web_delivery_verification.json').write_text(json.dumps({
        'candidate':candidate_run,'representatives':6,'test_runs':71,'test_rows':int(metrics.split.eq('test').sum()),
        'excluded':lock.get('evaluation_exclusions'),'metric_alignment':'verified','training_executed':False,
        'notebook':str(PROJECT / 'notebooks/TP2_Darija_complet_resultats.ipynb')},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Site alimenté par les CSV courants, alignement des six scores vérifié.', flush=True)


if __name__=='__main__':
    import_colab()
    build()
