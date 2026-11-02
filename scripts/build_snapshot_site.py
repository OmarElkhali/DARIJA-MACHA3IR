"""Alimenter le site depuis les outputs sauvegardés, sans exécuter un notebook."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json
import re
import shutil
import urllib.request
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
DIST=PROJECT/'site/dist'
DOWNLOADS=DIST/'downloads'
SNAPSHOT=PROJECT/'research/notebook_tables_snapshot.json'


def scalar(value):
    if value in {'NaN','nan','None',''}:
        return None
    if value in {'True','False'}:
        return value=='True'
    try:
        return float(value) if any(c in value for c in '.eE') else int(value)
    except (ValueError,TypeError):
        return value


def main():
    snapshot=json.loads(SNAPSHOT.read_text(encoding='utf-8'))
    def table(index):
        t=snapshot['tables'][index]
        headers=[c['text'] for c in t['header_rows'][-1]]
        rows=[]
        for cells in t['body_rows']:
            values=[c['text'] for c in cells]
            if any(v in {'...','…'} for v in values) and all(v in {'...','…',''} for v in values):
                continue
            rows.append({(h or '_index'):scalar(v) for h,v in zip(headers,values) if h not in {'...','…'}})
        return rows
    comparison=table(34)
    variants=table(35)
    extra={(r['family'],r['variant']):r for r in table(27)}
    for row in comparison+variants:
        row.update(extra.get((row['family'],row['variant']),{}))
        for name in ['cpu_ms_per_tweet','weight_mb','train_seconds','trainable_parameters']:
            row[name]=None
        row['scripts']=None
    assert len(comparison)==6 and len(variants)==33
    assert {r['family'] for r in comparison}==set('ABCDEP')
    for family,index in [('E',14),('A',15),('C',16),('D',17),('P',18),('B',19)]:
        next(r for r in comparison if r['family']==family)['scripts']=table(index)
    candidate='E_arabizi_s42'
    assert next(r for r in comparison if r['run_id']==candidate)['weighted_f1']==0.608138
    models={}
    for row in comparison:
        p=PROJECT/f"experiments/campaign_v4/models/{row['run_id']}/meta.json"
        models[row['family']]=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    robust=[]
    for row in table(20):
        match=next(r for r in comparison if r['family']==row['family'])
        robust.append({**row,'run_id':match['run_id']})
    for row in table(22):
        if not any(r['run_id']==candidate and r['transformation']==row['transformation'] for r in robust):
            robust.append({**row,'family':'E','run_id':candidate})
    corpus=[]
    for split,name in [('train','Train'),('dev','Validation'),('test','Test')]:
        frame=pd.read_csv(PROJECT/f'data/ary/{split}.tsv',sep='\t')
        corpus.append({'split':split,'name':name,'n':len(frame),**frame.label.value_counts().to_dict()})
    mc={r['_index']:r.get('McNemar') for r in table(29)}
    figures=[]
    (DIST/'figures').mkdir(parents=True,exist_ok=True)
    manifest=json.loads((PROJECT/'research/notebook_figures/manifest.json').read_text(encoding='utf-8'))
    names={21:('F1_distributions.png','F1 : distributions validation et test'),
           29:('F4_bascules.png','F4 : bascules de prédiction'),35:('F5_intervalles.png','F5 : intervalles bootstrap 95 %'),
           41:('modele_final.png','Matrice de confusion du candidat fixé sur validation'),43:('G_quality_cost.png','G : qualité et coûts mesurés')}
    for item in manifest['figures']:
        if item['cell_index']==23:
            name='F1_threshold_grid.png' if item['output_index']==0 else 'F1_threshold_curve.png'
            title='F1 : diagnostic post hoc des décalages'
        else:
            name,title=names[item['cell_index']]
        shutil.copy2(PROJECT/item['path'],DIST/'figures'/name)
        figures.append({'file':name,'title':title})
    # Les courbes C/D/E/P proviennent du notebook principal, pas d’une exécution locale.
    import base64
    import nbformat
    main=nbformat.read(PROJECT/'notebooks/TP2_Darija.ipynb',as_version=4)
    import io
    dev_summary=[]
    for cell in main.cells:
        if cell.cell_type!='code':
            continue
        for output in cell.get('outputs',[]):
            html=output.get('data',{}).get('text/html')
            if not html or '<table' not in html:
                continue
            frames=pd.read_html(io.StringIO(html))
            if not frames:
                continue
            frame=frames[0]
            if {'family','variant','weighted_f1','macro_f1','graines'} <= set(frame.columns):
                dev_summary.extend(json.loads(frame.to_json(orient='records')))
    chart_outputs=[o for o in main.cells[66].outputs if 'image/png' in o.get('data',{})]
    for family,output in zip('CDEP',chart_outputs[-4:]):
        name=f'curves_{family}.png'
        (DIST/'figures'/name).write_bytes(base64.b64decode(output.data['image/png']))
        figures.append({'file':name,'title':f'{family} : courbes enregistrées d’apprentissage et de validation'})
    data={'meta':{'campaign':'campaign_v4','candidate_run':candidate,'test_runs':71,
          'source':'Outputs des notebooks sauvegardés, sans nouvelle exécution',
          'source_notebook_sha256':snapshot['notebook_sha256'],'precision':'Décimales affichées dans les outputs originaux',
          'unavailable_models':['E'],'hidden_numeric_columns':['cpu_ms_per_tweet','weight_mb','train_seconds','trainable_parameters']},
          'comparison':comparison,'variants':variants,'dev_summary':dev_summary, 'corpus':corpus,'models':models,
          'robustness':robust,'figures':figures,
          'tables':{'f1_conditional_labels':table(6),'f2_cross_script':table(10),'f6_30_erreurs':table(30),
                    'training_times_family':table(37),'human_test_scores':table(25),'f4_human_scores':table(24)},
          'json':{'f5_mcnemar':mc,'evaluation_exclusions':{'excluded_test_runs':['P_darijabert_without_both_s42']}}}
    raw=json.dumps(data,ensure_ascii=False,allow_nan=False)
    (DIST/'data.json').write_text(raw,encoding='utf-8')
    (DIST/'data.js').write_text('window.DARIJA_DATA = '+raw+';\n',encoding='utf-8')
    DOWNLOADS.mkdir(exist_ok=True)
    for name,rows in [('comparison.csv',comparison),('comparison_all_variants.csv',variants),('f6_30_erreurs.csv',table(30))]:
        pd.DataFrame(rows).to_csv(DOWNLOADS/name,index=False)
    for extension in ['ipynb','html']:
        p=PROJECT/f'notebooks/TP2_Darija_complet_resultats.{extension}'
        shutil.copy2(p,DOWNLOADS/p.name)
    shutil.copy2(SNAPSHOT,DOWNLOADS/SNAPSHOT.name)
    with ZipFile(DOWNLOADS/'resultats_campaign_v4.zip','w',ZIP_DEFLATED) as zipped:
        zipped.write(SNAPSHOT,'notebook_tables_snapshot.json')
        zipped.write(DIST/'data.json','donnees_presentation.json')
        for path in (DIST/'figures').glob('*.png'):
            zipped.write(path,'figures/'+path.name)
        for path in DOWNLOADS.glob('*.csv'):
            zipped.write(path,path.name)
        zipped.writestr('PROVENANCE.txt','Extraction des outputs sauvegardés, sans exécution. Les colonnes masquées sont nulles ; aucun chiffre n’est reconstruit. Les CSV exportés sont des transcriptions des valeurs affichées, pas les CSV bruts du runtime.')
    fonts=DIST/'fonts'
    fonts.mkdir(exist_ok=True)
    if not (fonts/'Manrope-latin.woff2').exists():
        req=urllib.request.Request('https://fonts.googleapis.com/css2?family=Manrope:wght@200..800&display=swap',headers={'User-Agent':'Mozilla/5.0'})
        css=urllib.request.urlopen(req,timeout=30).read().decode()
        urls=re.findall(r'url\((https://[^)]+)\)',css)
        if urls:
            urllib.request.urlretrieve(urls[-1],fonts/'Manrope-latin.woff2')
    print(f'Site : {len(comparison)} représentants, {len(variants)} variantes, {len(figures)} figures. Aucun notebook exécuté.')


if __name__=='__main__':
    main()
