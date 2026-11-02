"""Assembler un notebook de présentation à partir des sorties sauvegardées, sans l’exécuter."""
from pathlib import Path
import copy
import hashlib
import io
import json
import nbformat as nbf
import pandas as pd
from nbconvert import HTMLExporter

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / 'notebooks/TP2_Darija_complet_resultats.ipynb'


def main():
    source_paths = [PROJECT / 'notebooks/TP2_Darija.ipynb', PROJECT / 'notebooks/TP2_Darija_colab_FG.ipynb']
    original, final = [nbf.read(p, as_version=4) for p in source_paths]
    cells = []
    provenance = []
    def md(content):
        cells.append(nbf.v4.new_markdown_cell(content))
    def measured(notebook, index, label, source=None):
        cell = copy.deepcopy(notebook.cells[index])
        if cell.cell_type != 'code':
            cells.append(cell)
            return
        cell.source = source if source is not None else cell.source
        kept = []
        for output in cell.get('outputs', []):
            if output.output_type == 'error':
                continue
            if output.output_type == 'stream':
                # Les logs de téléchargement, checkpoints et barres de progression restent dans les originaux.
                continue
            if output.output_type in {'display_data', 'execute_result'}:
                mime = output.get('data', {})
                if 'text/html' in mime or 'image/png' in mime or 'image/svg+xml' in mime:
                    kept.append(output)
        cell.outputs = kept
        cell.metadata.update({'source_notebook':label,'source_cell_index':index,
                              'source_execution_count':cell.get('execution_count'),
                              'output_origin':'Sorties enregistrées, assemblées sans exécuter ce notebook.'})
        cell.execution_count = None
        compile(cell.source, f'{label}:cell{index}', 'exec')
        provenance.append({'notebook':label,'cell':index,'outputs_retained':len(kept)})
        cells.append(cell)
    comparison = None
    for cell in final.cells:
        if cell.cell_type != 'code' or 'final = show_file("comparison.csv")' not in cell.source:
            continue
        for output in cell.outputs:
            html = output.get('data', {}).get('text/html')
            if html:
                table = pd.read_html(io.StringIO(html))[0]
                if 'family' in table and len(table) == 6:
                    comparison = table
                    break
    if comparison is None:
        raise ValueError('Les six résultats finaux doivent être présents dans les sorties enregistrées.')
    md('''# Darija / Sentiment
## Une étude du sentiment marocain, du TF-IDF au fine-tuning

**Notebook final de présentation · campagne `campaign_v4` · novembre 2026**

Ce document réunit le code utile, les tableaux et les graphiques réellement enregistrés dans les notebooks de travail. Il a été **assemblé sans réexécuter de cellule, sans réentraîner et sans recalculer les scores**. Les compteurs d’exécution de ce nouveau document sont laissés vides ; les provenances des sorties figurent dans les métadonnées des cellules.

Le projet compare **six familles A, B, C, D, E et P**, avec trois sentiments : négatif, neutre et positif. Le candidat est choisi sur validation ; son représentant est toujours la graine 42.

**Périmètre final : 71 runs test.** Le run `P_darijabert_without_both_s42` est exclu car son fichier de poids est absent. Ses résultats de validation sont conservés. P reste représenté par le TF-IDF union sans emojis ni ponctuation.

**Limite centrale :** le test ary avait déjà été observé dans une étude historique. Ces résultats sont une réévaluation exploratoire, pas une validation indépendante.

### Parcours de lecture
1. Question, corpus et protocole
2. Six familles de méthodes et résultats de validation
3. Résultats test et analyses F : écritures, décalages, robustesse, incertitude et erreurs
4. Synthèse G, choix pratique et limites
5. Annexe technique : code source intégral et traçabilité
''')
    md('## Les résultats en un regard\n\nTableau importé de la sortie finale. Les décimales sont celles affichées dans le notebook source. Les colonnes masquées par `…` dans les sorties d’origine ne sont pas inventées.')
    visible = [c for c in ['family','variant','weighted_f1','macro_f1','dev_weighted_f1_mean'] if c in comparison]
    overview = comparison[visible].copy()
    cells.append(nbf.v4.new_code_cell('# Tableau de synthèse importé des sorties mesurées, sans nouvelle inférence.\nimport pandas as pd\nfrom IPython.display import display\nsynthese = pd.DataFrame(' + repr(json.loads(overview.to_json(orient='records'))) + ')\ndisplay(synthese)',
        outputs=[nbf.v4.new_output('display_data',data={'text/html':overview.to_html(index=False),'text/plain':overview.to_string(index=False)})]))
    md('''## La question de recherche

Comment reconnaître le sentiment dans des tweets marocains écrits en arabe, en Arabizi ou dans un mélange d’écritures ? Les formes lexicales, les négations, les chiffres Arabizi, les emojis et les changements de domaine rendent la généralisation difficile.

**Train apprend. Validation choisit. Le verrou fige. Test mesure.** Le Weighted-F1 est le critère principal ; le Macro-F1 complète l’analyse des trois classes. Les réseaux standards sont comparés sur 42/43/44 ; le représentant final est 42.

Les trois partitions ary restent officielles : **5 583 tweets train, 494 validation, 2 961 test**. `arq/train` ne sert qu’à l’auxiliaire E2. L’ablation P et E2 réduit ne permettent pas d’estimer une variabilité sur trois graines.
''')
    md('''## Préparation reproductible

Le bloc suivant contient l’amorçage autonome et la configuration. Les données et le code d’origine sont incorporés ; les poids volumineux restent externes. Les cellules ne sont pas exécutées pendant l’assemblage de ce document. Réutiliser les sauvegardes de campagne pour une nouvelle exécution ; ne pas présenter un simple replay comme une nouvelle mesure.
''')
    startup = '\n\n'.join(final.cells[i].source for i in [2,4,6,8,10,12,14])
    startup = startup.replace('RUN_TRAINING = True','RUN_TRAINING = False').replace('RUN_FINAL_EVALUATION = True','RUN_FINAL_EVALUATION = False')
    startup_cell=nbf.v4.new_code_cell(startup,metadata={'jupyter':{'source_hidden':True},'tags':['configuration-integrale']})
    compile(startup,'<startup>','exec')
    cells.append(startup_cell)
    groups=[('Corpus, audit et représentations du texte',[18,20,22,24,26]),
            ('A · TF-IDF et baselines',[28,30]),('B · Transfert sans apprentissage local',[32,34,36]),
            ('C · Réseaux compacts, embeddings et déséquilibre',[38,40,42,44,46,48,50]),
            ('D · DarijaBERT gelé',[52,54,56]),('E · Fine-tuning et auxiliaire algérien',[58,60,62,64,66]),
            ('P · Nettoyage sans emojis ni ponctuation',[68])]
    for heading,indices in groups:
        md('## '+heading)
        if heading.startswith('E '):
            md('E1 compare DarijaBERT et DarijaBERT-Arabizi sur trois graines. E2 réduit compare uniquement **λ=0,1 et 0,3**, graine 42, trois époques maximum. L’encodeur E2 est choisi sur E1/validation. Aucun résultat λ=0,7 n’est revendiqué.')
        if heading.startswith('P '):
            md('Le TF-IDF de P est évalué jusqu’au test. Le DarijaBERT de P a des résultats dev, mais son poids est absent : il est **exclu de F/G**, conformément au périmètre final. Le tableau de validation ne doit pas être lu comme un résultat test de ce modèle.')
        for index in indices:
            previous=original.cells[index-1]
            if previous.cell_type=='markdown':
                measured(original,index-1,'TP2_Darija.ipynb')
            measured(original,index,'TP2_Darija.ipynb')
    md('## Frontière validation / test\n\nLe candidat verrouillé est **E · DarijaBERT-Arabizi**. Le run P DarijaBERT absent est retiré du périmètre test, sans supprimer ses mesures dev. La déclaration d’exclusion est enregistrée dans le verrou de la campagne. Les autres sauvegardes sont vérifiées.')
    final_call=next(i for i,c in enumerate(final.cells) if c.cell_type=='code' and 'def skip_missing_p(namespace)' in c.source and 'final = campaign.finalize(CONFIG)' in c.source)
    clean_final='''# Code de finalisation utilisé par la campagne. Nécessite les checkpoints externes.
if RUN_FINAL_EVALUATION:
    final = campaign.finalize(CONFIG)
else:
    final = show_file("comparison.csv")
display(final)
'''
    measured(final,final_call,'TP2_Darija_colab_FG.ipynb',clean_final)
    # Le code complet de l’exclusion est conservé en annexe, pas répété au milieu de la présentation.
    for i in range(final_call+1,len(final.cells)):
        c=final.cells[i]
        if c.cell_type=='code' and ('EXPORT TERMINE' in c.source or 'LIVRAISON PRETE' in c.source or 'darija_E_weights' in c.source):
            break
        if c.cell_type=='markdown' and 'Conclusion à rédiger' in c.source:
            continue
        if c.cell_type=='code' and 'checklist = ' in c.source:
            continue
        measured(final,i,'TP2_Darija_colab_FG.ipynb')
    md('''## Conclusion

**DarijaBERT-Arabizi** est le candidat retenu sur validation. Son Weighted-F1 test est **0,608138** et son Macro-F1 est **0,600620**. Son IC bootstrap 95 % conditionnel au checkpoint 42 est **[0,590655 ; 0,624099]**. Le Weighted-F1 de validation moyen est **0,811256** : la baisse entre validation et test reste un résultat central à expliquer, sans modifier les choix après observation du test.

Les cinq autres représentants donnent des repères complémentaires : TF-IDF caractères (**0,553509**), CAMeLBERT (**0,420555**), BiLSTM-AraVec (**0,493340**), DarijaBERT gelé avec tête séquentielle (**0,571593**) et TF-IDF union sans emojis ni ponctuation (**0,567396**), selon la règle fixée sur validation.

Les analyses d’écriture, OOV et perturbations sont **diagnostiques**. Elles ne démontrent pas une cause unique de la baisse. Le retrait d’un emoji peut changer le sentiment réel. Les suggestions de catégories d’erreurs ne deviennent pas des observations humaines tant que la revue n’a pas été validée.

Le déploiement dépend aussi de la latence et du coût. Les graphiques G restent ceux mesurés sur le matériel de l’exécution. Une confirmation des conclusions demande de **nouveaux textes annotés indépendamment**, sans réglage sur ce benchmark déjà observé.
''')
    md('## Annexe · code source complet\n\nCes modules rendent le travail inspectable sans exécuter ce notebook. Les utilitaires de reprise sont distingués des sources scientifiques figées.')
    paths=sorted((PROJECT/'src').glob('*.py'))+[PROJECT/'scripts'/n for n in ['notebook_setup.py','reduce_remaining.py','skip_missing_p.py']]
    for path in paths:
        md(f"### {path.relative_to(PROJECT).as_posix()}\n\n<details><summary>Afficher le code intégral</summary>\n\n```python\n{path.read_text(encoding='utf-8')}\n```\n\n</details>")
    metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
              'darija_final':{'assembled_without_execution':True,'campaign':'campaign_v4',
               'originals':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},'cells':provenance}}
    notebook=nbf.v4.new_notebook(cells=cells,metadata=metadata)
    nbf.validate(notebook)
    nbf.write(notebook,OUT)
    html,_=HTMLExporter().from_notebook_node(notebook)
    OUT.with_suffix('.html').write_text(html,encoding='utf-8')
    print(f'Notebook final : {OUT}\n{len(cells)} cellules, sans exécution. HTML rendu depuis les sorties enregistrées.')


if __name__=='__main__':
    main()
