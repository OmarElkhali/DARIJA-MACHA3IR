"""Créer un diaporama éditable depuis les résultats sauvegardés, sans notebook exécuté."""
from pathlib import Path
import json
import math
import textwrap
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

PROJECT=Path(__file__).resolve().parents[1]
OUT=PROJECT/'presentation/Resultats_2026'
OUT.mkdir(parents=True,exist_ok=True)
FIG=OUT/'figures'
FIG.mkdir(exist_ok=True)
DATA=json.loads((PROJECT/'site/dist/data.json').read_text(encoding='utf-8'))
ROWS=DATA['comparison']
VAR=DATA['variants']
INK='183C35'; MUTED='63746E'; PAPER='FAFBF8'; LIME='DCE9B6'; WHITE='FFFFFF'
COLORS={'A':'7D9756','B':'A98063','C':'6A8994','D':'658672','E':INK,'P':'AAA16C'}
NAMES={'A':'TF-IDF caractères','B':'CAMeLBERT','C':'BiLSTM + AraVec','D':'DarijaBERT gelé','E':'DarijaBERT-Arabizi','P':'TF-IDF sans signes'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.labelcolor':'#'+INK,'text.color':'#'+INK,'xtick.color':'#'+MUTED,'ytick.color':'#'+MUTED,
                     'axes.edgecolor':'#DCE3DA','figure.facecolor':'#'+PAPER,'axes.facecolor':'#'+PAPER})
prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
previews=[];manifest=[]
FONT=Path(r'C:\Windows\Fonts\segoeui.ttf')
BOLD=Path(r'C:\Windows\Fonts\segoeuib.ttf')


def color(hexcode):return RGBColor.from_string(hexcode)


def text(slide,content,x,y,w,h,size=20,bold=False,fill=INK,align=None):
    shape=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    frame=shape.text_frame;frame.word_wrap=True
    frame.margin_left=frame.margin_right=frame.margin_top=frame.margin_bottom=0
    for i,line in enumerate(str(content).split('\n')):
        p=frame.paragraphs[0] if i==0 else frame.add_paragraph()
        p.text=line;p.font.name='Segoe UI';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=color(fill)
        p.space_after=Pt(7)
        if align is not None:p.alignment=align
    slide._preview_ops.append(('text',content,x,y,w,h,size,bold,fill))
    return shape


def box(slide,x,y,w,h,fill=INK):
    s=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h))
    s.fill.solid();s.fill.fore_color.rgb=color(fill);s.line.fill.background()
    slide._preview_ops.append(('box',x,y,w,h,fill))
    return s


def image(slide,path,x,y,w,h):
    with Image.open(path) as im:iw,ih=im.size
    ratio=min(w/iw,h/ih);nw,nh=iw*ratio,ih*ratio
    nx,ny=x+(w-nw)/2,y+(h-nh)/2
    slide.shapes.add_picture(str(path),Inches(nx),Inches(ny),width=Inches(nw),height=Inches(nh))
    slide._preview_ops.append(('image',str(path),nx,ny,nw,nh))


def slide(title,subtitle='',source='Sorties enregistrées de campaign_v4'):
    s=prs.slides.add_slide(prs.slide_layouts[6]);s._preview_ops=[]
    s.background.fill.solid();s.background.fill.fore_color.rgb=color(PAPER)
    text(s,title,.65,.45,12,1.05,size=31,bold=True)
    if subtitle:text(s,subtitle,.68,1.42,11.95,.57,size=15,fill=MUTED)
    text(s,'DARIJA / SENTIMENT',.68,7.1,3,.2,size=8,fill=MUTED)
    text(s,f'{len(prs.slides):02d}',12.08,7.04,.5,.27,size=10,fill=MUTED,align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text=f'{title}\n\n{subtitle}\n\nSource : {source}.\nLes métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.'
    manifest.append({'slide':len(prs.slides),'title':title,'source':source})
    return s


def note(s,content):
    text(s,content,.72,6.48,11.85,.5,size=13,fill=MUTED)
    s.notes_slide.notes_text_frame.text+='\n\nÀ expliquer : '+content


def two(title,left_title,left,right_title,right,subtitle='',source='Code et outputs enregistrés'):
    s=slide(title,subtitle,source)
    text(s,left_title,.72,2.13,5.6,.58,size=24,bold=True)
    text(s,left,.72,2.93,5.55,3.2,size=19,fill=MUTED)
    box(s,6.62,2.15,.025,3.9,'DCE3DA')
    text(s,right_title,7.05,2.13,5.5,.58,size=24,bold=True)
    text(s,right,7.05,2.93,5.5,3.2,size=19,fill=MUTED)
    return s


def save_plot(name):
    plt.tight_layout();p=FIG/name;plt.savefig(p,dpi=175,bbox_inches='tight');plt.close();return p


def bar_plot(name,rows,title,field='weighted_f1',dev=False):
    fig,ax=plt.subplots(figsize=(11.3,4.4))
    labels=[r['family']+' · '+str(r['variant']).replace('_',' ') for r in rows]
    y=np.arange(len(rows));values=[float(r[field]) for r in rows]
    ax.barh(y,values,color=['#'+COLORS[r['family']] for r in rows],height=.58)
    for i,v in enumerate(values):ax.text(v+.012,i,f'{v:.2%}',va='center',fontsize=10)
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel(title);ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    return save_plot(name)


def chart_slide(title,subtitle,path,caption='',source='Outputs du notebook Colab F/G'):
    s=slide(title,subtitle,source);image(s,path,.7,2.1,11.9,4.15)
    if caption:note(s,caption)
    return s


def table_slide(title,headers,rows,subtitle='',source='Tableaux enregistrés'):
    s=slide(title,subtitle,source);x=.7;y=2.18;w=11.9;rowh=min(.52,4.05/(len(rows)+1))
    widths=[w/len(headers)]*len(headers)
    if len(headers)>=5:widths=[w*.11,w*.34]+[(w*.55)/(len(headers)-2)]*(len(headers)-2)
    left=x
    for label,cw in zip(headers,widths):
        box(s,left,y,cw,rowh,INK);text(s,label,left+.08,y+.09,cw-.13,rowh-.08,size=12,bold=True,fill=WHITE);left+=cw
    for ri,row in enumerate(rows):
        left=x
        for value,cw in zip(row,widths):
            if ri%2==0:box(s,left,y+(ri+1)*rowh,cw,rowh,'EFF3E9')
            text(s,str(value),left+.08,y+(ri+1)*rowh+.065,cw-.13,rowh-.065,size=12)
            left+=cw
    return s


def build():
    s=slide('')
    box(s,0,0,13.333,7.5,INK)
    text(s,'DARIJA / SENTIMENT',.8,.7,10,.5,16,bold=True,fill=LIME)
    text(s,'Comprendre le sentiment.\nEn darija.',.8,2.0,9,2.1,48,bold=True,fill=WHITE)
    text(s,'Six méthodes, des résultats mesurés et leurs limites.',.85,4.55,11,1,24,fill='C0D0C4')
    text(s,'AfriSenti · campagne exploratoire · novembre 2026',.85,6.4,10,.5,15,fill=LIME)
    manifest[-1]['title']='Darija / Sentiment'
    candidate=next(r for r in ROWS if r['family']=='E')
    s=slide('Le candidat retenu est DarijaBERT-Arabizi','Le choix vient de la validation ; le test mesure ensuite ce choix.','comparison, cellule G.1')
    for x,value,label in [(0.8,'81,13 %','F1 validation moyen'),(5.0,'60,81 %','F1 test, graine 42'),(9.2,'60,06 %','Macro-F1 test')]:
        text(s,value,x,2.65,3.3,1,43,bold=True);text(s,label,x,3.8,3.4,.8,17,fill=MUTED)
    note(s,'La baisse de validation vers test est un résultat central. Elle ne justifie pas de changer le candidat après coup.')
    two('Le sentiment traverse les écritures','Arabe et Arabizi','هاد الخدمة زوينة بزاف\n\nhad lkhadma zwina bzzaf\n\nDeux graphies possibles, une opinion exprimée.','Un contexte difficile','Négation, graphies variables, français, emojis, répétitions et mélange d’alphabets.\n\nCes exemples pédagogiques n’ont pas de label humain validé.','Trois classes : négatif, neutre, positif.','Corpus et protocole, notebook final')
    table_slide('9 038 tweets, trois partitions officielles',['Partition','Négatif','Neutre','Positif','Total'],
        [[r['name'],r['negative'],r['neutral'],r['positive'],r['n']] for r in DATA['corpus']],
        'Train 5 583 · validation 494 · test 2 961. Les partitions restent intactes.','data/ary/*.tsv, révision figée')
    s=slide('Apprendre, choisir, figer, mesurer','Quatre rôles distincts pour éviter une nouvelle sélection sur test.','src/campaign.py et src/evaluate.py')
    for i,(title,body) in enumerate([('Train','Vocabulaire, embeddings, classifieurs et poids des réseaux.'),('Validation','Architecture, époque retenue et décalages de décision.'),('Verrou','Candidat, représentants, graine 42 et empreintes des artefacts.'),('Test','Qualité, écritures, perturbations, incertitude et coût.')]):
        x=.75+i*3.15;text(s,title,x,2.5,2.8,.6,26,bold=True);text(s,body,x,3.4,2.65,1.9,19,fill=MUTED)
    note(s,'Le test avait déjà été observé dans l’étude historique : une confirmation indépendante nécessite de nouveaux textes annotés.')
    two('Nettoyer sans effacer les indices de sentiment','La référence conservatrice','Normalisation Unicode et espaces.\nURL et mentions retirées.\n\nNégation, accents, français, chiffres Arabizi, emojis et répétitions conservés.','Des unités différentes','TF-IDF : texte complet.\nC / D / E : 128 tokens.\nChar-CNN : 512 caractères.\n\nUn mot n’est pas un sous-token BERT.','Le nettoyage de référence est figé avant les mesures test.','src/preprocess.py, src/models.py')
    table_slide('Les six familles comparées',['Famille','Méthode','Ce qui apprend'],[
        ['A','TF-IDF + régression logistique','Classifieur local'],['B','NLI / CAMeLBERT','Poids publics fixes'],
        ['C','BiLSTM, attention, Transformer, CNN','Réseau sur ary'],['D','DarijaBERT gelé + tête','Tête locale seule'],
        ['E','Fine-tuning dialectal + E2','Encodeur et tête'],['P','Nettoyage sans signes','TF-IDF sur texte transformé']],
        'Le laboratoire utilise un représentant par famille, choisi sur validation.','Code des familles et sorties sauvegardées')
    chart_slide('A : une référence simple et mesurable','TF-IDF mots, caractères et union ; C choisi sur validation.',
        bar_plot('A.png',[r for r in VAR if r['family']=='A'],'Weighted-F1 test'),
        'Le représentant A est le TF-IDF caractères, choisi sur validation, même si une autre variante gagne sur test.')
    chart_slide('B : le transfert fixe dépend du domaine','Les poids ne sont pas entraînés sur ary ; les modèles ont déjà été supervisés ailleurs.',
        bar_plot('B.png',[r for r in VAR if r['family']=='B'],'Weighted-F1 test'),
        'Les trois NLI changent langue et formulation. Une différence ne peut pas être attribuée à la langue seule.')
    two('C : comparer architectures et embeddings','Des encodeurs compacts','BiLSTM : 64 unités par direction.\nAttention masquée.\nTransformer : 2 couches, 4 têtes.\nChar-CNN : filtres 3, 5 et 7.','Un protocole commun','Embeddings de dimension 128.\nAléatoire, AraVec ou fastText.\nAdamW 10⁻³, batch effectif 32.\n15 époques maximum, trois graines.','Le vocabulaire est appris sur train, fréquence minimale 2.','src/models.py et src/train.py')
    crows=sorted([r for r in VAR if r['family']=='C'],key=lambda r:r['dev_weighted_f1_mean'],reverse=True)
    chart_slide('C : l’ordre de validation reste la référence','Les huit premières variantes C selon la validation sont montrées ici.',
        bar_plot('C.png',crows[:8],'Weighted-F1 test'),
        'Le meilleur couple architecture-embeddings est choisi sur la moyenne des graines, avant perte pondérée et oversampling.')
    chart_slide('C : les courbes décrivent les époques observées','Les sorties d’apprentissage sont sauvegardées ; elles ne prouvent pas une explication causale.',
        PROJECT/'site/dist/figures/curves_C.png',
        'Les époques perdues avant checkpoint ne sont pas récupérables dans ces mesures.','Notebook principal, cellule des historiques C/D/E/P')
    chart_slide('D : geler BERT ne rend pas son inférence gratuite','CLS, moyenne masquée et séquence avec tête BiLSTM-attention.',
        bar_plot('D.png',[r for r in VAR if r['family']=='D'],'Weighted-F1 test'),
        'DarijaBERT reste exécuté sur chaque texte libre. Le cache de représentations est un outil expérimental.')
    chart_slide('E1 : adapter l’encodeur au sentiment marocain','DarijaBERT et DarijaBERT-Arabizi, graines 42 / 43 / 44.',
        bar_plot('E1.png',[r for r in VAR if r['family']=='E' and '_mtl_' not in r['variant']],'Weighted-F1 test'),
        'AdamW 2×10⁻⁵, batch effectif 32, micro-batch 2, warmup 10 %, perte pondérée, jusqu’à cinq époques.')
    chart_slide('E2 : l’auxiliaire algérien n’est pas supposé aider','Deux valeurs de λ, une graine et trois époques maximum dans le plan réduit.',
        bar_plot('E2.png',[r for r in VAR if r['family']=='E' and (r['variant']=='arabizi' or '_mtl_' in r['variant'])],'Weighted-F1 test'),
        'Comparer à E1 avec prudence : le budget et le nombre de graines diffèrent. λ=0,7 n’a pas été évalué ici.')
    chart_slide('E : regarder aussi les courbes du fine-tuning','L’époque est retenue sur le F1 validation brut ; les décalages sont réglés ensuite.',
        PROJECT/'site/dist/figures/curves_E.png','Une courbe de validation n’est pas une nouvelle estimation de généralisation.','Historiques E sauvegardés')
    prows=[next(r for r in VAR if r['family']=='A' and r['variant']=='union'),next(r for r in VAR if r['family']=='P')]
    chart_slide('P : changer le pipeline de nettoyage','Réapprendre un TF-IDF union sans emojis ni ponctuation.',bar_plot('P.png',prows,'Weighted-F1 test'),
        'P DarijaBERT conserve ses scores dev mais est exclu du test : le poids manque. Aucun score test ne lui est attribué.')
    fig,ax=plt.subplots(figsize=(11.3,4.3));y=np.arange(6)
    for i,r in enumerate(ROWS):
        ax.plot([r['weighted_f1'],r['dev_weighted_f1_mean']],[i,i],color='#9BAF8B',linewidth=3)
        ax.scatter([r['dev_weighted_f1_mean']],[i],facecolors='#'+PAPER,edgecolors='#729267',s=80)
        ax.scatter([r['weighted_f1']],[i],color='#'+INK,s=55)
    ax.set_yticks(y,[r['family']+' · '+NAMES[r['family']] for r in ROWS]);ax.invert_yaxis();ax.set_xlim(.3,.9);ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));ax.grid(axis='x',alpha=.15)
    chart_slide('Validation et test : la baisse concerne les six familles','Cercle ouvert : validation moyenne. Cercle plein : test de la graine 42.',save_plot('generalisation.png'),
        'La variation d’écriture, les distributions conditionnelles et les raccourcis sont des pistes, pas des causes démontrées.')
    chart_slide('F1 : les proportions globales ne suffisent pas','Les distributions restent descriptives, après verrouillage.',PROJECT/'site/dist/figures/F1_distributions.png',
        'Un changement de proportions ne démontre pas un pur label shift. Examiner aussi les distributions conditionnelles.')
    cond=DATA['tables']['f1_conditional_labels'];neutral=[r for r in cond if r['label']=='neutral']
    rows=[]
    for script in ['arabic','latin','mixed']:
        rows.append([{'arabic':'Arabe','latin':'Latine','mixed':'Mixte'}[script]]+[f"{next((r['proportion'] for r in neutral if r['script']==script and r['split']==split),0):.1%}" for split in ['train','dev','test']])
    table_slide('F1 : le neutre varie selon l’écriture',['Écriture','Train','Validation','Test'],rows,
        'La relation alphabet / sentiment doit être examinée conditionnellement.','Distributions affichées, cellule F1.1')
    chart_slide('F1 : le maximum sur test reste un oracle','La grille post hoc ne modifie ni le candidat ni sa règle.',PROJECT/'site/dist/figures/F1_threshold_grid.png',
        'Optimiser de nouveau sur cette grille test invaliderait la séparation des rôles du protocole.')
    cross=DATA['tables']['f2_cross_script']
    table_slide('F2 : transférer vers l’écriture non vue',['Train','Test','N','Weighted-F1'],
        [[r['train_script'],r['test_script'],r['n'],f"{r['weighted_f1']:.4f}"] for r in cross],
        'Le TF-IDF union est entraîné sur une seule écriture.','Sortie F2 du notebook Colab')
    s=prs.slides[-1];note(s,'La taille du train et sa couverture changent ensemble. L’écart n’isole pas causalement l’alphabet.')
    matrix=np.full((6,3),np.nan)
    for i,r in enumerate(ROWS):
        for j,script in enumerate(['arabic','latin','mixed']):
            item=next((x for x in r['scripts'] if x['script']==script),None)
            if item:matrix[i,j]=item['weighted_f1']
    fig,ax=plt.subplots(figsize=(9,4.4));im=ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGn');ax.set_xticks(range(3),['Arabe','Latine','Mixte']);ax.set_yticks(range(6),[r['family']+' · '+NAMES[r['family']] for r in ROWS])
    for i in range(6):
        for j in range(3):ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',color='#183C35',fontsize=12)
    fig.colorbar(im,ax=ax,label='Weighted-F1 test')
    chart_slide('F3 : une moyenne peut cacher un sous-groupe','Résultats des six représentants par écriture.',save_plot('ecritures.png'),
        'Le groupe mixte peut être petit. Le détecteur identifie l’alphabet, pas la langue.')
    chart_slide('F4 : mesurer les bascules de décision','Même population test pour les trois perturbations déclarées.',PROJECT/'site/dist/figures/F4_bascules.png',
        'Les labels originaux sont conservés pour ce diagnostic. Une transformation peut changer le sentiment véritable.')
    fig,ax=plt.subplots(figsize=(11.2,4.2))
    for i,r in enumerate(ROWS):
        ax.errorbar(r['weighted_f1'],i,xerr=[[r['weighted_f1']-r['ci_low']],[r['ci_high']-r['weighted_f1']]],fmt='o',color='#'+COLORS[r['family']],capsize=5)
    ax.set_yticks(range(6),[r['family']+' · '+NAMES[r['family']] for r in ROWS]);ax.invert_yaxis();ax.set_xlim(.35,.7);ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));ax.grid(axis='x',alpha=.15)
    chart_slide('F5 : bootstrap et graines mesurent deux variabilités','IC bootstrap 95 %, conditionnel au checkpoint 42, sur 1 000 rééchantillonnages.',save_plot('incertitude.png'),
        'L’écart-type entre graines décrit les entraînements. Il ne remplace pas l’intervalle d’échantillonnage des tweets.')
    mc=DATA['json']['f5_mcnemar']
    two('F5 : comparer des erreurs appariées','McNemar exact',f"Premier : {mc.get('first','')}\n\nSecond : {mc.get('second','')}\n\np exact = {mc.get('p_exact','n.d.')}",'Le sens du test','Comparer les deux ensembles de tweets corrects pour un seul modèle.\n\nCe n’est pas un test direct de différence de F1. Les deux modèles sont choisis sur validation.','Le score et sa significativité restent des questions distinctes.','McNemar, cellule F5')
    errors=DATA['tables']['f6_30_erreurs'];validated=sum(r.get('validation_humaine')=='validé' for r in errors)
    two('F6 / F7 : la revue humaine reste indispensable','Trente erreurs à relire',f'{validated} / {len(errors)} validations humaines enregistrées.\n\nLes catégories automatiques sont provisoires : ironie, graphie, mélange, neutre/polarisé, annotation.','Pas de score indépendant inventé','Les cas linguistiques construits doivent être annotés.\n\nLe nouvel ensemble indépendant demande collecte, source, date, relecteur et contrôle des quasi-doublons.','Un échantillon diversifié ne mesure pas la fréquence des catégories dans toutes les erreurs.','F6 et F7, outputs enregistrés')
    chart_slide('G : qualité, latence CPU et taille des poids','Graphique original enregistré par la campagne.',PROJECT/'site/dist/figures/G_quality_cost.png',
        'La latence utilise trente inférences unitaires après échauffement. Le poids des fichiers n’est pas un pic de RAM.')
    two('Ce que les résultats permettent de dire','Le résultat mesuré','DarijaBERT-Arabizi est le candidat fixé sur validation.\n\nF1 test : 0,608138.\nIC 95 % : [0,590655 ; 0,624099].\nLa baisse vers test doit être analysée.','Les limites à conserver','Test historique déjà consulté.\nSous-groupes parfois petits.\nE2/P à budget réduit.\nP DarijaBERT exclu du test.\nRevue humaine inachevée.','La prochaine confirmation doit porter sur de nouveaux exemples annotés indépendamment.','Synthèse du notebook final')
    s=slide('Trois livrables, une même provenance','Le code, les résultats et les limites restent reliés aux sorties enregistrées.','Dépôt GitHub OmarElkhali/DARIJA-MACHA3IR')
    for i,(title,body) in enumerate([('Notebook final','Code complet, cellules utiles, outputs conservés et conclusion mesurée. Aucune exécution pour son assemblage.'),('Site HTML/CSS/JS','Corpus, méthodes, 33 variantes, graphiques, analyses et laboratoire relié aux checkpoints disponibles.'),('Dépôt GitHub','Sources, données figées, livrables et scripts de reconstruction. Les poids volumineux restent externes.')]):
        y=2.15+i*1.25;text(s,title,.8,y,3.3,.5,23,bold=True);text(s,body,4.4,y,7.8,.93,18,fill=MUTED)
    note(s,'github.com/OmarElkhali/DARIJA-MACHA3IR')
    two('Les références','Corpus et tâche','AfriSenti, Muhammad et al. (2023).\nACL : 2023.emnlp-main.862\n\nAfriSenti-SemEval Task 12.\nACL : 2023.semeval-1.315','Modèles et représentations','DarijaBERT, Gaanoun et al.\nDOI : 10.1007/s41060-023-00498-2\n\nfastText, Bojanowski et al. (2017).\nACL : Q17-1010','Les résultats publiés par leurs auteurs ne sont pas les scores mesurés dans cette campagne.','research/papers.md et sources primaires')
    target=OUT/'Darija_Sentiment_Resultats_2026.pptx'
    prs.save(target)
    with ZipFile(target) as zipped:assert zipped.testzip() is None
    layout=[]
    for i,s in enumerate(prs.slides,1):
        for shape in s.shapes:
            if shape.left<0 or shape.top<0 or shape.left+shape.width>prs.slide_width+1000 or shape.top+shape.height>prs.slide_height+1000:
                layout.append({'slide':i,'shape':shape.shape_id,'issue':'outside_slide'})
    assert not layout,layout
    (OUT/'manifest.json').write_text(json.dumps({'slides':manifest,'source':'site/dist/data.json, outputs sauvegardés','notebook_executed':False,'layout_errors':layout},ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'notes_orales.md').write_text('\n\n'.join(f"## {i+1}. {m['title']}\n\n{prs.slides[i].notes_slide.notes_text_frame.text}" for i,m in enumerate(manifest)),encoding='utf-8')
    # Aperçus depuis les mêmes coordonnées et textes, sans piloter PowerPoint.
    render_previews()
    print(f'{len(prs.slides)} diapositives éditables : {target}')


def render_previews():
    scale=96
    for index,s in enumerate(prs.slides):
        im=Image.new('RGB',(1280,720),'#'+(INK if index==0 else PAPER));draw=ImageDraw.Draw(im)
        for op in s._preview_ops:
            if op[0]=='box':
                _,x,y,w,h,fill=op;draw.rectangle((x*scale,y*scale,(x+w)*scale,(y+h)*scale),fill='#'+fill)
            elif op[0]=='image':
                _,path,x,y,w,h=op
                with Image.open(path) as pic:
                    pic=pic.convert('RGB').resize((max(1,int(w*scale)),max(1,int(h*scale))))
                    im.paste(pic,(int(x*scale),int(y*scale)))
            else:
                _,content,x,y,w,h,size,bold,fill=op
                font=ImageFont.truetype(str(BOLD if bold else FONT),max(8,int(size*96/72)))
                lines=[]
                for paragraph in str(content).split('\n'):
                    words=paragraph.split();line=''
                    if not words:lines.append('');continue
                    for word in words:
                        candidate=(line+' '+word).strip()
                        if draw.textlength(candidate,font=font)>w*scale and line:
                            lines.append(line);line=word
                        else:line=candidate
                    lines.append(line)
                draw.multiline_text((x*scale,y*scale),'\n'.join(lines),font=font,fill='#'+fill,spacing=7)
        im.save(OUT/f'slide_{index+1:02d}.png')
        previews.append(im.resize((384,216)))
    cols=4;rows=math.ceil(len(previews)/cols)
    sheet=Image.new('RGB',(cols*384,rows*240),'#E2E7DF');draw=ImageDraw.Draw(sheet)
    for i,img in enumerate(previews):
        x=i%cols*384;y=i//cols*240;sheet.paste(img,(x,y));draw.text((x+8,y+220),str(i+1),fill='#183C35')
    sheet.save(OUT/'planche_apercus.png')


if __name__=='__main__':build()
