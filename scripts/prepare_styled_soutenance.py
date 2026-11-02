"""Préparer la mise à jour codex-ppt de la soutenance, sans exécuter les notebooks."""
from pathlib import Path
import argparse
import copy
import json

PROJECT=Path(__file__).resolve().parents[1]
OLD=PROJECT/'presentation/Darija_Sentiment_Soutenance'
OUT=PROJECT/'presentation/Darija_Sentiment_Soutenance_Resultats'
OUT.mkdir(exist_ok=True)
(OUT/'origin_image').mkdir(exist_ok=True)
DATA=json.loads((PROJECT/'site/dist/data.json').read_text(encoding='utf-8'))
style_source=OLD/'deck_spec.json'
if not style_source.exists():
    style_source=OUT/'deck_spec.json'
old=json.loads(style_source.read_text(encoding='utf-8'))

# Chaque page conserve la fonction narrative et la direction visuelle de la soutenance.
# Les textes anciens sont remplacés entièrement pour éviter de mélanger les campagnes.
pages=[
(1,'Analyse de sentiment en darija marocaine',
 ['Trois sentiments. Plusieurs écritures.','Six familles comparées sur AfriSenti ary','Groupe 12 · Master Big Data'],
 'cover','Grande couverture éditoriale : titre sérif à gauche ; à droite illustration à l’encre de fragments arabe/Arabizi qui se rejoignent, branche de sauge et mot مشاعر correctement composé. Aucun score sur la couverture.'),
(2,'Une langue, plusieurs graphies',
 ['Arabe : هاد الخدمة زوينة بزاف','Arabizi : hadchi zwin bzaf','Mixte : هاد الفيلم زوين mais la fin ma3jbatnich','Exemples illustratifs, sans prédiction mesurée'],
 'concept','Trois lignes linguistiques très lisibles, annotations visuelles des chiffres Arabizi et de la négation. Illustration de flux de paroles. Les phrases sont distinctes : ne jamais les appeler traductions ni même contenu.'),
(3,'Train apprend, validation choisit, test mesure',
 ['Train : 5 583 tweets','Validation : 494 tweets','Test : 2 961 tweets','Apprendre → choisir → verrouiller → mesurer'],
 'process','Un grand processus horizontal avec trois étapes et un cadenas avant le test. Train : vocabulaire et poids. Validation : variante, époque, décalages. Test : représentant graine 42. Pied lisible : « Test déjà consulté historiquement : campagne exploratoire ». Aucune flèche du test vers le réglage.'),
(4,'Lire les distributions avant les scores',
 ['9 038 tweets · trois classes','Neutre majoritaire sur train','Majorité seule : F1 test 0,160795','F1 pondéré ≠ accuracy'],
 'data evidence','Graphique de distributions fourni à gauche, grand et inchangé. À droite un encadré éditorial sans carte avec le score de la majorité, puis explication : F1 pondéré selon les effectifs, Macro-F1 donne le même poids aux classes. Mention « Distribution test : analyse descriptive après verrouillage ».', 'F1_distributions.png'),
(5,'L’écriture peut devenir un raccourci',
 ['Part de neutres selon l’écriture','Arabe : train 29,2 % → test 36,1 %','Latin : train 52,3 % → test 29,9 %','Mixte : train 41,5 % → test 25,3 %'],
 'comparison','Deux colonnes Train et Test reliées par des lignes, trois rangées arabe/latin/mixte avec leurs pourcentages exacts. Petit schéma « écriture ↔ sentiment » barré comme raccourci fragile. Lien descriptif, pas causal. Source : distributions conditionnelles enregistrées.'),
(6,'Nettoyer sans effacer les indices : expérience P',
 ['Référence : conserver négation, chiffres Arabizi et emojis','P : réapprendre sans emojis ni ponctuation','TF-IDF union : F1 test 0,539908','P union : F1 test 0,567396'],
 'process','Moitié gauche : exemple illustratif « ma3jbnich 😒 !! » avant/après « ma3jbnich », étiqueté illustration. Moitié droite : deux barres horizontales sur axe 0 à 1 avec valeurs exactes A union et P union. En bas : « P DarijaBERT exclu du test ; scores validation conservés ». Distinguer réentraînement P de perturbation F4.'),
(7,'A — Comprendre TF-IDF et les caractères',
 ['Mots et fragments de caractères → vecteur TF-IDF','Régression logistique → trois sentiments','Choix sur validation : caractères, F1 0,764076','F1 test du représentant : 0,553509'],
 'architecture','Une phrase illustratrice devient une liste de fragments, puis une matrice de poids, puis trois sorties sans probabilités inventées. En dessous deux nombres validation/test clairement légendés. Petits détails : mots 1–2 ; caractères 2–5 ; C choisi sur validation.'),
(8,'B1 — Reformuler le sentiment comme une hypothèse',
 ['Tweet = prémisse','Une hypothèse pour chaque sentiment','Trois formulations : arabe, français, anglais','F1 validation ajusté : 0,501026 · 0,497245 · 0,463444'],
 'architecture','Schéma pédagogique à gauche : un tweet illustratif rejoint trois hypothèses puis un bloc NLI qui score l’implication. À droite tableau de trois lignes arabe/français/anglais avec les F1 validation exacts. En bas : « Langue et formulation varient ensemble ; leur effet n’est pas isolé ». Pas de probabilité inventée.'),
(9,'B2 — Le transfert fixe dépend de l’écriture',
 ['CAMeLBERT sans entraînement local sur ary','F1 test arabe : 0,558','F1 test latin : 0,144','Supervision externe : contamination non exclue'],
 'comparison','Deux grandes barres arabe/latin, même axe F1 de 0 à 1 ; chiffres présentés arrondis à trois décimales. Effectifs : arabe 1 367 ; latin 1 519. Petit schéma modèle sentiment préentraîné → tweets marocains. Ne pas prétendre que le modèle n’a jamais été supervisé.'),
(10,'Déséquilibre : agir sur l’apprentissage et la décision',
 ['Référence C : BiLSTM + AraVec','Validation : référence 0,753140','Pondération : 0,743939 · suréchantillonnage : 0,743941','Décision : argmax(log p + b)'],
 'concept','À gauche deux interventions sur train : pondérer la perte et rééchantillonner, avec pictogrammes précis. À droite trois points ou petites barres validation sur même axe, valeurs six décimales. Bandeau final : « Les décalages b sont choisis sur validation, jamais sur test ». Comparaison sur trois graines.'),
(11,'C — Des embeddings à la décision',
 ['Hasard, AraVec ou fastText','BiLSTM : lecture dans les deux directions','Attention : pondérer les représentations','Choix C : BiLSTM + AraVec, sans attention'],
 'architecture','Grande architecture illustrée : tokens → embeddings → deux chaînes LSTM en sens opposés → représentation → trois classes. Dessiner un embranchement en pointillés pour attention optionnelle. Ne pas inventer de poids d’attention ni réutiliser les anciens exemples numériques. Légende « Attention descriptive ≠ preuve causale ». Diagramme clair, très pédagogique.'),
(12,'C — Les courbes complètent le tableau des scores',
 ['BiLSTM + AraVec retenu sur validation','Validation moyenne : 0,753140','F1 test, graine 42 : 0,493340','Les courbes décrivent les époques observées'],
 'data evidence','Courbes originales fournies très grandes au centre, occupent 70 % de la page ; conserver axes, traces et légende. À droite ou en bas une courte colonne des deux scores et « Époque retenue sur validation ». Pas de tendance causale affirmée à partir des courbes.', 'curves_C.png'),
(13,'D et E — Geler ou ajuster l’encodeur',
 ['D : DarijaBERT gelé, seule la tête apprend','E : les poids de l’encodeur sont aussi ajustés','D states : F1 test 0,571593','E Arabizi : F1 test 0,608138'],
 'architecture','Deux architectures miroirs : D encodeur avec cadenas, tête apprenable ; E encodeur et tête apprenables. Flèches de gradients en terre cuite seulement sur les parties apprenables. Une comparaison complémentaire en bas : E DarijaBERT standard 0,579512 ; E Arabizi 0,608138, représentants graine 42. Aucune causalité isolée entre taille/coût et qualité.'),
(14,'E2 — Une tâche auxiliaire ne garantit pas un gain',
 ['Perte : L_ary + λ L_arq','λ = 0 : F1 test 0,608138','λ = 0,1 : 0,605137 · λ = 0,3 : 0,604073','Plan réduit : une graine, trois époques maximum'],
 'architecture','Illustration d’un encodeur partagé à deux têtes ary/arq ; tableau trois lignes des valeurs exactes à droite. En bas : « λ = 0,7 non évalué. Budgets et nombres de graines différents : comparaison exploratoire ». Le corpus principal reste ary ; arq intervient seulement en auxiliaire E2.'),
(15,'La sélection se fait avant la comparaison finale',
 ['Critère : F1 validation de chaque variante','E Arabizi : moyenne 0,811256','Représentant de chaque famille : graine 42','33 variantes · 71 évaluations finales'],
 'data evidence','Classement validation des six familles : E 0,811256 ; A 0,764076 ; C 0,753140 ; D 0,746022 ; P 0,735605 ; B 0,602343. Axe commun 0 à 1. Grand cadenas et texte « Candidat fixé : E Arabizi ». Préciser « Moyenne des graines disponibles ; toutes les familles n’ont pas trois graines ». Ne pas choisir après lecture du test.'),
(16,'Résultats sur le test officiel',
 ['Weighted-F1 · représentants graine 42','E Arabizi : 0,608138','Macro-F1 de E : 0,600620','Choix fixé sur validation'],
 'data evidence','Reprendre très fidèlement la mise en page de la diapositive 16 de la soutenance d’origine : graphique à barres horizontales à gauche, grand résultat typographique à droite et filets fins. SIX barres A TF-IDF caractères 0,553509 ; B CAMeLBERT 0,420555 ; C BiLSTM + AraVec 0,493340 ; D DarijaBERT gelé 0,571593 ; E DarijaBERT-Arabizi 0,608138 ; P TF-IDF sans signes 0,567396. Axe 0 à 1, longueurs proportionnelles. E vert forêt ; autres sauge. À droite grand « 0,608138 », « Candidat choisi sur validation », « Macro-F1 : 0,600620 ». Bas : « AfriSenti ary · n = 2 961 · campagne exploratoire ». SUPPRIMER toute ancienne mention E non exécuté ou A candidat.'),
(17,'Lire les erreurs du candidat, classe par classe',
 ['Candidat : E DarijaBERT-Arabizi','La diagonale indique les prédictions correctes','Les cases hors diagonale montrent les confusions','Un score global ne suffit pas'],
 'data evidence','Matrice originale du nouveau candidat très grande, totalement inchangée. À droite trois annotations génériques expliquant lignes, colonnes et diagonale sans inventer de nombre de cellules. Ne pas réutiliser la matrice de l’ancienne campagne A.', 'modele_final.png'),
(18,'Validation → test : une baisse à expliquer',
 ['E Arabizi : 0,811256 → 0,608138','Les six familles perdent en généralisation','Le maximum d’une grille test reste un oracle','Aucun réglage test ne remplace le verrou'],
 'comparison','Grand graphique dumbbell de six lignes : validation vers test. E 0,811256→0,608138 ; A 0,764076→0,553509 ; C 0,753140→0,493340 ; D 0,746022→0,571593 ; P 0,735605→0,567396 ; B 0,602343→0,420555. Cercles ouverts validation, pleins test, légende exacte, axe0à1. Note « Validation : moyenne disponible ; test : représentant 42 ». À droite petit schéma d’oracle barré, jamais de score optimisé inventé.'),
(19,'Écriture non vue : le transfert reste difficile',
 ['TF-IDF union, même architecture','Train arabe → test latin : F1 0,2254','Train latin → test arabe : F1 0,2110','Taille et couverture du train changent ensemble'],
 'comparison','Deux rangées horizontales séparées, SANS croisement. Rangée haute verte : « Train arabe » à gauche → flèche portant « F1 test : 0,2254 » → « Test latin · n = 1 519 » à droite. Rangée basse terre cuite : « Train latin » à gauche → flèche portant « F1 test : 0,2110 » → « Test arabe · n = 1 367 » à droite. Les labels doivent être placés directement sur leur propre flèche et non sur l’autre rangée. En bas : « Ce diagnostic ne mesure pas un effet causal pur de l’alphabet ». Interdire les anciennes valeurs 0,2098/0,1932.'),
(20,'Incertitude : situer le score, pas seulement le classer',
 ['E : IC bootstrap 95 % [0,590655 ; 0,624099]','1 000 rééchantillonnages conditionnels au checkpoint','Graines et bootstrap : deux variabilités distinctes','McNemar E / E2 λ=0,3 : p = 1,0'],
 'data evidence','Forest plot six lignes, points et intervalles exacts : A .553509 [.535264,.571928] ; B .420555 [.401828,.439308] ; C .493340 [.475292,.511506] ; D .571593 [.552955,.589547] ; E .608138 [.590655,.624099] ; P .567396 [.548377,.585336]. Axe de0,35à0,65 explicitement indiqué (ne pas commencer à0visuellement). À droite expliquer « McNemar compare les erreurs appariées, pas directement le F1 ». Aucune conclusion de supériorité statistique déduite du simple ordre.'),
(20,'G — Qualité, latence CPU et taille des poids',
 ['Mettre la qualité en regard du coût','Latence : inférence CPU unitaire, pas débit GPU','Poids sur disque ≠ mémoire vive','Temps cumulés : sans doubler raw/tuned ou reprises'],
 'data evidence','Insérer le graphique G original comme preuve centrale très grande et inchangée. Autour seulement trois annotations courtes pour interpréter les axes et le coût. Ne jamais recopier les anciens coûts ni inventer les colonnes masquées dans l’affichage. Aucun nouveau chiffre de coût.', 'G_quality_cost.png'),
(21,'Trente erreurs : comprendre avant de compter',
 ['30 exemples à relire','0 / 30 validations humaines enregistrées','Ironie, graphie, ambiguïté : hypothèses provisoires','Un désaccord ne prouve pas une erreur de label'],
 'concept','Illustration éditoriale de loupe et annotations sur trois fragments schématiques sans faux tweets réels. À droite processus « erreur observée → hypothèse → revue humaine → conclusion ». Ne pas reprendre les anciens comptes 8/8/7/7 comme résultats validés. Pas de taux de bruit inventé.'),
(22,'Du benchmark au laboratoire interactif',
 ['Site HTML · CSS · JavaScript','Une phrase → six familles affichées','Cinq modèles testés réellement via le moteur local','E : checkpoint absent localement, aucune simulation'],
 'process','Schéma fonctionnel élégant : zone de phrase illustrative → serveur local Python → six branches A/B/C/D/E/P. Cinq branches actives, E en contour pointillé avec étiquette « poids à restaurer ». Pas de capture fabriquée présentée comme réelle, pas de prédictions chiffrées inventées. URL lisible « omarelkhali.github.io/DARIJA-MACHA3IR ». Le site statique a besoin du moteur local pour prédire.'),
(23,'Ce que nous retenons du projet',
 ['Candidat mesuré : DarijaBERT-Arabizi','F1 test : 0,608138 · Macro-F1 : 0,600620','Les écarts par écriture restent décisifs','Prochaine étape : de nouvelles annotations indépendantes'],
 'summary','Composition finale très forte : un grand chiffre 0,608138 à gauche, puis trois enseignements visuels à droite : protocole, généralisation, coût. Bas : « Test historique déjà consulté · revue humaine inachevée · checkpoint E à restaurer ». Conclusion conditionnelle, aucune promesse de fiabilité universelle.'),
(13,'Annexe — Lire les courbes de fine-tuning',
 ['E : apprentissage et validation par époque','Sélection de l’époque sur validation','Les courbes ne prouvent pas une explication causale','Aucune nouvelle exécution pour cette présentation'],
 'data evidence','Courbes E fournies très grandes et conservées sans modification. Deux petites annotations hors figure : apprentissage / validation. Conserver les courbes de tous les runs présents, ne pas fabriquer de valeurs ou lisser les traces.', 'curves_E.png'),
(24,'Références et guide pour les questions',
 ['AfriSenti · Muhammad et al., 2023','DarijaBERT · Gaanoun et al.','fastText · Bojanowski et al., 2017','Code, notebooks et résultats : DARIJA-MACHA3IR'],
 'reference','Bibliographie courte à gauche, index thématique Q1–Q12 à droite : Q1/Q2 données et protocole ; Q3 nettoyage/P ; Q4/Q5 NLI/transfert ; Q6 déséquilibre ; Q7 réseaux ; Q8 D/E ; Q9 multi-tâche ; Q10 diagnostics test ; Q11 écritures ; Q12 erreurs/coûts. Petit mot arabe مشاعر et branche de sauge de la référence. Les références ne sont pas les mesures de notre campagne.')
]

slides=[]
for number,item in enumerate(pages,1):
    old_number,title,points,role,composition,*asset=item
    slide={'number':number,'title':title,'role':role,'intent':'Une idée expliquée visuellement, une preuve mesurée et sa limite.',
           'key_points':points,'local_context':'Les seuls faits à afficher sont dans ce job. Les anciens textes et scores de la référence ne sont pas des données actuelles.',
           'layout':{'composition':composition,'spacing':'Grandes marges, texte lisible en projection, varier les compositions comme la soutenance de référence.'},
           'visual_elements':{'main_visual':composition},
           'constraints':['Texte français exact, accents et arabe correctement formé.','Aucun numéro de page, filigrane ou logo extérieur.',
                          'Aucun résultat inventé. Conserver exactement les données des sources strictes.',
                          'La référence ancienne sert uniquement au style et à la mise en page ; ne pas reprendre ses anciens résultats.'],
           'legacy_style_slide':str(OLD/f'origin_image/slide_{old_number:02d}.png')}
    if asset:
        slide['required_images']=[{'path':str(PROJECT/'site/dist/figures'/asset[0]),'role':'strict input asset, figure de la campagne actuelle',
             'fidelity':'Preserve actual data, axes, labels, traces, legend and values; embed unchanged, never synthesize a similar graph.'}]
    if number==20:
        composition='Un tableau numérique exact remplace le forest plot pour éviter toute approximation géométrique : colonnes Méthode / F1 test / IC95%. Six lignes : A0,553509 [0,535264 ;0,571928] ; B0,420555 [0,401828 ;0,439308] ; C0,493340 [0,475292 ;0,511506] ; D0,571593 [0,552955 ;0,589547] ; E0,608138 [0,590655 ;0,624099] ; P0,567396 [0,548377 ;0,585336]. À droite grand IC de E et McNemar p=1,0. Le test compare les erreurs appariées, pas directement le F1. Pas de graphique ni figure source sur cette page. Style de tableau éditorial de la soutenance originale.'
        slide['layout']['composition']=composition
        slide['visual_elements']['main_visual']=composition
    slides.append(slide)

spec={'deck_name':OUT.name,'language':'French','goal':'Actualiser la soutenance originale avec les résultats complets et une explication visuelle fidèle.',
      'selected_image_backend':'built-in image tool','max_concurrent_slides':3,'style':old['style'],
      'deck_context':{'source_summary':'AfriSenti ary : train5583, validation494, test2961. Six familles A/B/C/D/E/P, 33 variantes, 71 évaluations finales. E1 exécuté ; E2 réduit λ0.1/0.3 exécuté. P DarijaBERT exclu du test.',
        'core_claim':'E Arabizi est choisi sur validation (moyenne .811256) ; représentant42 test weightedF1 .608138, macro .600620. Étude exploratoire, test historiquement consulté. Le laboratoire possède cinq checkpoints, pas E.',
        'canonical_terms':['ary : tâche principale','arq : auxiliaire E2 seulement','Weighted-F1 ≠ accuracy','IC conditionnel au checkpoint','Aucune exécution de notebook pour ce livrable']},
      'slides':slides,'authorization':{'source':'Instructions utilisateur persistantes et approvals.json de la soutenance originale',
        'instruction':'Sans me questionner sur rien ; même manière que Darija_Sentiment_Soutenance en changeant les résultats',
        'intermediate_confirmations_waived':True}}

outline=['# Darija Sentiment — Soutenance actualisée','',
 '26 diapositives. Même direction éditoriale et même méthode image-based que Darija_Sentiment_Soutenance.',
 'Les confirmations intermédiaires sont levées par la consigne utilisateur persistante. Les références anciennes ne fournissent aucun score actuel.','']
for slide in slides:
    outline += [f"## Slide {slide['number']} : {slide['title']}",f"- Rôle : {slide['role']}"]
    outline += ['- '+point for point in slide['key_points']]
    outline += ['- Visuel : '+slide['layout']['composition']]
    for asset in slide.get('required_images',[]):
        outline += [f"- Figure stricte : ![{Path(asset['path']).stem}]({asset['path']})"]
    outline += ['']
(OUT/'outline.md').write_text('\n'.join(outline),encoding='utf-8')
(OUT/'approvals.json').write_text(json.dumps(spec['authorization'],ensure_ascii=False,indent=2),encoding='utf-8')

parser=argparse.ArgumentParser()
parser.add_argument('--finalize',action='store_true')
args=parser.parse_args()
if args.finalize:
    sample=OUT/'origin_image/slide_16.png'
    assert sample.is_file()
    spec['slides'][15]['sample_approved']=True
    spec['approved_style_reference']={'path':str(sample),'role':'approved current sample, style only','fidelity':'palette, typography, illustration language; vary layout'}
    spec['sample_generation_method']={'backend_used':'built-in image tool','tool_name':'image_gen__imagegen',
        'mode':'generate','model_config':'built-in defaults; exact model and size not exposed',
        'prompt_source':str(OUT/'sample_prompt.txt'),'approved_sample_path':str(sample),
        'input_context_preparation':'view_image then referenced_image_paths; reference is style/layout only',
        'handoff_rule':'Same built-in tool, generate a new full slide using style-only references; no local rendering or overlays.'}
    (OUT/'deck_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
else:
    (OUT/'deck_spec.draft.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    sample_prompt='Generate a NEW complete 16:9 French presentation slide. Image 1 is ONLY the style/layout reference from the earlier soutenance. Keep its exquisite ivory/forest/sage editorial system, huge elegant serif title and warm paper, fine rules, مشاعر branch motif. Replace ALL its content using this CURRENT slide brief. No overlays, output the finished slide image itself.\n'+json.dumps(slides[15],ensure_ascii=False,indent=2)
    (OUT/'sample_prompt.txt').write_text(sample_prompt,encoding='utf-8')
print(f'{len(slides)} diapositives préparées : {OUT}')
