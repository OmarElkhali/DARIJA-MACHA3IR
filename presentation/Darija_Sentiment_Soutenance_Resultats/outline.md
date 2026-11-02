# Darija Sentiment — Soutenance actualisée

26 diapositives. Même direction éditoriale et même méthode image-based que Darija_Sentiment_Soutenance.
Les confirmations intermédiaires sont levées par la consigne utilisateur persistante. Les références anciennes ne fournissent aucun score actuel.

## Slide 1 : Analyse de sentiment en darija marocaine
- Rôle : cover
- Trois sentiments. Plusieurs écritures.
- Six familles comparées sur AfriSenti ary
- Groupe 12 · Master Big Data
- Visuel : Grande couverture éditoriale : titre sérif à gauche ; à droite illustration à l’encre de fragments arabe/Arabizi qui se rejoignent, branche de sauge et mot مشاعر correctement composé. Aucun score sur la couverture.

## Slide 2 : Une langue, plusieurs graphies
- Rôle : concept
- Arabe : هاد الخدمة زوينة بزاف
- Arabizi : hadchi zwin bzaf
- Mixte : هاد الفيلم زوين mais la fin ma3jbatnich
- Exemples illustratifs, sans prédiction mesurée
- Visuel : Trois lignes linguistiques très lisibles, annotations visuelles des chiffres Arabizi et de la négation. Illustration de flux de paroles. Les phrases sont distinctes : ne jamais les appeler traductions ni même contenu.

## Slide 3 : Train apprend, validation choisit, test mesure
- Rôle : process
- Train : 5 583 tweets
- Validation : 494 tweets
- Test : 2 961 tweets
- Apprendre → choisir → verrouiller → mesurer
- Visuel : Un grand processus horizontal avec trois étapes et un cadenas avant le test. Train : vocabulaire et poids. Validation : variante, époque, décalages. Test : représentant graine 42. Pied lisible : « Test déjà consulté historiquement : campagne exploratoire ». Aucune flèche du test vers le réglage.

## Slide 4 : Lire les distributions avant les scores
- Rôle : data evidence
- 9 038 tweets · trois classes
- Neutre majoritaire sur train
- Majorité seule : F1 test 0,160795
- F1 pondéré ≠ accuracy
- Visuel : Graphique de distributions fourni à gauche, grand et inchangé. À droite un encadré éditorial sans carte avec le score de la majorité, puis explication : F1 pondéré selon les effectifs, Macro-F1 donne le même poids aux classes. Mention « Distribution test : analyse descriptive après verrouillage ».
- Figure stricte : ![F1_distributions](D:\SetupGame\Desktop\TP LLM\DARIJA SENTIEMENTS\Darija-Sentiment\site\dist\figures\F1_distributions.png)

## Slide 5 : L’écriture peut devenir un raccourci
- Rôle : comparison
- Part de neutres selon l’écriture
- Arabe : train 29,2 % → test 36,1 %
- Latin : train 52,3 % → test 29,9 %
- Mixte : train 41,5 % → test 25,3 %
- Visuel : Deux colonnes Train et Test reliées par des lignes, trois rangées arabe/latin/mixte avec leurs pourcentages exacts. Petit schéma « écriture ↔ sentiment » barré comme raccourci fragile. Lien descriptif, pas causal. Source : distributions conditionnelles enregistrées.

## Slide 6 : Nettoyer sans effacer les indices : expérience P
- Rôle : process
- Référence : conserver négation, chiffres Arabizi et emojis
- P : réapprendre sans emojis ni ponctuation
- TF-IDF union : F1 test 0,539908
- P union : F1 test 0,567396
- Visuel : Moitié gauche : exemple illustratif « ma3jbnich 😒 !! » avant/après « ma3jbnich », étiqueté illustration. Moitié droite : deux barres horizontales sur axe 0 à 1 avec valeurs exactes A union et P union. En bas : « P DarijaBERT exclu du test ; scores validation conservés ». Distinguer réentraînement P de perturbation F4.

## Slide 7 : A — Comprendre TF-IDF et les caractères
- Rôle : architecture
- Mots et fragments de caractères → vecteur TF-IDF
- Régression logistique → trois sentiments
- Choix sur validation : caractères, F1 0,764076
- F1 test du représentant : 0,553509
- Visuel : Une phrase illustratrice devient une liste de fragments, puis une matrice de poids, puis trois sorties sans probabilités inventées. En dessous deux nombres validation/test clairement légendés. Petits détails : mots 1–2 ; caractères 2–5 ; C choisi sur validation.

## Slide 8 : B1 — Reformuler le sentiment comme une hypothèse
- Rôle : architecture
- Tweet = prémisse
- Une hypothèse pour chaque sentiment
- Trois formulations : arabe, français, anglais
- F1 validation ajusté : 0,501026 · 0,497245 · 0,463444
- Visuel : Schéma pédagogique à gauche : un tweet illustratif rejoint trois hypothèses puis un bloc NLI qui score l’implication. À droite tableau de trois lignes arabe/français/anglais avec les F1 validation exacts. En bas : « Langue et formulation varient ensemble ; leur effet n’est pas isolé ». Pas de probabilité inventée.

## Slide 9 : B2 — Le transfert fixe dépend de l’écriture
- Rôle : comparison
- CAMeLBERT sans entraînement local sur ary
- F1 test arabe : 0,558
- F1 test latin : 0,144
- Supervision externe : contamination non exclue
- Visuel : Deux grandes barres arabe/latin, même axe F1 de 0 à 1 ; chiffres présentés arrondis à trois décimales. Effectifs : arabe 1 367 ; latin 1 519. Petit schéma modèle sentiment préentraîné → tweets marocains. Ne pas prétendre que le modèle n’a jamais été supervisé.

## Slide 10 : Déséquilibre : agir sur l’apprentissage et la décision
- Rôle : concept
- Référence C : BiLSTM + AraVec
- Validation : référence 0,753140
- Pondération : 0,743939 · suréchantillonnage : 0,743941
- Décision : argmax(log p + b)
- Visuel : À gauche deux interventions sur train : pondérer la perte et rééchantillonner, avec pictogrammes précis. À droite trois points ou petites barres validation sur même axe, valeurs six décimales. Bandeau final : « Les décalages b sont choisis sur validation, jamais sur test ». Comparaison sur trois graines.

## Slide 11 : C — Des embeddings à la décision
- Rôle : architecture
- Hasard, AraVec ou fastText
- BiLSTM : lecture dans les deux directions
- Attention : pondérer les représentations
- Choix C : BiLSTM + AraVec, sans attention
- Visuel : Grande architecture illustrée : tokens → embeddings → deux chaînes LSTM en sens opposés → représentation → trois classes. Dessiner un embranchement en pointillés pour attention optionnelle. Ne pas inventer de poids d’attention ni réutiliser les anciens exemples numériques. Légende « Attention descriptive ≠ preuve causale ». Diagramme clair, très pédagogique.

## Slide 12 : C — Les courbes complètent le tableau des scores
- Rôle : data evidence
- BiLSTM + AraVec retenu sur validation
- Validation moyenne : 0,753140
- F1 test, graine 42 : 0,493340
- Les courbes décrivent les époques observées
- Visuel : Courbes originales fournies très grandes au centre, occupent 70 % de la page ; conserver axes, traces et légende. À droite ou en bas une courte colonne des deux scores et « Époque retenue sur validation ». Pas de tendance causale affirmée à partir des courbes.
- Figure stricte : ![curves_C](D:\SetupGame\Desktop\TP LLM\DARIJA SENTIEMENTS\Darija-Sentiment\site\dist\figures\curves_C.png)

## Slide 13 : D et E — Geler ou ajuster l’encodeur
- Rôle : architecture
- D : DarijaBERT gelé, seule la tête apprend
- E : les poids de l’encodeur sont aussi ajustés
- D states : F1 test 0,571593
- E Arabizi : F1 test 0,608138
- Visuel : Deux architectures miroirs : D encodeur avec cadenas, tête apprenable ; E encodeur et tête apprenables. Flèches de gradients en terre cuite seulement sur les parties apprenables. Une comparaison complémentaire en bas : E DarijaBERT standard 0,579512 ; E Arabizi 0,608138, représentants graine 42. Aucune causalité isolée entre taille/coût et qualité.

## Slide 14 : E2 — Une tâche auxiliaire ne garantit pas un gain
- Rôle : architecture
- Perte : L_ary + λ L_arq
- λ = 0 : F1 test 0,608138
- λ = 0,1 : 0,605137 · λ = 0,3 : 0,604073
- Plan réduit : une graine, trois époques maximum
- Visuel : Illustration d’un encodeur partagé à deux têtes ary/arq ; tableau trois lignes des valeurs exactes à droite. En bas : « λ = 0,7 non évalué. Budgets et nombres de graines différents : comparaison exploratoire ». Le corpus principal reste ary ; arq intervient seulement en auxiliaire E2.

## Slide 15 : La sélection se fait avant la comparaison finale
- Rôle : data evidence
- Critère : F1 validation de chaque variante
- E Arabizi : moyenne 0,811256
- Représentant de chaque famille : graine 42
- 33 variantes · 71 évaluations finales
- Visuel : Classement validation des six familles : E 0,811256 ; A 0,764076 ; C 0,753140 ; D 0,746022 ; P 0,735605 ; B 0,602343. Axe commun 0 à 1. Grand cadenas et texte « Candidat fixé : E Arabizi ». Préciser « Moyenne des graines disponibles ; toutes les familles n’ont pas trois graines ». Ne pas choisir après lecture du test.

## Slide 16 : Résultats sur le test officiel
- Rôle : data evidence
- Weighted-F1 · représentants graine 42
- E Arabizi : 0,608138
- Macro-F1 de E : 0,600620
- Choix fixé sur validation
- Visuel : Reprendre très fidèlement la mise en page de la diapositive 16 de la soutenance d’origine : graphique à barres horizontales à gauche, grand résultat typographique à droite et filets fins. SIX barres A TF-IDF caractères 0,553509 ; B CAMeLBERT 0,420555 ; C BiLSTM + AraVec 0,493340 ; D DarijaBERT gelé 0,571593 ; E DarijaBERT-Arabizi 0,608138 ; P TF-IDF sans signes 0,567396. Axe 0 à 1, longueurs proportionnelles. E vert forêt ; autres sauge. À droite grand « 0,608138 », « Candidat choisi sur validation », « Macro-F1 : 0,600620 ». Bas : « AfriSenti ary · n = 2 961 · campagne exploratoire ». SUPPRIMER toute ancienne mention E non exécuté ou A candidat.

## Slide 17 : Lire les erreurs du candidat, classe par classe
- Rôle : data evidence
- Candidat : E DarijaBERT-Arabizi
- La diagonale indique les prédictions correctes
- Les cases hors diagonale montrent les confusions
- Un score global ne suffit pas
- Visuel : Matrice originale du nouveau candidat très grande, totalement inchangée. À droite trois annotations génériques expliquant lignes, colonnes et diagonale sans inventer de nombre de cellules. Ne pas réutiliser la matrice de l’ancienne campagne A.
- Figure stricte : ![modele_final](D:\SetupGame\Desktop\TP LLM\DARIJA SENTIEMENTS\Darija-Sentiment\site\dist\figures\modele_final.png)

## Slide 18 : Validation → test : une baisse à expliquer
- Rôle : comparison
- E Arabizi : 0,811256 → 0,608138
- Les six familles perdent en généralisation
- Le maximum d’une grille test reste un oracle
- Aucun réglage test ne remplace le verrou
- Visuel : Grand graphique dumbbell de six lignes : validation vers test. E 0,811256→0,608138 ; A 0,764076→0,553509 ; C 0,753140→0,493340 ; D 0,746022→0,571593 ; P 0,735605→0,567396 ; B 0,602343→0,420555. Cercles ouverts validation, pleins test, légende exacte, axe0à1. Note « Validation : moyenne disponible ; test : représentant 42 ». À droite petit schéma d’oracle barré, jamais de score optimisé inventé.

## Slide 19 : Écriture non vue : le transfert reste difficile
- Rôle : comparison
- TF-IDF union, même architecture
- Train arabe → test latin : F1 0,2254
- Train latin → test arabe : F1 0,2110
- Taille et couverture du train changent ensemble
- Visuel : Deux rangées horizontales séparées, SANS croisement. Rangée haute verte : « Train arabe » à gauche → flèche portant « F1 test : 0,2254 » → « Test latin · n = 1 519 » à droite. Rangée basse terre cuite : « Train latin » à gauche → flèche portant « F1 test : 0,2110 » → « Test arabe · n = 1 367 » à droite. Les labels doivent être placés directement sur leur propre flèche et non sur l’autre rangée. En bas : « Ce diagnostic ne mesure pas un effet causal pur de l’alphabet ». Interdire les anciennes valeurs 0,2098/0,1932.

## Slide 20 : Incertitude : situer le score, pas seulement le classer
- Rôle : data evidence
- E : IC bootstrap 95 % [0,590655 ; 0,624099]
- 1 000 rééchantillonnages conditionnels au checkpoint
- Graines et bootstrap : deux variabilités distinctes
- McNemar E / E2 λ=0,3 : p = 1,0
- Visuel : Un tableau numérique exact remplace le forest plot pour éviter toute approximation géométrique : colonnes Méthode / F1 test / IC95%. Six lignes : A0,553509 [0,535264 ;0,571928] ; B0,420555 [0,401828 ;0,439308] ; C0,493340 [0,475292 ;0,511506] ; D0,571593 [0,552955 ;0,589547] ; E0,608138 [0,590655 ;0,624099] ; P0,567396 [0,548377 ;0,585336]. À droite grand IC de E et McNemar p=1,0. Le test compare les erreurs appariées, pas directement le F1. Pas de graphique ni figure source sur cette page. Style de tableau éditorial de la soutenance originale.

## Slide 21 : G — Qualité, latence CPU et taille des poids
- Rôle : data evidence
- Mettre la qualité en regard du coût
- Latence : inférence CPU unitaire, pas débit GPU
- Poids sur disque ≠ mémoire vive
- Temps cumulés : sans doubler raw/tuned ou reprises
- Visuel : Insérer le graphique G original comme preuve centrale très grande et inchangée. Autour seulement trois annotations courtes pour interpréter les axes et le coût. Ne jamais recopier les anciens coûts ni inventer les colonnes masquées dans l’affichage. Aucun nouveau chiffre de coût.
- Figure stricte : ![G_quality_cost](D:\SetupGame\Desktop\TP LLM\DARIJA SENTIEMENTS\Darija-Sentiment\site\dist\figures\G_quality_cost.png)

## Slide 22 : Trente erreurs : comprendre avant de compter
- Rôle : concept
- 30 exemples à relire
- 0 / 30 validations humaines enregistrées
- Ironie, graphie, ambiguïté : hypothèses provisoires
- Un désaccord ne prouve pas une erreur de label
- Visuel : Illustration éditoriale de loupe et annotations sur trois fragments schématiques sans faux tweets réels. À droite processus « erreur observée → hypothèse → revue humaine → conclusion ». Ne pas reprendre les anciens comptes 8/8/7/7 comme résultats validés. Pas de taux de bruit inventé.

## Slide 23 : Du benchmark au laboratoire interactif
- Rôle : process
- Site HTML · CSS · JavaScript
- Une phrase → six familles affichées
- Cinq modèles testés réellement via le moteur local
- E : checkpoint absent localement, aucune simulation
- Visuel : Schéma fonctionnel élégant : zone de phrase illustrative → serveur local Python → six branches A/B/C/D/E/P. Cinq branches actives, E en contour pointillé avec étiquette « poids à restaurer ». Pas de capture fabriquée présentée comme réelle, pas de prédictions chiffrées inventées. URL lisible « omarelkhali.github.io/DARIJA-MACHA3IR ». Le site statique a besoin du moteur local pour prédire.

## Slide 24 : Ce que nous retenons du projet
- Rôle : summary
- Candidat mesuré : DarijaBERT-Arabizi
- F1 test : 0,608138 · Macro-F1 : 0,600620
- Les écarts par écriture restent décisifs
- Prochaine étape : de nouvelles annotations indépendantes
- Visuel : Composition finale très forte : un grand chiffre 0,608138 à gauche, puis trois enseignements visuels à droite : protocole, généralisation, coût. Bas : « Test historique déjà consulté · revue humaine inachevée · checkpoint E à restaurer ». Conclusion conditionnelle, aucune promesse de fiabilité universelle.

## Slide 25 : Annexe — Lire les courbes de fine-tuning
- Rôle : data evidence
- E : apprentissage et validation par époque
- Sélection de l’époque sur validation
- Les courbes ne prouvent pas une explication causale
- Aucune nouvelle exécution pour cette présentation
- Visuel : Courbes E fournies très grandes et conservées sans modification. Deux petites annotations hors figure : apprentissage / validation. Conserver les courbes de tous les runs présents, ne pas fabriquer de valeurs ou lisser les traces.
- Figure stricte : ![curves_E](D:\SetupGame\Desktop\TP LLM\DARIJA SENTIEMENTS\Darija-Sentiment\site\dist\figures\curves_E.png)

## Slide 26 : Références et guide pour les questions
- Rôle : reference
- AfriSenti · Muhammad et al., 2023
- DarijaBERT · Gaanoun et al.
- fastText · Bojanowski et al., 2017
- Code, notebooks et résultats : DARIJA-MACHA3IR
- Visuel : Bibliographie courte à gauche, index thématique Q1–Q12 à droite : Q1/Q2 données et protocole ; Q3 nettoyage/P ; Q4/Q5 NLI/transfert ; Q6 déséquilibre ; Q7 réseaux ; Q8 D/E ; Q9 multi-tâche ; Q10 diagnostics test ; Q11 écritures ; Q12 erreurs/coûts. Petit mot arabe مشاعر et branche de sauge de la référence. Les références ne sont pas les mesures de notre campagne.
