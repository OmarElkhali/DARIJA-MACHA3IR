## Slide 1: Analyse de sentiment en darija marocaine

Notre projet cherche à reconnaître le sentiment exprimé dans des tweets en darija marocaine : négatif, neutre ou positif. La difficulté ne se limite pas au choix du réseau. Une même langue s’écrit avec des alphabets différents, des chiffres et des mélanges de langues.

Nous avons donc comparé six familles sur les partitions officielles d’AfriSenti. Je vais d’abord expliquer ce qui rend les données difficiles, puis montrer comment nous avons choisi les modèles et ce que les résultats permettent de conclure. Les performances présentées viennent des sorties sauvegardées de la campagne.

## Slide 2: Une langue, plusieurs graphies

Ces trois phrases illustrent des écritures différentes. Elles ne sont ni des traductions exactes ni des exemples dont nous revendiquons une prédiction mesurée. Regardons les chiffres Arabizi et la négation : les supprimer peut retirer une partie du sens.

Le troisième exemple mélange arabe et français et contient deux appréciations opposées. Le modèle doit interpréter cette combinaison. Détecter l’alphabet ne suffit donc pas à comprendre le sentiment. Cela nous conduit à la première exigence du travail : une évaluation qui distingue les écritures et qui évite les raccourcis de corpus.

## Slide 3: Train apprend, validation choisit, test mesure

Les rôles des partitions sont séparés. Les 5 583 tweets de train servent à apprendre les transformations et les poids. Les 494 tweets de validation servent à choisir la variante, l’époque et les décalages de décision. Nous figeons ensuite ces choix avant de mesurer sur les 2 961 tweets de test.

Le représentant annoncé pour chaque famille utilise la graine 42 ; quand plusieurs graines existent, la sélection de variante utilise leur moyenne de validation. Il faut toutefois conserver une limite historique : le test avait déjà été consulté dans le travail antérieur. Cette campagne reste donc exploratoire. Une confirmation indépendante demanderait de nouvelles données annotées et réservées.

## Slide 4: Lire les distributions avant les scores

Avant de discuter des architectures, nous regardons les classes. La classe neutre est la plus fréquente sur train et validation, tandis que la distribution test diffère. Le graphique fourni montre les proportions enregistrées de validation et de test, sans transformer ce constat en preuve d’un mécanisme causal.

Un classifieur qui prédit toujours neutre obtient seulement 0,160795 de F1 pondéré sur test. Cette référence montre pourquoi une distribution de classes ne constitue pas à elle seule une solution. Le F1 pondéré combine précision et rappel par classe en tenant compte des effectifs ; ce n’est pas l’accuracy. Le Macro-F1 traite les trois classes à poids égaux.

## Slide 5: L’écriture peut devenir un raccourci

La relation entre sentiment et écriture change. Parmi les tweets latins, la proportion de neutres passe de 52,3 % sur train à 29,9 % sur test. Pour l’arabe, elle passe au contraire de 29,2 % à 36,1 %. Le groupe mixte change lui aussi.

Un modèle peut exploiter une corrélation entre l’écriture et le label pendant l’apprentissage. Lorsque cette corrélation varie, ce raccourci devient fragile. Les chiffres sont descriptifs : ils ne prouvent pas que l’alphabet cause un sentiment ou qu’il explique toute la baisse. Nous allons garder cette question en tête pendant la lecture des résultats.

## Slide 6: A — Comprendre TF-IDF et les caractères

TF-IDF représente un tweet par l’importance de ses mots ou fragments. Les mots utilisent des n-grammes de longueur un à deux et les caractères de deux à cinq. Des variantes d’un même mot peuvent ainsi partager certains fragments même si leur orthographe diffère. La régression logistique transforme ce vecteur en scores pour les trois classes.

Dans la famille A, les caractères sont choisis sur validation avec 0,764076, puis obtiennent 0,553509 sur test. Une autre variante peut avoir un meilleur score test, mais elle ne remplace pas après coup le candidat fixé sur validation. C’est cette règle qui rend la comparaison interprétable.

## Slide 7: A — Résultats et temps d’entraînement

Les méthodes linéaires donnent une référence rapide. Pour la graine 42, les temps sauvegardés sont 1,79 seconde pour les mots, 8,39 pour les caractères et 9,06 pour leur union. Le schéma des vecteurs est illustratif : il ne présente pas des poids réellement extraits.

Le représentant caractères, choisi sur validation, obtient 0,553509 de F1 pondéré test. Les huit runs A cumulent 49,15 secondes. Ces durées décrivent les exécutions enregistrées ; elles ne constituent pas un benchmark matériel normalisé.

## Slide 8: B1 — Reformuler le sentiment comme une hypothèse

Le NLI transforme la classification en question d’implication. Le tweet est la prémisse. Nous lui associons successivement une hypothèse de sentiment négatif, neutre et positif. Le score d’implication permet de comparer ces trois hypothèses.

Nous essayons trois formulations linguistiques. Après ajustement sur validation, les F1 sont 0,501026 en arabe, 0,497245 en français et 0,463444 en anglais. Cette différence ne mesure pas un effet pur de langue : nous changeons aussi les mots de l’hypothèse. Les poids sont fixes sur notre corpus, mais le modèle a déjà reçu une supervision externe. Les décalages de décision, eux, sont appris sur validation.

## Slide 9: B — Transfert fixe et coût local

Le zéro seconde correspond à l’absence d’entraînement local sur ary. CAMeLBERT est un modèle arabe déjà entraîné et supervisé ailleurs ; son téléchargement et son inférence ont un coût. Le schéma ne doit donc pas être lu comme la preuve d’une supervision nulle.

Son F1 pondéré global passe de 0,602343 en validation à 0,420555 en test. Les résultats test sont environ 0,558 pour l’écriture arabe et 0,144 pour l’écriture latine. Ce sont deux sous-groupes de test, pas deux partitions d’entraînement. La couverture de l’Arabizi est une explication plausible, sans effet causal isolé ici.

## Slide 10: Déséquilibre : agir sur l’apprentissage et la décision

Nous distinguons deux endroits où agir. Sur train, une perte pondérée augmente l’importance de certaines classes, tandis que le suréchantillonnage change leur fréquence dans les lots. Le découpage des partitions précède ces interventions.

Sur la référence BiLSTM avec AraVec, la moyenne de validation est 0,753140 ; elle est d’environ 0,74394 pour les deux interventions. Elles n’améliorent donc pas automatiquement le critère de sélection. À la décision, une autre intervention consiste à ajouter des décalages aux logarithmes des probabilités. Ces décalages sont choisis sur validation puis figés. Ils ne constituent pas une calibration garantie des probabilités.

## Slide 11: C — Des embeddings à la décision

Suivons le schéma de gauche à droite. Les tokens deviennent des vecteurs denses, initialisés au hasard, avec AraVec ou avec fastText. Le BiLSTM parcourt la séquence dans les deux directions, ce qui lui permet de réunir du contexte à gauche et à droite de chaque position.

Une couche d’attention peut pondérer les représentations avant la classification. Elle fait partie des variantes, mais la variante C retenue est le BiLSTM avec AraVec sans cette couche d’attention. Visualiser un poids d’attention explique une opération du modèle ; cela ne démontre pas qu’un mot cause sa décision. Les inconnus et le padding doivent également être traités correctement.

## Slide 12: C — Apprentissage, résultats et durée

La perte d’apprentissage diminue, mais le F1 de validation fluctue et ne suit pas toujours cette amélioration. Les courbes proviennent des traces sauvegardées et montrent les époques réellement observées. Elles précèdent les décalages de décision ; leur sommet n’est donc pas nécessairement le score final ajusté.

Le run BiLSTM avec AraVec dure 43,43 secondes et comporte six époques observées. La validation moyenne de la variante sélectionnée vaut 0,753140 et le test du représentant de graine 42 vaut 0,493340. Les 45 runs de la famille C cumulent environ 20 minutes 08 secondes. La page a été adaptée par édition d’image ; des variations de rasterisation subsistent, sans changement visuel identifié des traces.

## Slide 13: D — DarijaBERT comme encodeur gelé

Dans D, les poids de DarijaBERT restent gelés. Nous utilisons ses représentations comme des caractéristiques et entraînons une tête plus légère pour le sentiment. Le cadenas concerne donc l’encodeur, pas toute la chaîne.

Les trois variantes extraient le token CLS, la moyenne des états ou la séquence complète des états. Ce sont des alternatives expérimentales. La tête de la variante states comporte 435 715 paramètres entraînables. Chaque alternative produit une classification parmi négatif, neutre et positif.

## Slide 14: D — Résultats et durée de calcul

Le temps de D dépend de l’extraction des caractéristiques et de leur réutilisation. Pour la graine 42, CLS dure environ 1 minute 34, la moyenne 1 minute 11 et states 1 minute 59. Dans ce dernier total, 64,18 secondes d’extraction train et validation sont déjà comprises : il ne faut pas les ajouter une seconde fois.

Le représentant states obtient 0,571593 sur test. Les cinq runs D totalisent environ 6 minutes 43. Le cache et le périmètre des chronométrages diffèrent entre exécutions : ces nombres décrivent le budget observé, sans prouver un rapport de vitesse indépendant du matériel.

## Slide 15: E — Ajuster DarijaBERT sur le sentiment

Dans E, nous ajustons l’ensemble de DarijaBERT-Arabizi, encodeur et tête, soit environ 170 millions de paramètres. Le gradient traverse donc les couches du modèle. Cette adaptation est plus coûteuse que l’apprentissage d’une tête sur un encodeur gelé.

La configuration utilise AdamW, un taux de 2 fois 10 puissance moins 5, un micro-batch de 2 et une accumulation donnant un batch effectif de 32. La longueur maximale est de 128 tokens et le budget maximal de cinq époques. La sélection reste basée sur la validation.

## Slide 16: E — Le fine-tuning concentre le temps

Le fine-tuning concentre une grande part du budget. Pour Arabizi, les graines 42 et 43 prennent respectivement 36 minutes 14 et 37 minutes 02. La graine 44 totalise 26 minutes 32 avec reprise ; sa durée a été reconstruite à partir des segments enregistrés avant et après interruption.

Les six runs E1, standard et Arabizi réunis, cumulent 2 heures 51 minutes 10. Le représentant Arabizi choisi sur validation obtient 0,608138 sur test. Une reprise ne signifie pas un nouvel entraînement indépendant ; elle prolonge le même run.

## Slide 17: E2 — Une tâche auxiliaire ne garantit pas un gain

L’idée du multi-tâche est de partager l’encodeur entre la tâche principale marocaine ary et une tâche auxiliaire algérienne arq. Le coefficient lambda règle le poids de la perte auxiliaire. Un partage utile peut aider, mais un transfert négatif est aussi possible.

Ici, le plan réduit mesure lambda égal à 0,1 et 0,3, avec une graine et trois époques maximum. Leurs F1 test sont 0,605137 et 0,604073. E Arabizi sans auxiliaire atteint 0,608138 pour le représentant 42. Nous ne revendiquons pas de gain de l’auxiliaire dans ce plan. Les budgets et nombres de graines diffèrent, et lambda égal à 0,7 n’a pas été évalué.

## Slide 18: E2 — Budget réduit et suite souhaitée

L’expérience E2 ajoute une tâche auxiliaire, avec deux poids testés dans le budget disponible. Pour lambda 0,1, trois époques prennent 23 minutes 08 et donnent 0,605137 de F1 test. Pour lambda 0,3, trois époques prennent 24 minutes 19 et donnent 0,604073. Ces deux essais utilisent seulement la graine 42.

Je souhaite prolonger E2 avec un budget de cinq à dix époques maximum, un arrêt anticipé sur validation et plusieurs graines. Lambda 0,7 reste à tester. Ce sont des expériences futures : je ne présente aucun gain attendu comme un résultat acquis.

## Slide 19: Nettoyer sans effacer les indices : expérience P

Le nettoyage de référence conserve les indices potentiellement utiles : négation, chiffres Arabizi, répétitions et emojis. L’exemple affiché sert uniquement à montrer une transformation, sans prédiction associée.

Dans P, nous réapprenons le TF-IDF union après suppression conjointe des emojis et de la ponctuation. Son représentant obtient 0,567396 sur test, contre 0,539908 pour l’union de référence. C’est une comparaison descriptive de pipelines ; elle ne justifie pas de supprimer tous les indices pour toutes les familles. Le run P DarijaBERT est exclu du test faute de poids. Enfin, P modifie le pipeline d’apprentissage, tandis que F4 perturbe les textes d’évaluation d’un modèle déjà appris.

Pour F4, retirer les deux types de signes modifie 1 246 tweets. Le modèle E change de décision sur 19,26 % de ces textes et son F1 passe de 0,608138 à 0,589451, avec les labels d’origine. Une relecture doit confirmer que la transformation conserve réellement le sentiment avant de conclure à un défaut de robustesse.

## Slide 20: P — Temps mesurés et étude incomplète

P étudie un changement du pipeline d’apprentissage : on supprime conjointement les emojis et la ponctuation puis on réentraîne. Le TF-IDF union sans signes prend 14,98 secondes et obtient 0,567396 de F1 test.

DarijaBERT sans signes prend environ 20 minutes 13 et obtient 0,761560 en validation. Il est exclu du test faute de poids utilisables : il serait incorrect de lui attribuer un score test. L’étude ne couvre donc pas encore tous les modèles. Elle se distingue de F4, où l’on perturbe seulement les entrées d’évaluation d’un modèle déjà appris.

## Slide 21: La sélection se fait avant la comparaison finale

Le tableau de sélection ne se lit pas comme le classement test. Nous comparons d’abord le F1 de validation des variantes et utilisons la moyenne des graines quand elles sont disponibles. Toutes les familles ne possèdent pas trois graines, ce qui doit rester visible dans l’interprétation.

E Arabizi est le candidat retenu, avec 0,811256 de moyenne de validation. Les six représentants utilisent la graine 42 pour la comparaison annoncée. La campagne sauvegardée comporte 33 variantes et 71 évaluations finales après l’exclusion de P DarijaBERT. Le verrou protège la règle de décision ; il ne transforme pas un test déjà consulté historiquement en nouvelle évaluation indépendante.

## Slide 22: Résultats sur le test officiel

Voici le résultat principal. Chaque barre correspond au représentant 42 choisi selon la validation dans sa famille. E DarijaBERT-Arabizi atteint 0,608138 de F1 pondéré et 0,600620 de Macro-F1. D et P se situent ensuite autour de 0,57 ; A atteint environ 0,55, C environ 0,49 et B environ 0,42.

Le modèle E n’est pas choisi parce que nous avons regardé ce graphique : il était déjà retenu sur validation. Nous ne confondons pas non plus le score d’un représentant et la moyenne test de plusieurs graines. Le graphique situe les performances mesurées, mais un classement seul ne renseigne ni sur l’incertitude, ni sur les erreurs, ni sur le coût.

## Slide 23: Nos résultats face aux publications sur AfriSenti–ary

Je compare ici des publications qui utilisent AfriSenti pour la darija marocaine dans la tâche A monolingue. Les valeurs sont le F1 pondéré sur test, exprimé en pourcentage. NLNDE publie 64,82 dans son tableau 2, ABCD 61,54 dans son tableau 1 et UM6P 60,15 dans son tableau 4. Notre représentant E, de graine 42, donne 60,8138 %, arrondi à 60,81 %.

Cette comparaison situe le travail sans constituer un classement direct. Notre validation contient 494 exemples, alors que le bilan de la compétition en annonce 1 216. La préparation des données, la sélection et l’agrégation des graines diffèrent ; certaines équipes utilisent des ensembles. Le test de notre campagne avait historiquement été consulté. Aucun test de significativité ni équivalence exacte de révision du corpus n’est établi.

La valeur NLNDE de son propre papier est 64,82 ; le bilan officiel indique 64,83, soit 0,01 point d’écart. Nous conservons la valeur du papier cité. Le tableau de baselines de l’article EMNLP AfriSenti utilise l’accuracy : ses nombres ne sont pas mélangés à ce F1 pondéré.

Sources : https://aclanthology.org/2023.semeval-1.68.pdf (tableau 2, colonne ma) ; https://aclanthology.org/2023.semeval-1.44.pdf (tableau 1, Track 7) ; https://aclanthology.org/2023.semeval-1.138.pdf (tableau 4, colonne ma) ; https://aclanthology.org/2023.semeval-1.315.pdf (bilan, tableaux 1 et 3).

## Slide 24: Lire les erreurs du candidat, classe par classe

La matrice appartient au nouveau candidat E Arabizi. Les lignes indiquent les labels réels et les colonnes les prédictions. La diagonale donne les décisions correctes ; les autres cellules montrent où les classes se confondent.

Nous comptons 440 négatifs, 652 neutres et 702 positifs correctement classés. Les négatifs sont notamment souvent prédits neutres : cette cellule contient 299 tweets. Cela donne une direction pour la revue qualitative, sans expliquer automatiquement pourquoi le modèle se trompe. Le total correspond bien aux 2 961 tweets du test. Une moyenne globale peut masquer ces différences de comportement par classe.

## Slide 25: Validation → test : une baisse à expliquer

Les cercles ouverts représentent la validation et les cercles pleins le test. Pour chaque famille, le score baisse entre ces deux lectures. Pour E, nous passons de la moyenne de validation 0,811256 au représentant test 0,608138. Comme il s’agit d’une moyenne d’un côté et d’un représentant de l’autre, ce n’est pas une décomposition causale de la baisse.

Le changement des distributions et des écritures fournit des pistes, mais ne constitue pas une preuve d’un pur changement de prior. Le balayage d’une grille sur test reste un diagnostic rétrospectif optimiste. Son meilleur point ne remplace pas les décalages verrouillés et ne peut pas servir à annoncer une performance indépendante.

## Slide 26: Écriture non vue : le transfert reste difficile

Nous entraînons ici la même architecture TF-IDF union sur une écriture, puis la testons sur l’autre. Le transfert arabe vers latin donne 0,2254 de F1 ; le transfert latin vers arabe donne 0,2110. Nous utilisons les populations test correspondantes, soit 1 519 tweets latins et 1 367 arabes.

Ces niveaux soulignent la difficulté de couvrir une écriture absente de l’apprentissage. Il faut cependant regarder le témoin avec prudence : la taille du train et sa couverture changent ensemble. L’expérience ne mesure donc pas un effet pur de l’alphabet. La conclusion opérationnelle est de valider la couverture des écritures présentes dans le domaine cible.

## Slide 27: Incertitude : situer le score, pas seulement le classer

Un point n’est pas toute l’information. Les intervalles proviennent de 1 000 rééchantillonnages bootstrap du test, conditionnels au checkpoint évalué. Pour E, l’intervalle à 95 % est de 0,590655 à 0,624099. Il ne représente pas toutes les sources d’incertitude possibles.

La variabilité entre graines répond à une autre question : que se passe-t-il si l’apprentissage est répété ? Enfin, le test exact de McNemar entre E et E2 lambda 0,3 compare les erreurs appariées : 143 tweets sont corrects uniquement pour E et 144 uniquement pour E2, avec p égal à 1. Ce test ne porte pas directement sur la différence de F1. Le simple ordre des points ne suffit pas à annoncer une supériorité statistique.

## Slide 28: G — Qualité, latence CPU et taille des poids

La qualité doit être mise en regard du coût. Le graphique original relie le F1 test à trois mesures : la latence CPU par tweet, le poids des fichiers et le temps d’apprentissage enregistré. Les axes horizontaux sont logarithmiques ; il faut donc éviter de lire les distances comme sur une échelle linéaire.

La latence est une mesure unitaire dans l’environnement de la campagne, pas un débit GPU ni une promesse de service. Les poids sur disque ne sont pas la mémoire vive. Les colonnes numériques tronquées n’ont pas été reconstituées pour cette présentation. Enfin, les temps cumulés évitent les doubles comptes raw/tuned et reprises ; les époques perdues avant checkpoint et les coupures réseau ne sont pas mesurables.

## Slide 29: Temps enregistrés par famille

Les journaux recensent 72 runs distincts : 8 pour A, 4 transferts fixes B, 45 pour C, 5 pour D, 8 pour E et 2 pour P. Ce comptage n’est pas celui des 71 évaluations finales. Les quatre B ne réalisent aucun entraînement local.

Le total enregistré est 4 heures 26 minutes 46 secondes. E représente 3 heures 38 minutes 37, en réunissant E1 et E2. Les téléchargements, les coupures et les calculs perdus ne sont pas quantifiés. Ces durées de campagne, avec caches et matériels variables, ne sont pas une comparaison de vitesse standardisée.

## Slide 30: Quatre changements de session Colab

J’ai dû changer de session Colab quatre fois, car les sessions se terminaient alors que certains entraînements n’étaient pas achevés. Quatre changements ne signifie pas quatre sessions au total. Les conséquences étaient concrètes : relancer l’environnement, restaurer les données et reprendre les calculs sauvegardés.

La reprise d’E Arabizi avec la graine 44 est tracée dans les résultats. Le temps perdu lors des interruptions n’est pas mesuré ; il ne figure donc pas dans le total précédent. Ces contraintes expliquent la réduction de certaines expériences, notamment E2 et les ablations sur tous les modèles.

## Slide 31: Améliorations souhaitées

Je souhaite comparer quatre conditions : texte intact, sans emojis, sans ponctuation et sans les deux. Pour A, C, D, E et E2, cela implique de réentraîner à budget comparable. Pour B, dont les poids restent fixes, on compare les entrées perturbées. Une relecture doit vérifier que les transformations conservent le sentiment.

Pour E2, l’objectif est un budget plus long, plusieurs graines et un arrêt anticipé décidé uniquement sur validation. Les sauvegardes persistantes devront conserver modèle, optimiseur et état aléatoire pour reprendre proprement. Toutes ces améliorations restent à réaliser ; aucun résultat futur n’est inventé.

## Slide 32: Trente erreurs : comprendre avant de compter

Nous avons préparé trente erreurs pour une relecture, mais aucune validation humaine n’est encore enregistrée. Les catégories comme ironie, graphie ou ambiguïté doivent donc être comprises comme des hypothèses de travail.

Le chemin affiché distingue l’observation et sa conclusion : constater un désaccord, proposer une explication, faire relire, puis décider ce que les preuves soutiennent. Un modèle différent du label ne prouve pas une erreur d’annotation. De même, un petit échantillon ciblé ne permet pas d’estimer la fréquence de ces phénomènes dans toutes les erreurs. La revue humaine et un nouvel ensemble indépendant restent nécessaires.

## Slide 33: Du benchmark au laboratoire interactif

Le site présente la méthode, les résultats et un laboratoire de phrases. L’interface est en HTML, CSS et JavaScript ; les véritables modèles sont chargés par un serveur Python local. GitHub Pages héberge la partie statique, pas les calculs PyTorch.

Les six familles sont affichées, mais cinq ont été testées avec de véritables prédictions. Les poids ajustés du modèle E manquent dans les fichiers locaux, et cette absence est signalée. Nous ne remplaçons pas E par une simulation. Une courte démonstration permet de voir les désaccords entre méthodes, mais une phrase réussie ou ratée n’est pas une nouvelle mesure du F1 de benchmark.

## Slide 34: Ce que nous retenons du projet

Notre résultat mesuré principal est DarijaBERT-Arabizi, retenu sur validation, avec 0,608138 de F1 pondéré et 0,600620 de Macro-F1 sur le représentant test. Le travail montre aussi pourquoi un résultat global doit être accompagné d’une lecture par écriture, d’une incertitude et d’un coût.

La prochaine étape n’est pas de continuer à régler le même test. Elle consiste à obtenir de nouvelles annotations indépendantes dans le domaine cible, terminer la revue des erreurs et restaurer le checkpoint E pour la démonstration complète. Nous proposons donc un prototype accompagné de ses limites, sans garantie de fiabilité universelle. Le notebook, le site et le dépôt rendent les preuves accessibles pour poursuivre ce travail.

## Slide 35: Annexe — Lire les courbes de fine-tuning

Cette annexe permet de revenir sur le fine-tuning pendant les questions. À gauche, la perte d’apprentissage baisse ; à droite, les F1 bruts de validation décrivent les époques enregistrées. Les runs multi-tâches du plan réduit s’arrêtent à trois époques maximum, alors que d’autres runs peuvent aller plus loin.

Ces courbes brutes précèdent le réglage des décalages de décision. Elles ne sont donc pas obligées d’atteindre le F1 final ajusté du tableau de sélection. Nous ne les avons ni lissées ni réexécutées pour fabriquer une meilleure trajectoire. Elles restent des traces de la campagne, pas une preuve causale d’un effet du multi-tâche.

## Slide 36: Références et guide pour les questions

Les sources distinguent les données, les modèles et nos propres mesures. AfriSenti fournit le corpus et le cadre de la tâche. Les travaux sur DarijaBERT et fastText décrivent les représentations employées. Leurs résultats publiés ne sont pas les scores de notre campagne.

L’index à droite permet de retrouver les réponses aux questions du projet : protocole, nettoyage, transfert, déséquilibre, réseaux, fine-tuning, auxiliaire, diagnostics, écritures et erreurs. Le dépôt contient les notebooks avec leurs sorties, les sources et les livrables. Les mesures de cette soutenance sont reliées aux sorties sauvegardées, sans nouvel entraînement pour préparer les diapositives.


## Slide 37: Questions de soutenance — Q1 à Q6

Q1 — Quelle métrique pour des classes déséquilibrées ?

Énoncé complet : Distribution des labels dans train, val et test, globalement et par écriture (arabe / latine). Scores du classifieur trivial « classe majoritaire ». Quelle métrique retenez-vous, et pourquoi ?

F1 pondéré pour le benchmark, Macro-F1 pour équilibrer les classes. Majorité seule : 0,160795 sur test. Détails : p. 3–4. Pour les détails, je reviens aux diapositives indiquées.

Q2 — L’écriture crée-t-elle un raccourci ?

Énoncé complet : Raccourcis : la distribution des labels dépend-elle de l'écriture ? Que vaut la règle fondée sur l'écriture ? Qu'en concluez-vous sur la validité de vos scores ?

Neutres latins : 52,3 % sur train, 29,9 % sur test. Cette corrélation change ; elle menace la généralisation. Détails : p. 5, 25–26. Pour les détails, je reviens aux diapositives indiquées.

Q3 — Faut-il supprimer emojis et ponctuation ?

Énoncé complet : Quels choix de prétraitement avez-vous faits par famille de modèles ? Lesquels détruisent des indices utiles (emojis, lettres répétées, mots français, arabizi) ? Chiffres à l'appui (ablation).

Conserver les indices par défaut. P union sans signes : F1 test 0,567396. Ablation sur tous les modèles encore à réaliser. Détails : p. 19–20, 31. Pour les détails, je reviens aux diapositives indiquées.

Q4 — Comment le NLI classe-t-il sans train local ?

Énoncé complet : Zero-shot (B1) : score obtenu et sensibilité à la formulation des hypothèses. Pourquoi un modèle NLI peut-il classer sans exemple d'entraînement ? Quel rôle joue l'attention entre prémisse et hypothèse ?

Comparer l’implication de trois hypothèses. F1 validation : arabe 0,501026 ; français 0,497245 ; anglais 0,463444. Supervision externe préalable. Détails : p. 8. Pour les détails, je reviens aux diapositives indiquées.

Q5 — Pourquoi B2 diffère-t-il selon l’écriture ?

Énoncé complet : Modèle de sentiment B2 : scores sur les tweets arabes et latins séparément. Pourquoi cet écart ? Son score est-il exempt de contamination ?

CAMeLBERT : F1 test arabe 0,558 ; latin 0,144. Couverture de l’Arabizi limitée, cause non isolée ; contamination externe non certifiée. Détails : p. 9. Pour les détails, je reviens aux diapositives indiquées.

Q6 — Comment traiter le déséquilibre ?

Énoncé complet : Déséquilibre : comparez pondération de la perte, sur-échantillonnage et optimisation du seuil (tableau). Pourquoi le sur-échantillonnage doit-il être fait après la séparation train / val ?

Pondération, suréchantillonnage et décalages de décision sur validation. Suréchantillonner uniquement train après séparation, pour éviter les fuites. Détails : p. 10, 21. Pour les détails, je reviens aux diapositives indiquées.



## Slide 38: Questions de soutenance — Q7 à Q12

Q7 — Quel réseau C et que prouve l’attention ?

Énoncé complet : C1 vs C2 vs C3 (et C4) : résultats. Les poids d'attention se concentrent-ils sur les emojis, les mots d'opinion et les mots français ? Effet de l'encodage positionnel ?

BiLSTM + AraVec retenu : F1 test 0,493340. L’attention ne prouve pas une cause ; les courbes décrivent les époques observées. Détails : p. 11–12. Pour les détails, je reviens aux diapositives indiquées.

Q8 — Quel gain et quel coût pour le fine-tuning ?

Énoncé complet : PLM gelé (D) vs fine-tuné (E1) : gain et coût (temps, paramètres entraînés). SI2M-Lab/DarijaBERT vs SI2M-Lab/DarijaBERT-arabizi : lequel, et pourquoi ?

D : 0,571593 ; E Arabizi : 0,608138, soit +3,65 points. Environ 170 M de paramètres ajustés ; run 42 : 36 min 14 s. Détails : p. 13–16, 21–22. Pour les détails, je reviens aux diapositives indiquées.

Q9 — La tâche auxiliaire améliore-t-elle E2 ?

Énoncé complet : Multi-tâche (E2) : l'ajout de l'algérien améliore-t-il la tâche principale ? Effet de λ. Pourquoi une tâche auxiliaire peut-elle aider… ou nuire ?

F1 : λ 0,1 → 0,605137 ; λ 0,3 → 0,604073. Aucun gain établi face à E1 : une graine, trois époques, budgets différents. Détails : p. 17–18, 27. Pour les détails, je reviens aux diapositives indiquées.

Q10 — Comment gérer le décalage validation–test ?

Énoncé complet : Décalage de prior (F1) : comparez la distribution des labels de val et de test. Écart entre le seuil (ou la règle de décision) choisi sur val et l'optimum a posteriori sur le test. Que mettriez-vous en place en production (recalibration, monitoring) ?

Décision figée sur validation. L’optimum trouvé sur test reste un diagnostic. En production : suivi des dérives et recalibration sur de nouvelles données. Détails : p. 3, 21, 25. Pour les détails, je reviens aux diapositives indiquées.

Q11 — Que se passe-t-il sur une écriture non vue ?

Énoncé complet : Écriture non vue (F2) : chute mesurée. Quel modèle déployer pour des commentaires marocains mêlant les deux écritures, et pourquoi ?

TF-IDF : arabe → latin 0,2254 ; latin → arabe 0,2110. Couvrir les deux écritures et valider le domaine cible avant déploiement. Détails : p. 26, 34. Pour les détails, je reviens aux diapositives indiquées.

Q12 — Quel modèle déployer et quelles limites ?

Énoncé complet : Analyse d'erreurs (F6) : catégories dominantes, part du bruit d'annotation. Quel modèle mettriez-vous en production, et pourquoi (qualité, robustesse, coût, latence — chiffres à l'appui) ?

E : meilleure validation, F1 test 0,608138 ; A : option légère à valider. Trente erreurs à relire ; bruit d’annotation non quantifié, checkpoint E à restaurer. Détails : p. 21–22, 28–29, 32–34. Pour les détails, je reviens aux diapositives indiquées.

