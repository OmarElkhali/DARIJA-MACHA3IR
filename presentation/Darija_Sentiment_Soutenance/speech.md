# Darija-Sentiment — notes de présentation

## Slide 1: Analyse de sentiment en darija marocaine

Dans ce projet, j'étudie la reconnaissance de trois sentiments dans des tweets marocains : positif, négatif et neutre. La difficulté est que la darija s'écrit avec plusieurs alphabets et des orthographes variables. Je compare donc des méthodes simples, des réseaux neuronaux et des modèles préentraînés.

Je vais d'abord expliquer les données et le protocole, puis les méthodes, et enfin les résultats et leurs limites. Les réponses aux douze questions sont intégrées dans ce parcours. Le périmètre de l'étude reste exclusivement marocain, avec le corpus ary.

Référence : research/questions_officielles.md ; README.md ; notebook TP2_Darija.ipynb.

## Slide 2: Une langue, plusieurs graphies

Ces exemples illustrent trois écritures possibles. La première utilise des lettres arabes. La deuxième utilise des lettres latines. La troisième mélange des caractères arabes, du français et de l'Arabizi. Ce sont des exemples pédagogiques, pas trois traductions exactes ni des prédictions mesurées.

Une suppression trop agressive des chiffres, des négations ou du français peut modifier le sens. Les emojis et les répétitions peuvent aussi exprimer de l'intensité. Enfin, une écriture latine ne prouve pas qu'un texte est entièrement en Arabizi : elle peut contenir du français ou d'autres langues. Cette diversité motive les comparaisons par écriture que je présenterai ensuite.

Référence : src/preprocess.py ; exemples du laboratoire app.py.

## Slide 3: Le protocole expérimental

Les partitions officielles ont des fonctions différentes. Les 5 583 tweets train servent à apprendre les paramètres et les représentations locales. Les 494 tweets val, appelés dev dans le code, servent à choisir les variantes, les époques et les règles de décision. Les 2 961 tweets test servent à l'évaluation après verrouillage.

Le vocabulaire, les poids TF-IDF et les embeddings fastText locaux sont appris uniquement sur train. Les checkpoints et les règles sélectionnés sont ensuite figés par leurs empreintes. Cela protège la comparaison contre une sélection rétroactive sur test. Les diagnostics a posteriori demandés dans les questions seront signalés séparément et ne modifieront pas ce choix.

Référence : data/manifest.json ; src/preprocess.py ; results/selection_lock.json.

## Slide 4: Distribution des labels et référence majoritaire

Le graphique présente train et val, tandis que le tableau ajoute le test dans l'analyse finale. La classe majoritaire de train est neutre : 2 161 tweets sur 5 583. Le classifieur trivial conserve toujours cette classe, même si la classe la plus fréquente du test est positive. Rechoisir la majorité sur test utiliserait ses réponses.

Son Weighted-F1 vaut 0,2160 sur train, 0,2117 sur val et 0,1608 sur test. Ses accuracy sont respectivement 38,7 %, 38,3 % et 32,7 %. Cette différence rappelle qu'un F1 n'est pas une accuracy.

Pour Q1, je retiens le Weighted-F1 comme critère principal du protocole, accompagné du Macro-F1. Le second donne une importance égale aux trois sentiments. Les effectifs globaux négatif, neutre, positif sont 1 664, 2 161, 1 758 sur train ; 138, 189, 167 sur val ; 850, 967, 1 144 sur test.

Référence : results/f1_conditional_labels.csv ; evidence.draft.json, compléments de majorité calculés sans apprentissage.

## Slide 5: Le raccourci d'écriture

Pour Q1 et Q2, la distribution dépend clairement de l'alphabet. Sur train arabe, les effectifs négatif, neutre, positif sont 1 137, 954 et 1 174 ; sur train latin, 514, 1 190 et 572. Sur val arabe, ils sont 103, 75 et 99 ; sur val latin, 34, 112 et 67. Sur test arabe, ils deviennent 308, 494 et 565 ; sur test latin, 527, 454 et 538. Les autres groupes expliquent la différence avec les totaux globaux.

Une règle très simple apprend la majorité par écriture sur train : positif pour l'arabe, neutre pour le latin. Elle obtient 0,3656 de Weighted-F1 sur val, contre 0,2117 pour la majorité globale. L'alphabet seul fournit donc déjà un raccourci prédictif. Sur test, son score tombe à 0,2945.

Cela ne rend pas tous les scores invalides. Cela impose de vérifier si les modèles généralisent au-delà des corrélations du corpus. Le changement de la proportion neutre sur les textes latins est particulièrement important.

Référence : results/f1_conditional_labels.csv ; scripts/build_notebook.py, baseline script-only ; evidence.draft.json.

## Slide 6: Prétraitement : conserver les indices utiles

Pour Q3, le nettoyage commun remplace les mentions et URLs, mais conserve les négations, chiffres Arabizi, accents, mots français, lettres répétées et emojis. Ensuite, la représentation dépend de la famille. TF-IDF mots ignore les emojis et la ponctuation isolée avec son tokenizer standard. TF-IDF caractères peut en exploiter des motifs. C utilise soit des tokens regex, soit les caractères. Les modèles préentraînés utilisent leur propre tokenizer.

Les limites sont de 64 tokens pour les réseaux C à mots, 256 caractères pour le Character-CNN, 128 sous-tokens pour B et 64 pour D/E. La fin d'un texte trop long peut donc être tronquée.

L'ablation mesurée retire conjointement emojis et ponctuation répétée. A change vers une erreur dans 9 cas sur 102, soit 8,8 %. Les autres familles ont d'autres sous-ensembles : on ne peut pas lire ce tableau comme un classement causal. Les suppressions du français, des chiffres et des répétitions de lettres n'ont pas été mesurées séparément. Je ne leur attribue donc aucun effet chiffré inventé.

Référence : src/preprocess.py ; src/models.py ; src/train.py ; results/f4_A/B/C/D_perturbations.csv ; results/comparison.csv.

## Slide 7: A — TF-IDF et régression logistique

TF-IDF transforme un texte en un vecteur décrivant la présence de fragments. La régression logistique apprend ensuite les associations entre ces fragments et les trois sentiments. Je compare les mots, les caractères, leur combinaison et une référence majoritaire.

Les n-grammes de caractères sont des séquences de deux à cinq caractères. Ils peuvent partager de l'information entre des graphies proches. Le petit dessin illustre le partage de caractères ; les caractéristiques effectivement utilisées sont bien des fragments de longueur deux à cinq.

Sur val, la variante à mots obtient en moyenne 0,7189 et la variante à caractères 0,7683. Ce résultat établit une référence solide avec peu de coût. Il ne prouve pas que les caractères seront supérieurs dans tous les domaines. Les transformations sont ajustées sur train uniquement, et la qualité finale doit être confirmée sur test.

Référence : src/models.py, classical_model ; results/results.csv ; results/figures/A_dev.png.

## Slide 8: B1 — Classer par inférence textuelle

Pour Q4, le tweet devient une prémisse. Je lui associe trois hypothèses exprimant respectivement un sentiment négatif, neutre ou positif. Le modèle a déjà appris ailleurs à reconnaître une relation d'inférence entre deux textes. Il peut donc appliquer cette compétence sans apprentissage local de ses poids sur ary.

Les tokens de la prémisse et de l'hypothèse sont traités conjointement. L'auto-attention permet leurs interactions dans les représentations. Cela aide la comparaison du sens, sans fournir une preuve causale par la simple lecture de poids d'attention.

Sur val, l'argmax donne environ 0,3980 en arabe et français, et 0,3860 en anglais. Les règles ajustées donnent 0,5010, 0,4972 et 0,4634. La deuxième colonne inclut un réglage sur val et ne représente donc pas une décision entièrement sans adaptation. Une seule formulation par langue a été testée. Les NLI ne sont pas les représentants B retenus sur test.

Références : results/results.csv ; src/models.py ; https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 ; https://arxiv.org/abs/1909.00161 ; https://arxiv.org/abs/2006.03654.

## Slide 9: B2 — CAMeLBERT face aux deux écritures

CAMeLBERT est déjà spécialisé en analyse de sentiment sur des corpus externes. Il ne reçoit ici aucun apprentissage local de ses poids. Pour Q5, la différence par écriture est très forte : le Weighted-F1 test est de 0,5581 sur les 1 367 textes arabes et de 0,1442 sur les 1 519 textes latins. Sur les 75 mixtes, il vaut 0,4923, avec un faible effectif à garder en tête.

La spécialisation arabe et la représentation moins adaptée de l'Arabizi constituent des explications plausibles. Cette comparaison ne permet pas d'attribuer toute la différence au tokenizer.

Je ne peux pas garantir l'absence de contamination externe. La fiche officielle indique un entraînement sentiment sur ASTD, ArSAS et SemEval. L'audit local ne recouvre pas tous les textes utilisés lors du préentraînement et des apprentissages externes.

Références : results/B_camelbert_s42_test_tuned.csv ; https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment.

## Slide 10: Déséquilibre et règle de décision

Pour Q6, je garde la même architecture attention et la même initialisation aléatoire. Je compare alors la référence, la pondération de la perte et le suréchantillonnage. La pondération augmente le coût des erreurs sur certaines classes. Le suréchantillonnage tire plus souvent les exemples de classes moins fréquentes.

Sur l'argmax, les deux stratégies améliorent légèrement la référence dans ces runs. Après réglage des décisions sur val, la référence obtient 0,7370, contre 0,7356 et 0,7294. Le gain n'est donc pas uniforme. Les valeurs ajustées sont mesurées sur la partition qui a servi à régler la règle : ce n'est pas une validation indépendante du gain.

Le suréchantillonnage intervient après la séparation train/val et dans train uniquement. Sinon, des copies pourraient passer des deux côtés et gonfler la validation. Ici, le seuil est une règle multiclasse : j'ajoute des décalages aux logarithmes des scores et je choisis le maximum.

Référence : results/results.csv ; src/train.py, fit_tensor ; src/evaluate.py, tune_offsets.

## Slide 11: C — Embeddings, BiLSTM et attention

Un embedding représente un token par un vecteur. Je compare une initialisation aléatoire, AraVec et fastText appris sur train. Le BiLSTM lit la séquence dans les deux directions. L'attention pondère ensuite les états pour produire la représentation de classification.

Pour Q7, les poids visibles viennent d'un véritable exemple dev. Le terme « niveau » reçoit environ 0,181, tandis que « tala3 » reçoit environ 0,451. On observe donc un mot français dans une pondération importante, mais cela ne démontre pas une préférence générale pour le français. Dans d'autres exemples, un nom propre ou une ponctuation domine. Les cinq cas ne suffisent pas à établir une concentration systématique sur les emojis.

Le padding est masqué. Le vocabulaire reste fermé : un token absent devient UNK, même avec l'initialisation fastText. Ces visualisations décrivent le modèle ; elles ne prouvent pas la cause de ses décisions.

Référence : results/attention_C_5_examples.csv ; results/attention_D_5_examples.csv ; src/models.py.

## Slide 12: C1 à C4 : résultats et positions

Ce tableau répond à la comparaison C1, C2, C3 et C4. Toutes les valeurs sont des moyennes Weighted-F1 sur val avec trois graines et la décision ajustée. Avec des vecteurs aléatoires, le BiLSTM obtient 0,7385, légèrement au-dessus de l'attention à 0,7370. Avec fastText, l'attention atteint 0,7476 et devient la variante C retenue. L'attention n'apporte donc pas un gain automatique.

Le Mini-Transformer utilise deux couches, 128 dimensions et quatre têtes. Sans positions, la variante aléatoire obtient 0,7210, contre 0,7068 avec positions. L'effet s'inverse pour fastText et AraVec, où les positions améliorent le score. Je conserve ces trois comparaisons au lieu d'annoncer un effet universel.

Le Character-CNN travaille directement sur les caractères et obtient 0,6983 avec son initialisation aléatoire. Ses cases fastText et AraVec ne sont pas des scores manquants d'une même expérience à mots : ces initialisations de mots ne s'appliquent pas à cette variante.

Référence : results/results.csv ; src/models.py, TextNetwork ; evidence.draft.json, C_dev.

## Slide 13: DarijaBERT gelé ou ajusté

Pour Q8, D et E diffèrent par ce qui apprend. Dans D, DarijaBERT reste gelé et seule la tête apprend le sentiment marocain. Les variantes comparent le premier état CLS, la moyenne des états et la séquence donnée à un BiLSTM-attention. Le représentant retenu atteint 0,5321 sur test avec 435 715 paramètres entraînables.

Le coût D comprend 52,34 secondes pour le run de tête et 486,77 secondes d'extraction partagée train/dev. Son inférence nécessite toujours l'encodeur complet et mesure 88,21 millisecondes par tweet sur cette machine, chargement exclu.

E ajusterait aussi l'encodeur, avec environ 146,9 millions de paramètres pour le modèle standard et sa tête. La campagne complète n'a pas été exécutée. Les pas AdamW sur deux exemples sont des vérifications techniques. Je ne peux donc ni chiffrer un gain D/E, ni déclarer DarijaBERT-Arabizi meilleur. Ce serait une hypothèse à tester équitablement sur les deux écritures.

Référence : results/comparison.csv ; results/execution_status.csv ; results/pretrained_smoke.json ; results/resource_estimate_E.json.

## Slide 14: Multi-tâche et poids λ

Q9 demande l'effet de l'algérien et du poids lambda. La consigne finale de mon projet exclut arq : je n'ai donc ni expérience multi-tâche exécutée ni gain à annoncer. Cette diapositive explique seulement le principe.

Un encodeur pourrait être partagé entre deux tâches, chacune possédant sa tête. La perte combinerait la tâche marocaine et la tâche auxiliaire, pondérée par lambda. Un partage pertinent peut aider à apprendre des représentations plus générales. Des données ou des objectifs mal assortis peuvent aussi provoquer un transfert négatif. Une valeur plus grande de lambda augmente l'influence de l'auxiliaire, sans garantir une amélioration.

Dans une future étude autorisée, il faudrait comparer lambda zéro, 0,3, 0,5 et un, avec un budget comparable et une sélection sur val ary. Ici, cette extension reste hors périmètre.

Référence théorique : Sebastian Ruder, An Overview of Multi-Task Learning in Deep Neural Networks, 2017, https://arxiv.org/html/1706.05098v1.

## Slide 15: Choisir avant de regarder le test

Les variantes sont comparées sur leur Weighted-F1 moyen de validation. Pour A, C et D, une variante doit disposer de trois graines. B utilise des modèles fixes, sans réplications artificielles de leurs paramètres.

Les représentants retenus sont TF-IDF caractères pour A, CAMeLBERT pour B, attention-fastText pour C et états cachés avec BiLSTM-attention pour D. Le meilleur critère dev est celui de A, à 0,7683. La graine 42 a été fixée comme représentant avant de regarder les scores test ; ce n'est pas la meilleure graine choisie après coup.

Ce point relie Q1 et Q12 : la métrique sert à prendre une décision scientifique selon un protocole annoncé. Le tableau test qui suit mesure ce choix ; il ne sert pas à recommencer librement la sélection.

Référence : results/selection_lock.json ; src/evaluate.py, select_on_dev et lock_selection.

## Slide 16: Résultats sur le test officiel

Le test est sensiblement plus difficile que val. Les quatre valeurs représentent les checkpoints de graine 42, avec leur règle choisie sur val. A obtient 0,5365, D 0,5321, C 0,4962 et B 0,4209. E reste sans score complet.

A et D sont proches. Le fait que A ait le plus grand score représentatif ne démontre pas sa supériorité universelle. La moyenne test des trois graines de C vaut par exemple 0,5098, différente de son représentant à 0,4962. Il faut donc toujours préciser ce que l'on compare.

La première conclusion est que la méthode simple reste compétitive dans cette étude. La seconde est la baisse importante entre validation et test. Pour la comprendre, je vais examiner les confusions et les différences entre écritures.

Référence : results/comparison.csv ; résultats principaux verrouillés, sans remplacement par les diagnostics a posteriori.

## Slide 17: Lire les confusions du candidat

Les lignes indiquent les labels officiels et les colonnes les prédictions. La diagonale contient les bonnes décisions. Par exemple, 255 textes négatifs restent négatifs, mais 515 sont prédits neutres. Parmi les positifs, 509 restent positifs et 426 deviennent neutres.

La matrice montre pourquoi une seule moyenne ne suffit pas : le neutre attire beaucoup de décisions. Le Weighted-F1 vaut 0,5365 et le Macro-F1 0,5272. L'accuracy est une autre mesure : elle correspond à la somme de la diagonale divisée par 2 961.

Pour Q12, ces chiffres orientent la lecture qualitative des erreurs. Ils ne disent pas encore si une erreur vient d'une graphie rare, d'une cible ambiguë ou d'un label discutable. Il faut relire les textes et confronter plusieurs sources d'information.

Référence : results/figures/modele_final.png ; results/A_char_s42_test_tuned.csv.

## Slide 18: Décalage de distribution et décision

Pour Q10, les proportions globales val/test sont de 27,94/28,71 % négatif, 38,26/32,66 % neutre et 33,81/38,64 % positif. Le changement est particulièrement fort dans le latin : le neutre passe de 52,6 % à 29,9 %. Cela ne prouve pas les conditions d'un pur décalage de prior avec toutes les autres distributions invariantes.

La règle fixée sur val utilise les décalages zéro, plus 0,3, zéro et obtient 0,5365 sur test. L'argmax brut donne 0,5536. Le meilleur point a posteriori des 25 réglages de la grille d'origine utilise zéro, moins 0,6, zéro et donne 0,5738. Son écart de 3,72 points est un diagnostic optimiste, pas une performance à déployer ni un optimum continu garanti.

En production, je proposerais le suivi des écritures et des prédictions, puis des mesures sur de nouvelles annotations. Une recalibration serait validée sur des données réservées et contrôlée sur une autre évaluation. Ce suivi reste proposé, pas implémenté.

Référence : results/f1_label_shift.csv ; results/f1_conditional_labels.csv ; evidence.draft.json, Q10_posthoc_diagnostic. Le résultat principal et les checkpoints ne changent pas.

## Slide 19: Transférer vers une écriture non vue

Pour Q11, les expériences F2 entraînent TF-IDF union sur une seule écriture, puis évaluent sur l'autre. Sur les textes arabes test, le modèle entraîné en latin obtient 0,1932. Le témoin union de même graine entraîné sur tout train obtient 0,7481. La différence descriptive est de 55,48 points.

Sur les textes latins, le modèle entraîné en arabe obtient 0,2098, contre 0,3461 pour le témoin complet. La différence vaut 13,64 points. Le témoin complet est un checkpoint déjà figé, évalué rétrospectivement pour cette question. Il ne remplace pas le candidat principal.

La comparaison utilise les mêmes populations test et la même architecture. Cependant, la taille de train et la couverture de l'écriture changent ensemble : je n'attribue donc pas l'écart à l'alphabet seul. Pour des commentaires comportant les deux écritures, il faut des données représentatives des deux et une validation sur le domaine réellement visé.

Référence : results/f2_cross_script.csv ; models/A_union_s42 ; evidence.draft.json, Q11_posthoc_control.

## Slide 20: Mesurer l'incertitude

Les intervalles proviennent de 1 000 rééchantillonnages de tweets, conditionnellement au checkpoint représentatif. Celui de A va approximativement de 0,5171 à 0,5544 et celui de D de 0,5129 à 0,5515. Ces intervalles ne mesurent pas la même source de variabilité que les trois graines d'entraînement.

Le test exact de McNemar compare les erreurs appariées de A et D. A réussit seul sur 333 cas, D sur 307. La valeur p est 0,3231 : ce test ne détecte pas une différence de taux d'erreur au niveau 5 %. Cela ne prouve pas une équivalence, et McNemar ne teste pas directement une différence de F1.

Pour Q11 et Q12, ces résultats invitent à rester prudent sur les petites différences. La qualité doit être examinée avec les écritures, les coûts et les conséquences des erreurs dans le domaine cible.

Référence : results/comparison.csv ; results/f5_mcnemar.json.

## Slide 21: Trente erreurs à examiner

Pour Q12, les 30 erreurs comprennent huit cas de bruit d'annotation possible, huit confusions neutre/polarisé, sept ironies ou autres cas à examiner et sept difficultés de graphie Arabizi possibles. Ces catégories sont provisoires. L'échantillonnage a équilibré les catégories : leurs proportions ne représentent pas celles de toutes les erreurs.

Les extraits montrent une négation Arabizi, « mafikomch ti9a », et un éloge français, « tu es magnifique ». Le contexte complet et la cible de l'opinion restent importants. Le candidat utilise les caractères : l'étiquette de difficulté de tokenisation n'est pas une preuve d'échec WordPiece.

Les votes ont été rapprochés pour 18 des 30 cas, dont huit avec désaccord, soit 44,4 % des cas rapprochés. Les 12 autres n'ont pas de votes retrouvés. Aucun de ces constats n'établit le taux de labels faux. La revue humaine reste à faire.

Référence : results/f6_30_erreurs.csv ; results/error_analysis.md ; evidence.draft.json, error_summary.

## Slide 22: Une application pour tester et expliquer

L'application Streamlit sert à explorer le corpus, lire les résultats et tester une phrase avec les modèles réellement disponibles. Elle possède six espaces : vue d'ensemble, laboratoire, corpus, modèles, robustesse et recherche.

Le schéma décrit le fonctionnement ; ce n'est pas une capture de l'interface. L'analyse charge les checkpoints un par un depuis les fichiers locaux, puis libère leur mémoire. Elle affiche les trois scores et la règle qui transforme ces scores en classe. Ces scores ne sont pas des probabilités calibrées de vérité.

L'historique, les exports et le mode soutenance facilitent la démonstration. Je peux montrer une phrase préparée, mais une réussite sur un exemple isolé ne mesure pas la qualité générale du modèle. Les résultats de benchmark restent séparés de ces analyses de session.

Référence : app.py ; results/ui_verification.json ; tests/check_app.py.

## Slide 23: Quel modèle pour un premier déploiement ?

Pour Q11 et Q12, je proposerais A caractères comme premier prototype. C'est le candidat fixé sur val. Son représentant obtient 0,5365 sur test avec une latence de 3,62 millisecondes par tweet et environ 5,39 Mo de poids. Le coût d'entraînement représentatif est de 5,42 secondes dans ce relevé CPU.

C est plus rapide, à 1,85 milliseconde et 4,17 Mo, mais son score représentatif vaut 0,4962. D atteint une qualité proche de A, avec 88,21 millisecondes et environ 799,28 Mo, encodeur compris. Les latences excluent le chargement et ne garantissent pas le même débit sur une autre machine.

Cette proposition reste conditionnelle. Le F1 latin de A vaut environ 0,3251 et il n'y a que 75 textes mixtes test. Avant un déploiement large, je validerais le domaine cible avec de nouvelles annotations et un suivi par écriture. La campagne E et la revue humaine compléteraient ensuite l'étude, sans gains annoncés à l'avance.

Référence : results/comparison.csv ; results/generalization_analysis.md ; results/selection_lock.json. Monitoring et déploiement proposés, non exécutés.

## Slide 24: Références et index des questions

Cette annexe permet de retrouver les réponses sans interrompre le récit principal. Q1 et Q2 sont liées aux données et au protocole. Q3 concerne les indices conservés ou perdus. Q4 à Q8 comparent les méthodes et les conditions expérimentales. Q9 est discutée théoriquement, car arq est exclu. Q10 traite le décalage de distribution. Q11 et Q12 conduisent au choix pratique et à ses limites.

Les références motivent les méthodes, mais leurs résultats ne sont pas présentés comme les nôtres. Les preuves locales sont les prédictions, les métriques, les configurations et les checkpoints sauvegardés. La bibliographie détaillée, les énoncés officiels et les réponses développées accompagnent la présentation.

Références : research/papers.md ; research/questions_officielles.md ; answers.draft.md ; evidence.draft.json ; https://arxiv.org/abs/2302.08956 ; https://aclanthology.org/2023.semeval-1.315/ ; https://doi.org/10.1007/s41060-023-00498-2 ; https://aclanthology.org/Q17-1010/ ; https://arxiv.org/abs/1909.00161 ; https://arxiv.org/abs/1706.05098.
