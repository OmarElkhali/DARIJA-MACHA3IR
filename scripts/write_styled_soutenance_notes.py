"""Notes orales de la soutenance actualisée ; aucun notebook exécuté."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
DECK=ROOT/'presentation/Darija_Sentiment_Soutenance_Resultats'
spec=json.loads((DECK/'deck_spec.json').read_text(encoding='utf-8'))
notes=[
'''Notre projet cherche à reconnaître le sentiment exprimé dans des tweets en darija marocaine : négatif, neutre ou positif. La difficulté ne se limite pas au choix du réseau. Une même langue s’écrit avec des alphabets différents, des chiffres et des mélanges de langues.

Nous avons donc comparé six familles sur les partitions officielles d’AfriSenti. Je vais d’abord expliquer ce qui rend les données difficiles, puis montrer comment nous avons choisi les modèles et ce que les résultats permettent de conclure. Les performances présentées viennent des sorties sauvegardées de la campagne.''',
'''Ces trois phrases illustrent des écritures différentes. Elles ne sont ni des traductions exactes ni des exemples dont nous revendiquons une prédiction mesurée. Regardons les chiffres Arabizi et la négation : les supprimer peut retirer une partie du sens.

Le troisième exemple mélange arabe et français et contient deux appréciations opposées. Le modèle doit interpréter cette combinaison. Détecter l’alphabet ne suffit donc pas à comprendre le sentiment. Cela nous conduit à la première exigence du travail : une évaluation qui distingue les écritures et qui évite les raccourcis de corpus.''',
'''Les rôles des partitions sont séparés. Les 5 583 tweets de train servent à apprendre les transformations et les poids. Les 494 tweets de validation servent à choisir la variante, l’époque et les décalages de décision. Nous figeons ensuite ces choix avant de mesurer sur les 2 961 tweets de test.

Le représentant annoncé pour chaque famille utilise la graine 42 ; quand plusieurs graines existent, la sélection de variante utilise leur moyenne de validation. Il faut toutefois conserver une limite historique : le test avait déjà été consulté dans le travail antérieur. Cette campagne reste donc exploratoire. Une confirmation indépendante demanderait de nouvelles données annotées et réservées.''',
'''Avant de discuter des architectures, nous regardons les classes. La classe neutre est la plus fréquente sur train et validation, tandis que la distribution test diffère. Le graphique fourni montre les proportions enregistrées de validation et de test, sans transformer ce constat en preuve d’un mécanisme causal.

Un classifieur qui prédit toujours neutre obtient seulement 0,160795 de F1 pondéré sur test. Cette référence montre pourquoi une distribution de classes ne constitue pas à elle seule une solution. Le F1 pondéré combine précision et rappel par classe en tenant compte des effectifs ; ce n’est pas l’accuracy. Le Macro-F1 traite les trois classes à poids égaux.''',
'''La relation entre sentiment et écriture change. Parmi les tweets latins, la proportion de neutres passe de 52,3 % sur train à 29,9 % sur test. Pour l’arabe, elle passe au contraire de 29,2 % à 36,1 %. Le groupe mixte change lui aussi.

Un modèle peut exploiter une corrélation entre l’écriture et le label pendant l’apprentissage. Lorsque cette corrélation varie, ce raccourci devient fragile. Les chiffres sont descriptifs : ils ne prouvent pas que l’alphabet cause un sentiment ou qu’il explique toute la baisse. Nous allons garder cette question en tête pendant la lecture des résultats.''',
'''Le nettoyage de référence conserve les indices potentiellement utiles : négation, chiffres Arabizi, répétitions et emojis. L’exemple affiché sert uniquement à montrer une transformation, sans prédiction associée.

Dans P, nous réapprenons le TF-IDF union après suppression conjointe des emojis et de la ponctuation. Son représentant obtient 0,567396 sur test, contre 0,539908 pour l’union de référence. C’est une comparaison descriptive de pipelines ; elle ne justifie pas de supprimer tous les indices pour toutes les familles. Le run P DarijaBERT est exclu du test faute de poids. Enfin, P modifie le pipeline d’apprentissage, tandis que F4 perturbe les textes d’évaluation d’un modèle déjà appris.

Pour F4, retirer les deux types de signes modifie 1 246 tweets. Le modèle E change de décision sur 19,26 % de ces textes et son F1 passe de 0,608138 à 0,589451, avec les labels d’origine. Une relecture doit confirmer que la transformation conserve réellement le sentiment avant de conclure à un défaut de robustesse.''',
'''TF-IDF représente un tweet par l’importance de ses mots ou fragments. Les mots utilisent des n-grammes de longueur un à deux et les caractères de deux à cinq. Des variantes d’un même mot peuvent ainsi partager certains fragments même si leur orthographe diffère. La régression logistique transforme ce vecteur en scores pour les trois classes.

Dans la famille A, les caractères sont choisis sur validation avec 0,764076, puis obtiennent 0,553509 sur test. Une autre variante peut avoir un meilleur score test, mais elle ne remplace pas après coup le candidat fixé sur validation. C’est cette règle qui rend la comparaison interprétable.''',
'''Le NLI transforme la classification en question d’implication. Le tweet est la prémisse. Nous lui associons successivement une hypothèse de sentiment négatif, neutre et positif. Le score d’implication permet de comparer ces trois hypothèses.

Nous essayons trois formulations linguistiques. Après ajustement sur validation, les F1 sont 0,501026 en arabe, 0,497245 en français et 0,463444 en anglais. Cette différence ne mesure pas un effet pur de langue : nous changeons aussi les mots de l’hypothèse. Les poids sont fixes sur notre corpus, mais le modèle a déjà reçu une supervision externe. Les décalages de décision, eux, sont appris sur validation.''',
'''CAMeLBERT est déjà spécialisé en sentiment arabe. Il ne reçoit pas d’entraînement local sur ary dans cette expérience. Sur le sous-groupe arabe du test, il obtient environ 0,558 ; sur le sous-groupe latin, environ 0,144. Les deux populations comptent respectivement 1 367 et 1 519 tweets.

L’écart montre l’importance de regarder les sous-groupes. Une différence de couverture linguistique est plausible, mais elle n’a pas été isolée expérimentalement. Il faut aussi conserver la provenance de la supervision externe : l’absence de contamination n’est pas certifiée. Nous parlons donc de transfert fixe localement, pas d’un modèle qui n’aurait jamais vu de labels.''',
'''Nous distinguons deux endroits où agir. Sur train, une perte pondérée augmente l’importance de certaines classes, tandis que le suréchantillonnage change leur fréquence dans les lots. Le découpage des partitions précède ces interventions.

Sur la référence BiLSTM avec AraVec, la moyenne de validation est 0,753140 ; elle est d’environ 0,74394 pour les deux interventions. Elles n’améliorent donc pas automatiquement le critère de sélection. À la décision, une autre intervention consiste à ajouter des décalages aux logarithmes des probabilités. Ces décalages sont choisis sur validation puis figés. Ils ne constituent pas une calibration garantie des probabilités.''',
'''Suivons le schéma de gauche à droite. Les tokens deviennent des vecteurs denses, initialisés au hasard, avec AraVec ou avec fastText. Le BiLSTM parcourt la séquence dans les deux directions, ce qui lui permet de réunir du contexte à gauche et à droite de chaque position.

Une couche d’attention peut pondérer les représentations avant la classification. Elle fait partie des variantes, mais la variante C retenue est le BiLSTM avec AraVec sans cette couche d’attention. Visualiser un poids d’attention explique une opération du modèle ; cela ne démontre pas qu’un mot cause sa décision. Les inconnus et le padding doivent également être traités correctement.''',
'''Les courbes montrent les pertes d’apprentissage et les F1 de validation observés au fil des époques. La perte baisse souvent alors que la validation fluctue ou plafonne. Nous retenons l’époque selon la validation, et non selon la qualité du test.

Les traces de validation pendant l’entraînement utilisent la décision brute. Le F1 final de validation peut ensuite changer après les décalages choisis sur validation : il ne faut pas confondre ces deux valeurs. La variante C choisie atteint une moyenne finale de validation de 0,753140 et son représentant 42 un F1 test de 0,493340. Le graphique décrit les runs sauvegardés ; il ne prouve pas une explication causale du surapprentissage.

Nous avons aussi comparé Transformer avec et sans positions et Character-CNN. Sur validation, les positions aident pour fastText et AraVec, mais pas pour l’initialisation aléatoire dans cette campagne. Le Character-CNN atteint 0,730284 en moyenne. Ni l’attention ni une architecture plus complexe ne gagnent systématiquement ; c’est pourquoi nous conservons le choix expérimental plutôt qu’une préférence théorique.''',
'''Les deux branches permettent de voir ce qui apprend. Dans D, l’encodeur DarijaBERT est gelé : nous utilisons ses représentations et apprenons une tête. Avec la séquence d’états, le représentant test atteint 0,571593. Dans E, les gradients ajustent également les poids de l’encodeur.

Le fine-tuning standard obtient 0,579512 et la variante Arabizi 0,608138 pour les représentants 42. E Arabizi est aussi la variante retenue sur validation. La comparaison ne permet pas d’attribuer tout l’écart à une seule propriété du préentraînement. Il faut examiner ensemble qualité, couverture des écritures, coût et variabilité entre graines.''',
'''L’idée du multi-tâche est de partager l’encodeur entre la tâche principale marocaine ary et une tâche auxiliaire algérienne arq. Le coefficient lambda règle le poids de la perte auxiliaire. Un partage utile peut aider, mais un transfert négatif est aussi possible.

Ici, le plan réduit mesure lambda égal à 0,1 et 0,3, avec une graine et trois époques maximum. Leurs F1 test sont 0,605137 et 0,604073. E Arabizi sans auxiliaire atteint 0,608138 pour le représentant 42. Nous ne revendiquons pas de gain de l’auxiliaire dans ce plan. Les budgets et nombres de graines diffèrent, et lambda égal à 0,7 n’a pas été évalué.''',
'''Le tableau de sélection ne se lit pas comme le classement test. Nous comparons d’abord le F1 de validation des variantes et utilisons la moyenne des graines quand elles sont disponibles. Toutes les familles ne possèdent pas trois graines, ce qui doit rester visible dans l’interprétation.

E Arabizi est le candidat retenu, avec 0,811256 de moyenne de validation. Les six représentants utilisent la graine 42 pour la comparaison annoncée. La campagne sauvegardée comporte 33 variantes et 71 évaluations finales après l’exclusion de P DarijaBERT. Le verrou protège la règle de décision ; il ne transforme pas un test déjà consulté historiquement en nouvelle évaluation indépendante.''',
'''Voici le résultat principal. Chaque barre correspond au représentant 42 choisi selon la validation dans sa famille. E DarijaBERT-Arabizi atteint 0,608138 de F1 pondéré et 0,600620 de Macro-F1. D et P se situent ensuite autour de 0,57 ; A atteint environ 0,55, C environ 0,49 et B environ 0,42.

Le modèle E n’est pas choisi parce que nous avons regardé ce graphique : il était déjà retenu sur validation. Nous ne confondons pas non plus le score d’un représentant et la moyenne test de plusieurs graines. Le graphique situe les performances mesurées, mais un classement seul ne renseigne ni sur l’incertitude, ni sur les erreurs, ni sur le coût.''',
'''La matrice appartient au nouveau candidat E Arabizi. Les lignes indiquent les labels réels et les colonnes les prédictions. La diagonale donne les décisions correctes ; les autres cellules montrent où les classes se confondent.

Nous comptons 440 négatifs, 652 neutres et 702 positifs correctement classés. Les négatifs sont notamment souvent prédits neutres : cette cellule contient 299 tweets. Cela donne une direction pour la revue qualitative, sans expliquer automatiquement pourquoi le modèle se trompe. Le total correspond bien aux 2 961 tweets du test. Une moyenne globale peut masquer ces différences de comportement par classe.''',
'''Les cercles ouverts représentent la validation et les cercles pleins le test. Pour chaque famille, le score baisse entre ces deux lectures. Pour E, nous passons de la moyenne de validation 0,811256 au représentant test 0,608138. Comme il s’agit d’une moyenne d’un côté et d’un représentant de l’autre, ce n’est pas une décomposition causale de la baisse.

Le changement des distributions et des écritures fournit des pistes, mais ne constitue pas une preuve d’un pur changement de prior. Le balayage d’une grille sur test reste un diagnostic rétrospectif optimiste. Son meilleur point ne remplace pas les décalages verrouillés et ne peut pas servir à annoncer une performance indépendante.''',
'''Nous entraînons ici la même architecture TF-IDF union sur une écriture, puis la testons sur l’autre. Le transfert arabe vers latin donne 0,2254 de F1 ; le transfert latin vers arabe donne 0,2110. Nous utilisons les populations test correspondantes, soit 1 519 tweets latins et 1 367 arabes.

Ces niveaux soulignent la difficulté de couvrir une écriture absente de l’apprentissage. Il faut cependant regarder le témoin avec prudence : la taille du train et sa couverture changent ensemble. L’expérience ne mesure donc pas un effet pur de l’alphabet. La conclusion opérationnelle est de valider la couverture des écritures présentes dans le domaine cible.''',
'''Un point n’est pas toute l’information. Les intervalles proviennent de 1 000 rééchantillonnages bootstrap du test, conditionnels au checkpoint évalué. Pour E, l’intervalle à 95 % est de 0,590655 à 0,624099. Il ne représente pas toutes les sources d’incertitude possibles.

La variabilité entre graines répond à une autre question : que se passe-t-il si l’apprentissage est répété ? Enfin, le test exact de McNemar entre E et E2 lambda 0,3 compare les erreurs appariées : 143 tweets sont corrects uniquement pour E et 144 uniquement pour E2, avec p égal à 1. Ce test ne porte pas directement sur la différence de F1. Le simple ordre des points ne suffit pas à annoncer une supériorité statistique.''',
'''La qualité doit être mise en regard du coût. Le graphique original relie le F1 test à trois mesures : la latence CPU par tweet, le poids des fichiers et le temps d’apprentissage enregistré. Les axes horizontaux sont logarithmiques ; il faut donc éviter de lire les distances comme sur une échelle linéaire.

La latence est une mesure unitaire dans l’environnement de la campagne, pas un débit GPU ni une promesse de service. Les poids sur disque ne sont pas la mémoire vive. Les colonnes numériques tronquées n’ont pas été reconstituées pour cette présentation. Enfin, les temps cumulés évitent les doubles comptes raw/tuned et reprises ; les époques perdues avant checkpoint et les coupures réseau ne sont pas mesurables.''',
'''Nous avons préparé trente erreurs pour une relecture, mais aucune validation humaine n’est encore enregistrée. Les catégories comme ironie, graphie ou ambiguïté doivent donc être comprises comme des hypothèses de travail.

Le chemin affiché distingue l’observation et sa conclusion : constater un désaccord, proposer une explication, faire relire, puis décider ce que les preuves soutiennent. Un modèle différent du label ne prouve pas une erreur d’annotation. De même, un petit échantillon ciblé ne permet pas d’estimer la fréquence de ces phénomènes dans toutes les erreurs. La revue humaine et un nouvel ensemble indépendant restent nécessaires.''',
'''Le site présente la méthode, les résultats et un laboratoire de phrases. L’interface est en HTML, CSS et JavaScript ; les véritables modèles sont chargés par un serveur Python local. GitHub Pages héberge la partie statique, pas les calculs PyTorch.

Les six familles sont affichées, mais cinq ont été testées avec de véritables prédictions. Les poids ajustés du modèle E manquent dans les fichiers locaux, et cette absence est signalée. Nous ne remplaçons pas E par une simulation. Une courte démonstration permet de voir les désaccords entre méthodes, mais une phrase réussie ou ratée n’est pas une nouvelle mesure du F1 de benchmark.''',
'''Notre résultat mesuré principal est DarijaBERT-Arabizi, retenu sur validation, avec 0,608138 de F1 pondéré et 0,600620 de Macro-F1 sur le représentant test. Le travail montre aussi pourquoi un résultat global doit être accompagné d’une lecture par écriture, d’une incertitude et d’un coût.

La prochaine étape n’est pas de continuer à régler le même test. Elle consiste à obtenir de nouvelles annotations indépendantes dans le domaine cible, terminer la revue des erreurs et restaurer le checkpoint E pour la démonstration complète. Nous proposons donc un prototype accompagné de ses limites, sans garantie de fiabilité universelle. Le notebook, le site et le dépôt rendent les preuves accessibles pour poursuivre ce travail.''',
'''Cette annexe permet de revenir sur le fine-tuning pendant les questions. À gauche, la perte d’apprentissage baisse ; à droite, les F1 bruts de validation décrivent les époques enregistrées. Les runs multi-tâches du plan réduit s’arrêtent à trois époques maximum, alors que d’autres runs peuvent aller plus loin.

Ces courbes brutes précèdent le réglage des décalages de décision. Elles ne sont donc pas obligées d’atteindre le F1 final ajusté du tableau de sélection. Nous ne les avons ni lissées ni réexécutées pour fabriquer une meilleure trajectoire. Elles restent des traces de la campagne, pas une preuve causale d’un effet du multi-tâche.''',
'''Les sources distinguent les données, les modèles et nos propres mesures. AfriSenti fournit le corpus et le cadre de la tâche. Les travaux sur DarijaBERT et fastText décrivent les représentations employées. Leurs résultats publiés ne sont pas les scores de notre campagne.

L’index à droite permet de retrouver les réponses aux questions du projet : protocole, nettoyage, transfert, déséquilibre, réseaux, fine-tuning, auxiliaire, diagnostics, écritures et erreurs. Le dépôt contient les notebooks avec leurs sorties, les sources et les livrables. Les mesures de cette soutenance sont reliées aux sorties sauvegardées, sans nouvel entraînement pour préparer les diapositives.'''
]
assert len(notes)==len(spec['slides'])==26
speech='\n\n'.join(f"## Slide {slide['number']}: {slide['title']}\n\n{note}" for slide,note in zip(spec['slides'],notes))+'\n'
(DECK/'speech.md').write_text(speech,encoding='utf-8')
(DECK/'sample_qa.json').write_text(json.dumps({'slide':16,'accepted':True,
 'intermediate_confirmation_waived':True,'source':'Same style as original soutenance explicitly requested; persistent no-question authorization',
 'checks':['six current family labels and F1 values correct','E selected on validation, not A','Macro-F1 current','bar lengths proportional on0to1axis','French readable and no clipping','same ivory forest sage style']},ensure_ascii=False,indent=2),encoding='utf-8')
(DECK/'README.md').write_text('''# Darija Sentiment — Soutenance actualisée

26 diapositives en 16:9, dans la même direction éditoriale ivoire/forêt/sauge et avec la même méthode de génération d’images que `Darija_Sentiment_Soutenance`.

Le PowerPoint contient des pages-images complètes ; les textes et graphes ne sont pas séparément éditables. Les notes orales françaises sont intégrées et livrées dans `speech.md`. Le parcours principal compte 24 pages, suivi des courbes E et des références en annexe ; prévoir environ 20–25 minutes en adaptant les notes.

Les chiffres viennent des sorties sauvegardées de campaign_v4. E1 et E2 sont désormais mesurés. E Arabizi est le candidat retenu. P DarijaBERT reste exclu du test, la revue humaine est inachevée et le laboratoire possède cinq checkpoints locaux. Le test ayant été consulté historiquement, la campagne reste exploratoire. Aucun notebook n’a été exécuté pour ce livrable.

Les jobs, dispatchs et résultats sont enregistrés avec les scripts codex-ppt. Chaque page est générée par l’outil d’images intégré, puis vérifiée avant assemblage ; aucune diapositive n’est dessinée localement. `outline.md`, `deck_spec.json`, `slide_jobs.json` et `slide_run_state.json` assurent la traçabilité. L’ancienne soutenance sert de référence visuelle ; ses anciens scores ne sont pas réutilisés.

Les images ont été inspectées et la structure PPTX vérifiée ; aucune inspection native dans PowerPoint n’est revendiquée.
''',encoding='utf-8')
print('Notes françaises prêtes pour 26 diapositives ; aperçu et chiffres de la référence vérifiés.')
