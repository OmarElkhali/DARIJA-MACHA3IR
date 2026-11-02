# Script oral simple — 38 diapositives

Texte à dire à l’oral. Les scores sont arrondis pour faciliter la lecture ; les valeurs exactes restent sur les diapositives. Les pages 35 à 38 peuvent servir de support pendant les questions du jury.

## Diapositive 1 — Présentation du projet

Bonjour à toutes et à tous. Mon projet porte sur l’analyse de sentiment en darija marocaine. L’objectif est de reconnaître si un tweet exprime un sentiment positif, négatif ou neutre. Pour cela, j’ai comparé six familles de méthodes, des plus simples aux modèles de langue adaptés à la darija.

## Diapositive 2 — Plusieurs façons d’écrire la darija

La darija peut s’écrire en lettres arabes, en lettres latines ou avec un mélange des deux. Elle peut aussi contenir du français et des chiffres, comme dans l’Arabizi. Ces variations rendent la reconnaissance du sentiment plus difficile. Le modèle doit comprendre le message malgré ces différences d’écriture.

## Diapositive 3 — Séparation des données

J’ai utilisé 5 583 tweets pour l’apprentissage, 494 pour la validation et 2 961 pour le test. L’apprentissage sert à entraîner les modèles et la validation sert à faire les choix. Le test mesure ensuite les résultats avec ces choix fixés. Comme ce test avait déjà été consulté dans le travail précédent, une nouvelle collecte sera nécessaire pour confirmer les résultats de manière indépendante.

## Diapositive 4 — Répartition des sentiments

Les trois sentiments ne sont pas présents dans les mêmes proportions. Prédire toujours la classe neutre donne seulement environ 0,16 de F1 pondéré sur le test. J’utilise le F1 pondéré pour suivre la métrique du benchmark. Je regarde aussi le Macro-F1, qui donne la même importance à chaque classe.

## Diapositive 5 — Le risque des raccourcis

Un modèle peut associer une écriture à un sentiment sans vraiment comprendre le texte. Par exemple, les tweets latins neutres représentent 52,3 % du train, mais seulement 29,9 % du test. Un raccourci appris sur le train peut donc devenir moins utile sur le test. C’est pourquoi je regarde les résultats par écriture, en plus du score global.

## Diapositive 6 — Méthode A : TF-IDF

La première méthode transforme les mots ou les fragments de mots en nombres. Un classifieur utilise ensuite ces nombres pour prédire le sentiment. Les fragments de caractères peuvent aider lorsque l’orthographe varie. Dans cette famille, la variante caractères a été retenue sur validation et obtient environ 0,55 de F1 sur le test.

## Diapositive 7 — Temps de la méthode A

Cette méthode est rapide à entraîner. Le modèle sur les mots prend environ 2 secondes, celui sur les caractères environ 8 secondes et leur union environ 9 secondes. Les huit essais de la famille A totalisent environ 49 secondes. Elle fournit donc une référence simple avec un faible coût d’apprentissage.

## Diapositive 8 — Méthode B1 : NLI

Avec le NLI, je présente au modèle un tweet et une hypothèse, par exemple : « ce message est positif ». Je compare ensuite les hypothèses positive, négative et neutre. La formulation arabe donne ici le meilleur F1 de validation, autour de 0,50. Le modèle n’est pas réentraîné sur notre train, mais il a déjà été entraîné sur d’autres données.

## Diapositive 9 — Méthode B2 : CAMeLBERT

CAMeLBERT est déjà spécialisé dans le sentiment en arabe. Sur notre test, son F1 est d’environ 0,56 pour l’écriture arabe, contre 0,14 pour l’écriture latine. La couverture de l’Arabizi peut contribuer à cet écart, mais je n’ai pas isolé cette cause. Il n’y a pas d’entraînement local, mais la prédiction conserve un coût de calcul et l’absence de contamination externe n’est pas certifiée.

## Diapositive 10 — Gérer les classes déséquilibrées

J’ai comparé plusieurs solutions : donner plus de poids à certaines classes, répéter des exemples du train et ajuster la règle de décision sur validation. Ces solutions n’améliorent pas automatiquement les résultats. Le suréchantillonnage doit rester limité au train, après la séparation des données. Sinon, des copies pourraient se retrouver dans plusieurs partitions.

## Diapositive 11 — Méthode C : réseaux compacts

Ces réseaux représentent les mots par des vecteurs qui peuvent venir d’AraVec ou de fastText. Le BiLSTM lit la phrase dans les deux directions pour prendre en compte le contexte. Certaines variantes ajoutent une attention. La variante retenue dans C est le BiLSTM avec AraVec, sans attention supplémentaire.

## Diapositive 12 — Courbes et temps de C

Les courbes montrent que la perte d’apprentissage baisse, mais que le résultat de validation varie encore. Le run BiLSTM avec AraVec prend environ 43 secondes pour six époques observées. Il obtient environ 0,49 de F1 sur le test. Les 45 essais de la famille C totalisent environ 20 minutes.

## Diapositive 13 — Méthode D : encodeur gelé

Dans D, je garde les poids de DarijaBERT fixes. Il transforme les tweets en représentations, puis j’entraîne seulement la partie qui prédit le sentiment. Je compare trois façons d’utiliser ces représentations : CLS, la moyenne et la séquence des états. La tête de cette dernière variante contient 435 715 paramètres entraînables.

## Diapositive 14 — Résultats et temps de D

La variante states obtient environ 0,57 de F1 sur le test. Son run de référence dure environ 2 minutes, en incluant l’extraction des représentations. Il ne faut donc pas ajouter une deuxième fois ce temps d’extraction. Les cinq essais D totalisent environ 6 minutes 43 secondes.

## Diapositive 15 — Méthode E : fine-tuning

Avec E, j’adapte tout le modèle DarijaBERT-Arabizi à notre tâche. Cela représente environ 170 millions de paramètres entraînables. J’utilise un faible taux d’apprentissage et un budget maximal de cinq époques. Cette adaptation demande davantage de calcul que l’entraînement d’une petite tête sur un encodeur gelé.

## Diapositive 16 — Temps du fine-tuning

Les entraînements Arabizi prennent environ 36, 37 et 27 minutes selon l’initialisation aléatoire. Le troisième inclut une reprise après interruption. Les six essais E1, standard et Arabizi réunis, prennent environ 2 heures 51 minutes. Le représentant Arabizi obtient environ 0,61 de F1 sur le test.

## Diapositive 17 — Méthode E2 : tâche auxiliaire

E2 partage un encodeur entre la tâche marocaine et une tâche auxiliaire algérienne. Le coefficient lambda règle l’importance de cette tâche auxiliaire. Dans les essais réalisés, les scores restent légèrement sous celui d’E1. Je ne peux donc pas annoncer de gain, d’autant que les budgets d’entraînement sont différents.

## Diapositive 18 — Limites et suite de E2

Les deux essais E2 ont duré environ 23 et 24 minutes, avec trois époques et une seule initialisation. J’aimerais prolonger ces entraînements et les répéter avec plusieurs initialisations. Je souhaite aussi tester lambda égal à 0,7, qui n’a pas encore été exécuté. Ces améliorations restent à réaliser.

## Diapositive 19 — Méthode P : effet du nettoyage

Le nettoyage de référence conserve les emojis, la négation, les répétitions et les chiffres Arabizi. Dans P, j’ai supprimé les emojis et la ponctuation, puis réentraîné le modèle. Le TF-IDF union passe d’environ 0,54 à 0,57 de F1 sur le test. Ce résultat concerne ce modèle précis : il ne permet pas de généraliser à toutes les méthodes.

## Diapositive 20 — Temps et limites de P

Le TF-IDF sans signes s’entraîne en environ 15 secondes. DarijaBERT sans signes prend environ 20 minutes et atteint environ 0,76 en validation. Son fichier de poids manque dans la sauvegarde disponible : il est donc exclu du test. L’étude du nettoyage n’est pas encore complète pour tous les modèles.

## Diapositive 21 — Comment le modèle est choisi

Le choix du modèle repose sur la validation, avant la comparaison finale sur test. Quand plusieurs initialisations sont disponibles, je compare leur moyenne de validation. DarijaBERT-Arabizi est retenu avec un F1 moyen d’environ 0,81. Je ne remplace pas ce choix après avoir vu les résultats du test.

## Diapositive 22 — Résultats principaux

Sur le test, E DarijaBERT-Arabizi atteint environ 0,61 de F1 pondéré. D et P se situent autour de 0,57, puis viennent A, C et B. E était déjà le candidat choisi sur validation. Pour juger son intérêt, il faut aussi regarder les erreurs, l’incertitude et le coût.

## Diapositive 23 — Comparaison avec les publications

J’ai comparé notre résultat avec des publications utilisant AfriSenti pour la darija marocaine. Notre F1 pondéré est de 60,81 %, contre 64,82 % pour NLNDE, 61,54 % pour ABCD et 60,15 % pour UM6P. Cette comparaison reste indicative : les validations, la préparation et les combinaisons de modèles diffèrent. Elle ne constitue donc pas un classement direct à protocole identique.

## Diapositive 24 — Comprendre les confusions

La matrice montre les vrais sentiments et les sentiments prédits. La diagonale correspond aux bonnes réponses ; les autres cases montrent les confusions. Par exemple, 299 tweets négatifs sont prédits neutres. Cela indique quels cas relire pour mieux comprendre les erreurs.

## Diapositive 25 — Pourquoi le score baisse sur test

Les scores sont plus faibles sur test que sur validation. Les différences de distribution et d’écriture peuvent contribuer à cette baisse, sans tout expliquer à elles seules. Chercher après coup une meilleure règle sur le test reste un diagnostic. Cette règle ne doit pas remplacer celle choisie sur validation.

## Diapositive 26 — Une écriture absente de l’apprentissage

J’ai entraîné le TF-IDF sur une écriture, puis testé sur l’autre. Le F1 tombe à environ 0,23 de l’arabe vers le latin et à 0,21 dans l’autre sens. Le transfert est donc difficile dans cette expérience. Avant un déploiement, il faut vérifier que les données couvrent les écritures réellement utilisées.

## Diapositive 27 — Incertitude des résultats

Un seul score ne résume pas toute l’incertitude. Pour E, l’intervalle obtenu par rééchantillonnage du test est d’environ 0,59 à 0,62. Il concerne le modèle évalué et ne couvre pas toutes les variations possibles d’entraînement. De petites différences entre scores ne suffisent donc pas à prouver qu’un modèle est supérieur.

## Diapositive 28 — Qualité et coût d’utilisation

Le meilleur choix dépend aussi de la vitesse de prédiction et de la taille du modèle. Un modèle plus lourd peut demander davantage de ressources même si son score est meilleur. Les temps présentés ont été mesurés sur le matériel de la campagne. Ils doivent être vérifiés sur la machine de déploiement avant de promettre un temps de réponse.

## Diapositive 29 — Temps total enregistré

Les journaux comptent 72 exécutions distinctes, y compris les quatre transferts fixes B. Le temps cumulé enregistré est d’environ 4 heures 27 minutes. La famille E en représente la plus grande partie, avec environ 3 heures 39 minutes. Ce total ne comprend pas les téléchargements ni le temps perdu lors des coupures.

## Diapositive 30 — Difficultés avec Colab

J’ai dû changer de session Colab quatre fois, car les sessions se terminaient avant la fin de certains entraînements. Il fallait relancer l’environnement, restaurer les fichiers et reprendre les calculs. La reprise d’E Arabizi avec l’initialisation 44 est enregistrée. Je n’ai pas de mesure précise du temps perdu pendant ces interruptions.

## Diapositive 31 — Améliorations souhaitées

Je souhaite comparer quatre versions des textes : intacte, sans emojis, sans ponctuation et sans les deux. Pour les modèles entraînés localement, il faudra refaire l’apprentissage dans chaque condition. Je souhaite aussi prolonger E2 et améliorer les sauvegardes pour reprendre plus facilement après une coupure. Ces expériences ne sont pas encore réalisées.

## Diapositive 32 — Analyse humaine des erreurs

J’ai préparé trente erreurs pour une relecture humaine. L’ironie, l’écriture et l’ambiguïté sont des explications possibles, mais elles restent à vérifier. Un désaccord entre le modèle et le label ne prouve pas que le label est faux. Je ne peux donc pas encore annoncer une proportion de bruit d’annotation.

## Diapositive 33 — Laboratoire interactif

Le site permet de consulter les résultats et de tester des phrases avec les modèles disponibles. Cinq familles ont été vérifiées avec de vraies prédictions. Le checkpoint ajusté d’E manque localement, et cette limite est indiquée. La démonstration complète utilise un serveur Python local ; GitHub Pages présente la partie statique du site.

## Diapositive 34 — Conclusion

Le principal résultat est DarijaBERT-Arabizi, choisi sur validation et évalué à environ 0,61 de F1 pondéré sur test. Le projet montre aussi l’importance des écritures, des erreurs et des ressources de calcul. La suite consiste à terminer la revue humaine et à tester sur de nouvelles données indépendantes. Merci pour votre attention ; je suis prêt à répondre à vos questions.

## Diapositive 35 — Annexe : courbes de fine-tuning

Cette page permet de revenir sur l’évolution des entraînements. La perte baisse, mais le résultat de validation peut encore varier. Les essais E2 ont été limités à trois époques. Ces courbes décrivent les exécutions enregistrées ; elles ne démontrent pas à elles seules l’intérêt de la tâche auxiliaire.

## Diapositive 36 — Références et guide

Cette page rassemble les principales références du corpus et des modèles. Elle donne aussi un index des douze questions du projet. Les deux pages suivantes résument les réponses et indiquent où retrouver les explications. Les publications citées et nos propres mesures restent bien distinguées.

## Diapositive 37 — Questions Q1 à Q6

**Q1.** Je retiens le F1 pondéré pour suivre le benchmark et le Macro-F1 pour donner le même poids aux trois classes. Les données et le protocole sont détaillés pages 3 et 4.

**Q2.** La relation entre écriture et sentiment change entre train et test. Un modèle peut donc apprendre un raccourci fragile ; je l’explique pages 5, 25 et 26.

**Q3.** Je conserve les indices de sentiment par défaut. Le résultat du nettoyage dépend du modèle, et les ablations complètes restent à réaliser ; voir pages 19, 20 et 31.

**Q4.** Le NLI compare le tweet à trois hypothèses de sentiment. Il n’utilise pas d’apprentissage local sur notre train, mais il bénéficie d’un entraînement externe ; voir page 8.

**Q5.** CAMeLBERT fonctionne mieux sur l’écriture arabe que sur l’écriture latine. La couverture de l’Arabizi est une piste, et l’absence de contamination n’est pas certifiée ; voir page 9.

**Q6.** Je compare la pondération, le suréchantillonnage et l’ajustement de décision. Le suréchantillonnage reste limité au train pour éviter les fuites ; voir pages 10 et 21.

## Diapositive 38 — Questions Q7 à Q12

**Q7.** Dans C, le BiLSTM avec AraVec est retenu. Une attention visualisée ne prouve pas qu’un mot cause la décision ; voir pages 11 et 12.

**Q8.** E Arabizi dépasse D d’environ 3,65 points de F1 test, mais ajuste environ 170 millions de paramètres. Son run de référence prend environ 36 minutes ; voir pages 13 à 16, 21 et 22.

**Q9.** Les deux essais E2 ne montrent pas de gain établi face à E1. Le budget plus court et l’unique initialisation limitent la conclusion ; voir pages 17, 18 et 27.

**Q10.** Je garde la décision choisie sur validation. En production, je suivrais les changements de données et vérifierais une recalibration sur de nouvelles données annotées ; voir pages 3, 21 et 25.

**Q11.** Le transfert vers une écriture absente du train donne des scores faibles. Il faut couvrir les deux écritures et valider le domaine cible ; voir pages 26 et 34.

**Q12.** E reste le candidat choisi sur validation ; A peut être étudié comme solution légère. Le choix final demande de vérifier le coût, de restaurer E et de terminer la revue humaine ; voir pages 21, 22, 28, 29 et 32 à 34.
