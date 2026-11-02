# Analyse a posteriori de la généralisation

Les choix ont été verrouillés avant ces observations. Aucun réglage ne change.

- Pour les textes latins, la classe neutre représente 52.3% du train, 52.6% de dev et 29.9% du test. La relation écriture/sentiment change davantage que la distribution globale des classes.
- Le candidat classe 86.8% des textes latins test comme neutres. Cette prédominance est compatible avec un raccourci appris sur train/dev ; elle ne prouve pas à elle seule son mécanisme causal.
- Le Weighted-F1 test de la règle brute vaut 0.5536, celui des décalages choisis sur dev 0.5365. Malgré cela, le résultat principal conserve la règle fixée sur dev. Choisir la règle brute parce qu'elle gagne sur test serait une nouvelle sélection sur test.
- Les taux OOV sont enregistrés dans oov.csv : ils décrivent le vocabulaire de mots min_freq=2, pas directement la couverture des n-grammes du candidat TF-IDF caractères. Ils ne suffisent pas à expliquer causalement sa baisse.
- Les mentions/URLs concernent 1 tweet train, 0 tweet dev et 1335 tweets test. Le remplacement conservateur par USER/URL crée donc des formes presque absentes de l'apprentissage. Une influence sur les prédictions est plausible, mais aucune ablation causale de ces marqueurs n'a été exécutée ; la règle de nettoyage reste celle fixée avant test.
- Des marqueurs littéraux anormaux, dont [[positiveHOTO]] et COpositive22, apparaissent dans les fichiers bruts officiels (effectifs dans corpus_markers.csv). Leur origine n'est pas démontrée par cet audit et leur présence ne prouve pas une fuite exploitable. Les données officielles restent intactes.
- Les taux de perturbation F4 de familles différentes sont conditionnés à des sous-ensembles initiaux différents. Éviter de les interpréter comme un classement causal de robustesse sur une population commune.

Hypothèses pour une future étude indépendante : contrôler les raccourcis d'écriture, annoter les domaines/cibles, examiner la normalisation des mentions et les marqueurs de corpus, puis valider ces interventions sans réutiliser ce test pour les choisir. Aucune de ces interventions n'est déclarée efficace ici.
