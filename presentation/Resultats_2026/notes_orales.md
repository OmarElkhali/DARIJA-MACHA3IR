## 1. Darija / Sentiment





Source : Sorties enregistrées de campaign_v4.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

## 2. Le candidat retenu est DarijaBERT-Arabizi

Le candidat retenu est DarijaBERT-Arabizi

Le choix vient de la validation ; le test mesure ensuite ce choix.

Source : comparison, cellule G.1.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

À expliquer : La baisse de validation vers test est un résultat central. Elle ne justifie pas de changer le candidat après coup.

## 3. Le sentiment traverse les écritures

Le sentiment traverse les écritures

Trois classes : négatif, neutre, positif.

Source : Corpus et protocole, notebook final.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

## 4. 9 038 tweets, trois partitions officielles

9 038 tweets, trois partitions officielles

Train 5 583 · validation 494 · test 2 961. Les partitions restent intactes.

Source : data/ary/*.tsv, révision figée.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

## 5. Apprendre, choisir, figer, mesurer

Apprendre, choisir, figer, mesurer

Quatre rôles distincts pour éviter une nouvelle sélection sur test.

Source : src/campaign.py et src/evaluate.py.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

À expliquer : Le test avait déjà été observé dans l’étude historique : une confirmation indépendante nécessite de nouveaux textes annotés.

## 6. Nettoyer sans effacer les indices de sentiment

Nettoyer sans effacer les indices de sentiment

Le nettoyage de référence est figé avant les mesures test.

Source : src/preprocess.py, src/models.py.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

## 7. Les six familles comparées

Les six familles comparées

Le laboratoire utilise un représentant par famille, choisi sur validation.

Source : Code des familles et sorties sauvegardées.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

## 8. A : une référence simple et mesurable

A : une référence simple et mesurable

TF-IDF mots, caractères et union ; C choisi sur validation.

Source : Outputs du notebook Colab F/G.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

À expliquer : Le représentant A est le TF-IDF caractères, choisi sur validation, même si une autre variante gagne sur test.

## 9. B : le transfert fixe dépend du domaine

B : le transfert fixe dépend du domaine

Les poids ne sont pas entraînés sur ary ; les modèles ont déjà été supervisés ailleurs.

Source : Outputs du notebook Colab F/G.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

À expliquer : Les trois NLI changent langue et formulation. Une différence ne peut pas être attribuée à la langue seule.

## 10. C : comparer architectures et embeddings

C : comparer architectures et embeddings

Le vocabulaire est appris sur train, fréquence minimale 2.

Source : src/models.py et src/train.py.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

## 11. C : l’ordre de validation reste la référence

C : l’ordre de validation reste la référence

Les huit premières variantes C selon la validation sont montrées ici.

Source : Outputs du notebook Colab F/G.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

À expliquer : Le meilleur couple architecture-embeddings est choisi sur la moyenne des graines, avant perte pondérée et oversampling.

## 12. C : les courbes décrivent les époques observées

C : les courbes décrivent les époques observées

Les sorties d’apprentissage sont sauvegardées ; elles ne prouvent pas une explication causale.

Source : Notebook principal, cellule des historiques C/D/E/P.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

À expliquer : Les époques perdues avant checkpoint ne sont pas récupérables dans ces mesures.

## 13. D : geler BERT ne rend pas son inférence gratuite

D : geler BERT ne rend pas son inférence gratuite

CLS, moyenne masquée et séquence avec tête BiLSTM-attention.

Source : Outputs du notebook Colab F/G.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

À expliquer : DarijaBERT reste exécuté sur chaque texte libre. Le cache de représentations est un outil expérimental.

## 14. E1 : adapter l’encodeur au sentiment marocain

E1 : adapter l’encodeur au sentiment marocain

DarijaBERT et DarijaBERT-Arabizi, graines 42 / 43 / 44.

Source : Outputs du notebook Colab F/G.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

À expliquer : AdamW 2×10⁻⁵, batch effectif 32, micro-batch 2, warmup 10 %, perte pondérée, jusqu’à cinq époques.

## 15. E2 : l’auxiliaire algérien n’est pas supposé aider

E2 : l’auxiliaire algérien n’est pas supposé aider

Deux valeurs de λ, une graine et trois époques maximum dans le plan réduit.

Source : Outputs du notebook Colab F/G.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

À expliquer : Comparer à E1 avec prudence : le budget et le nombre de graines diffèrent. λ=0,7 n’a pas été évalué ici.

## 16. E : regarder aussi les courbes du fine-tuning

E : regarder aussi les courbes du fine-tuning

L’époque est retenue sur le F1 validation brut ; les décalages sont réglés ensuite.

Source : Historiques E sauvegardés.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

À expliquer : Une courbe de validation n’est pas une nouvelle estimation de généralisation.

## 17. P : changer le pipeline de nettoyage

P : changer le pipeline de nettoyage

Réapprendre un TF-IDF union sans emojis ni ponctuation.

Source : Outputs du notebook Colab F/G.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

À expliquer : P DarijaBERT conserve ses scores dev mais est exclu du test : le poids manque. Aucun score test ne lui est attribué.

## 18. Validation et test : la baisse concerne les six familles

Validation et test : la baisse concerne les six familles

Cercle ouvert : validation moyenne. Cercle plein : test de la graine 42.

Source : Outputs du notebook Colab F/G.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

À expliquer : La variation d’écriture, les distributions conditionnelles et les raccourcis sont des pistes, pas des causes démontrées.

## 19. F1 : les proportions globales ne suffisent pas

F1 : les proportions globales ne suffisent pas

Les distributions restent descriptives, après verrouillage.

Source : Outputs du notebook Colab F/G.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

À expliquer : Un changement de proportions ne démontre pas un pur label shift. Examiner aussi les distributions conditionnelles.

## 20. F1 : le neutre varie selon l’écriture

F1 : le neutre varie selon l’écriture

La relation alphabet / sentiment doit être examinée conditionnellement.

Source : Distributions affichées, cellule F1.1.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

## 21. F1 : le maximum sur test reste un oracle

F1 : le maximum sur test reste un oracle

La grille post hoc ne modifie ni le candidat ni sa règle.

Source : Outputs du notebook Colab F/G.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

À expliquer : Optimiser de nouveau sur cette grille test invaliderait la séparation des rôles du protocole.

## 22. F2 : transférer vers l’écriture non vue

F2 : transférer vers l’écriture non vue

Le TF-IDF union est entraîné sur une seule écriture.

Source : Sortie F2 du notebook Colab.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

À expliquer : La taille du train et sa couverture changent ensemble. L’écart n’isole pas causalement l’alphabet.

## 23. F3 : une moyenne peut cacher un sous-groupe

F3 : une moyenne peut cacher un sous-groupe

Résultats des six représentants par écriture.

Source : Outputs du notebook Colab F/G.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

À expliquer : Le groupe mixte peut être petit. Le détecteur identifie l’alphabet, pas la langue.

## 24. F4 : mesurer les bascules de décision

F4 : mesurer les bascules de décision

Même population test pour les trois perturbations déclarées.

Source : Outputs du notebook Colab F/G.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

À expliquer : Les labels originaux sont conservés pour ce diagnostic. Une transformation peut changer le sentiment véritable.

## 25. F5 : bootstrap et graines mesurent deux variabilités

F5 : bootstrap et graines mesurent deux variabilités

IC bootstrap 95 %, conditionnel au checkpoint 42, sur 1 000 rééchantillonnages.

Source : Outputs du notebook Colab F/G.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

À expliquer : L’écart-type entre graines décrit les entraînements. Il ne remplace pas l’intervalle d’échantillonnage des tweets.

## 26. F5 : comparer des erreurs appariées

F5 : comparer des erreurs appariées

Le score et sa significativité restent des questions distinctes.

Source : McNemar, cellule F5.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

## 27. F6 / F7 : la revue humaine reste indispensable

F6 / F7 : la revue humaine reste indispensable

Un échantillon diversifié ne mesure pas la fréquence des catégories dans toutes les erreurs.

Source : F6 et F7, outputs enregistrés.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

## 28. G : qualité, latence CPU et taille des poids

G : qualité, latence CPU et taille des poids

Graphique original enregistré par la campagne.

Source : Outputs du notebook Colab F/G.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

À expliquer : La latence utilise trente inférences unitaires après échauffement. Le poids des fichiers n’est pas un pic de RAM.

## 29. Ce que les résultats permettent de dire

Ce que les résultats permettent de dire

La prochaine confirmation doit porter sur de nouveaux exemples annotés indépendamment.

Source : Synthèse du notebook final.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

## 30. Trois livrables, une même provenance

Trois livrables, une même provenance

Le code, les résultats et les limites restent reliés aux sorties enregistrées.

Source : Dépôt GitHub OmarElkhali/DARIJA-MACHA3IR.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.

À expliquer : github.com/OmarElkhali/DARIJA-MACHA3IR

## 31. Les références

Les références

Les résultats publiés par leurs auteurs ne sont pas les scores mesurés dans cette campagne.

Source : research/papers.md et sources primaires.
Les métriques sont celles affichées dans les notebooks, sans exécution ni recalcul d’entraînement.