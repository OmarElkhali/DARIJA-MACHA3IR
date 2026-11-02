# Questions officielles Q1 à Q12

Énoncés fournis par l'utilisateur dans la conversation. Typographie normalisée : parenthèses et tirets remplacent les caractères défectueux du collage. « val » correspond à la partition `dev` du projet.

## Q1

Distribution des labels dans train, val et test, globalement et par écriture (arabe / latine). Scores du classifieur trivial « classe majoritaire ». Quelle métrique retenez-vous, et pourquoi ?

## Q2

Raccourcis : la distribution des labels dépend-elle de l'écriture ? Que vaut la règle fondée sur l'écriture ? Qu'en concluez-vous sur la validité de vos scores ?

## Q3

Quels choix de prétraitement avez-vous faits par famille de modèles ? Lesquels détruisent des indices utiles (emojis, lettres répétées, mots français, arabizi) ? Chiffres à l'appui (ablation).

## Q4

Zero-shot (B1) : score obtenu et sensibilité à la formulation des hypothèses. Pourquoi un modèle NLI peut-il classer sans exemple d'entraînement ? Quel rôle joue l'attention entre prémisse et hypothèse ?

## Q5

Modèle de sentiment B2 : scores sur les tweets arabes et latins séparément. Pourquoi cet écart ? Son score est-il exempt de contamination ?

## Q6

Déséquilibre : comparez pondération de la perte, sur-échantillonnage et optimisation du seuil (tableau). Pourquoi le sur-échantillonnage doit-il être fait après la séparation train / val ?

## Q7

C1 vs C2 vs C3 (et C4) : résultats. Les poids d'attention se concentrent-ils sur les emojis, les mots d'opinion et les mots français ? Effet de l'encodage positionnel ?

## Q8

PLM gelé (D) vs fine-tuné (E1) : gain et coût (temps, paramètres entraînés). SI2M-Lab/DarijaBERT vs SI2M-Lab/DarijaBERT-arabizi : lequel, et pourquoi ?

## Q9

Multi-tâche (E2) : l'ajout de l'algérien améliore-t-il la tâche principale ? Effet de λ. Pourquoi une tâche auxiliaire peut-elle aider… ou nuire ?

## Q10

Décalage de prior (F1) : comparez la distribution des labels de val et de test. Écart entre le seuil (ou la règle de décision) choisi sur val et l'optimum a posteriori sur le test. Que mettriez-vous en place en production (recalibration, monitoring) ?

## Q11

Écriture non vue (F2) : chute mesurée. Quel modèle déployer pour des commentaires marocains mêlant les deux écritures, et pourquoi ?

## Q12

Analyse d'erreurs (F6) : catégories dominantes, part du bruit d'annotation. Quel modèle mettriez-vous en production, et pourquoi (qualité, robustesse, coût, latence — chiffres à l'appui) ?

## Consignes de périmètre à conserver dans les réponses

- Le projet reste exclusivement marocain `ary`, conformément à la demande explicite de l'utilisateur. Q9 sera discutée théoriquement, avec « expérience exclue du périmètre » et aucun résultat multi-tâche inventé.
- La campagne E complète n'a pas été exécutée. Q8 doit distinguer le protocole et les coûts estimés des gains non mesurés.
- Les catégories F6 restent provisoires. Un désaccord entre annotateurs ne prouve pas une erreur du label officiel.
- Toute analyse a posteriori demandée en Q10 sera identifiée comme diagnostic sur un test déjà observé. Elle ne modifie pas la décision ni le candidat verrouillés avant test.
