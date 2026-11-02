# Validation humaine

`behavioral_cases.csv` contient des exemples pédagogiques construits, **non annotés**.
Les suggestions ne sont jamais utilisées comme labels. Faire vérifier les formulations
et remplir `label`, `reviewer`, `validation_humaine=validé`. Ajouter d'autres formulations
pour chaque catégorie. Ce fichier n'est pas un ensemble indépendant de tweets réels.

`independent.csv` est volontairement vide : collecter de nouveaux textes réels hors du
benchmark, renseigner source et date ISO de collecte, puis faire annoter sans montrer
les prédictions. Idéalement deux annotateurs et une adjudication des désaccords.
Respecter les conditions de collecte et retirer les identifiants personnels inutiles.
Ne pas utiliser cet ensemble pour régler les modèles. La proximité avec les partitions
officielles est contrôlée et bloque la publication d'un score en cas de collision.

Après verrouillage du candidat, `evaluate_human_sets` évalue seulement les lignes
validées avec un label negative/neutral/positive et un nom/code de relecteur.
Une source/date sont obligatoires pour l'ensemble indépendant. Aucun score n'est
produit tant qu'aucun exemple n'a été validé.

`results/f4_human_review.csv` est généré après F/G sur les textes perturbés. Renseigner
`sentiment_preserved` (oui/non/incertain), le sentiment du texte transformé dans
`human_label`, `reviewer` et `validation_humaine=validé`. Les relances conservent ces
colonnes si le texte d'origine et sa transformation sont identiques.
