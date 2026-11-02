# Problèmes et solutions inspirées de la recherche

Comparaisons descriptives sur dev ; ni test causal ni garantie de gain sur test. Les valeurs sont calculées à partir des runs disponibles.

| Problème | Observation mesurée | Publication | Solution proposée | Résultat après essai | Statut |
| --- | --- | --- | --- | --- | --- |
| Graphies variables | 5563 textes nettoyés distincts sur 5583 | https://aclanthology.org/Q17-1010/ | TF-IDF caractères | Dev : 0.7189 → 0.7683, delta=+0.0494 ; graines=3/3 | Testée avec amélioration |
| Mots rares / OOV | OOV mesurés dans le notebook avec vocabulaire train | https://aclanthology.org/Q17-1010/ | Embeddings fastText train | Dev : 0.7370 → 0.7476, delta=+0.0106 ; graines=3/3 | Testée avec amélioration |
| Déséquilibre | {'neutral': 2161, 'positive': 1758, 'negative': 1664} | https://doi.org/10.1016/j.jksuci.2023.101570 | Perte pondérée | Dev : 0.7370 → 0.7356, delta=-0.0015 ; graines=3/3 | Testée sans amélioration |
| Rééchantillonnage | Aucun exemple dev/test ajouté au train | https://doi.org/10.1016/j.jksuci.2023.101570 | Oversampling train uniquement | Dev : 0.7370 → 0.7294, delta=-0.0076 ; graines=3/3 | Testée sans amélioration |
| Ordre des mots | Architecture 128, 4 têtes, 2 couches | https://arxiv.org/abs/1706.03762 | Encodage positionnel | Dev : 0.7210 → 0.7068, delta=-0.0142 ; graines=3/3 | Testée sans amélioration |
| Adaptation Arabizi | {'arabic': 3265, 'latin': 2276, 'mixed': 41, 'other': 1} | https://link.springer.com/article/10.1007/s41060-023-00498-2 | Fine-tuning deux checkpoints | Non mesuré | Déjà implémentée ; non exécutée ou comparaison incomplète |
| Mélange arabe/français | Écriture mixte ≠ français identifié ; validation humaine nécessaire | https://doi.org/10.1016/j.jksuci.2023.101570 | Nettoyage conservateur | Implémenté, sans ablation causale de langue | Déjà implémentée |