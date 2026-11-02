# Préparation à l'oral — Groupe 12

**Les énoncés officiels Q1 à Q12 ne sont pas fournis. Les douze thèmes ci-dessous sont provisoires et ne prétendent pas reproduire les questions officielles. Fournir le sujet permettra la correspondance exacte.**

## Présentation personnelle proposée

Dans ce projet, j'étudie trois sentiments sur des tweets marocains en écritures arabe, latine et mixte. Je compare cinq familles, du TF-IDF au fine-tuning. Je sépare apprentissage, validation et test ; je distingue les résultats réellement calculés des expériences encore non exécutées. Le projet utilise uniquement ary, conformément à la consigne finale.

## Q1 provisoire — Données et protocole

**Explication simple :** Une partition est un rôle expérimental : train apprend, dev choisit, test évalue après verrouillage.

**Résultats réellement disponibles :** Train=5583, dev=494, test textes=2961.

**Tableau ou figure :** `results/audit/counts.csv`.

**Problème :** Fuites entre partitions et doublons.

**Solution implémentée ou testée :** Partitions officielles inchangées, audit texte et empreintes SHA-256.

**Conclusion scientifique :** Un audit local ne garantit pas l'absence de contamination externe.

**Référence :** https://arxiv.org/abs/2302.08956

## Q2 provisoire — Écritures et nettoyage

**Explication simple :** Latin, arabe et mixte décrivent l'alphabet, pas la langue exacte.

**Résultats réellement disponibles :** {'arabic': 3265, 'latin': 2276, 'mixed': 41, 'other': 1}

**Tableau ou figure :** `figures/scripts.png`.

**Problème :** Arabizi et français peuvent se ressembler.

**Solution implémentée ou testée :** Conserver chiffres, accents, négations et emojis ; signaler les heuristiques.

**Conclusion scientifique :** Une annotation humaine est nécessaire pour identifier le code-switching français.

**Référence :** https://aclanthology.org/2020.coling-main.51/

## Q3 provisoire — TF-IDF et n-grammes

**Explication simple :** TF-IDF donne plus de poids aux fragments discriminants ; les caractères partagent des fragments entre graphies.

**Résultats réellement disponibles :** {'weighted_f1': {'char': 0.7683, 'majority': 0.2117, 'union': 0.7458, 'union_train_arabic': 0.6586, 'union_train_latin': 0.3641, 'word': 0.7189}}

**Tableau ou figure :** `results.csv : A / dev`.

**Problème :** Orthographe variable.

**Solution implémentée ou testée :** Comparer mots, caractères et combinaison avec régression logistique pondérée.

**Conclusion scientifique :** Un écart dev ne prouve pas une amélioration générale.

**Référence :** https://doi.org/10.1016/j.jksuci.2023.101570

## Q4 provisoire — Zero-shot

**Explication simple :** Le NLI compare le tweet à trois hypothèses ; CAMeLBERT connaît déjà une tâche sentiment externe.

**Résultats réellement disponibles :** [{'variant': 'nli_ar', 'status': 'exécuté'}, {'variant': 'nli_fr', 'status': 'exécuté'}, {'variant': 'nli_en', 'status': 'exécuté'}, {'variant': 'camelbert', 'status': 'exécuté'}]

**Tableau ou figure :** `results.csv : B / dev`.

**Problème :** Effet de la langue de l'hypothèse et contamination possible.

**Solution implémentée ou testée :** Comparer arabe, français, anglais ; scores par écriture.

**Conclusion scientifique :** Sans run B, aucune performance B ne peut être annoncée.

**Référence :** https://aclanthology.org/2023.semeval-1.255/

## Q5 provisoire — Embeddings et OOV

**Explication simple :** Un embedding représente un mot par des nombres ; OOV signifie absent du vocabulaire appris sur train.

**Résultats réellement disponibles :** Voir oov.csv et embedding_coverage.json ; absence de fichier = non mesuré.

**Tableau ou figure :** `results/oov.csv`.

**Problème :** Mots rares, formes inédites.

**Solution implémentée ou testée :** Comparer hasard, AraVec et fastText entraîné exclusivement sur train.

**Conclusion scientifique :** Le vocabulaire min_freq=2 garde UNK pour les tokens hors vocabulaire, même si fastText sait composer un vecteur ; cette version compare l'initialisation des mots du vocabulaire.

**Référence :** https://aclanthology.org/Q17-1010/

## Q6 provisoire — BiLSTM et attention

**Explication simple :** BiLSTM lit les deux directions ; l'attention apprend une pondération des états.

**Résultats réellement disponibles :** [{'variant': 'attention_random', 'status': 'exécuté'}, {'variant': 'attention_fasttext', 'status': 'exécuté'}, {'variant': 'attention_aravec', 'status': 'exécuté'}, {'variant': 'attention_random_weighted', 'status': 'exécuté'}, {'variant': 'attention_random_oversampling', 'status': 'exécuté'}]

**Tableau ou figure :** `figures/attention_C.png`.

**Problème :** Padding et interprétation abusive des poids.

**Solution implémentée ou testée :** Masquer le padding et visualiser cinq vrais exemples dev.

**Conclusion scientifique :** Les poids d'attention sont descriptifs, pas une preuve causale.

**Référence :** https://doi.org/10.1016/j.jksuci.2023.101570

## Q7 provisoire — Mini-Transformer et CNN

**Explication simple :** Le Transformer met les mots en relation ; le CNN détecte des motifs locaux de caractères.

**Résultats réellement disponibles :** [{'variant': 'transformer_random', 'status': 'exécuté'}, {'variant': 'transformer_fasttext', 'status': 'exécuté'}, {'variant': 'transformer_aravec', 'status': 'exécuté'}, {'variant': 'transformer_random_nopos', 'status': 'exécuté'}, {'variant': 'transformer_fasttext_nopos', 'status': 'exécuté'}, {'variant': 'transformer_aravec_nopos', 'status': 'exécuté'}, {'variant': 'charcnn_random', 'status': 'exécuté'}]

**Tableau ou figure :** `figures/curves_C.png`.

**Problème :** Perte d'ordre sans positions et bruit orthographique.

**Solution implémentée ou testée :** Deux couches, dimension 128, quatre têtes, ablation des positions et char-CNN.

**Conclusion scientifique :** La taille du corpus peut favoriser les modèles simples.

**Référence :** https://arxiv.org/abs/1706.03762

## Q8 provisoire — Encodeur gelé

**Explication simple :** On extrait des représentations sans changer les paramètres BERT ; seule la tête apprend.

**Résultats réellement disponibles :** [{'variant': 'cls', 'status': 'exécuté'}, {'variant': 'mean', 'status': 'exécuté'}, {'variant': 'states', 'status': 'exécuté'}]

**Tableau ou figure :** `results.csv : D / dev`.

**Problème :** Coût d'extraction et mémoire des états cachés.

**Solution implémentée ou testée :** Comparer CLS, moyenne masquée et BiLSTM-attention ; caches identifiés.

**Conclusion scientifique :** Le nombre de paramètres entraînables n'est pas le nombre total de paramètres utilisés.

**Référence :** https://link.springer.com/article/10.1007/s41060-023-00498-2

## Q9 provisoire — Fine-tuning sur ary

**Explication simple :** Le fine-tuning ajuste aussi l'encodeur aux labels marocains.

**Résultats réellement disponibles :** [{'variant': 'darijabert', 'status': 'non exécuté (0/3 graines)'}, {'variant': 'arabizi', 'status': 'non exécuté (0/3 graines)'}]

**Tableau ou figure :** `results.csv : E / dev`.

**Problème :** Coût CPU et équité entre checkpoints.

**Solution implémentée ou testée :** AdamW 2e-5, batch effectif 32, 3 époques, warmup 10 %, mêmes graines ; aucun corpus d'un autre pays.

**Conclusion scientifique :** Un test de forme ne remplace pas un entraînement complet.

**Référence :** https://link.springer.com/article/10.1007/s41060-023-00498-2

## Q10 provisoire — Déséquilibre et décision

**Explication simple :** Une règle de décision transforme les scores en classe ; ses réglages appartiennent à dev.

**Résultats réellement disponibles :** Décalages réels dans models/*/meta.json ; ablations dans research_solutions.csv.

**Tableau ou figure :** `results/research_solutions.csv`.

**Problème :** Optimiser le Weighted-F1 peut masquer une classe faible.

**Solution implémentée ou testée :** Rapporter Macro-F1, perte pondérée, oversampling train et deux décalages relatifs dev.

**Conclusion scientifique :** Les scores bruts ne sont pas des probabilités calibrées ; le test ne règle pas les décalages.

**Référence :** https://doi.org/10.1016/j.jksuci.2023.101570

## Q11 provisoire — Robustesse et incertitude

**Explication simple :** Bootstrap mesure la variabilité d'échantillonnage ; trois graines mesurent celle de l'entraînement.

**Résultats réellement disponibles :** Résultats disponibles dans comparison.csv, f2_cross_script.csv, f4_perturbations.csv et f5_mcnemar.json ; fichier absent = non exécuté.

**Tableau ou figure :** `results/comparison.csv`.

**Problème :** Changement de distribution, emojis et erreurs humaines.

**Solution implémentée ou testée :** F1–F6, 1 000 rééchantillonnages, IC 95 %, McNemar exact et 30 erreurs à valider.

**Conclusion scientifique :** McNemar porte sur les taux d'erreur appariés ; un IC bootstrap conditionne sur le modèle entraîné.

**Référence :** https://aclanthology.org/2023.semeval-1.68/

## Q12 provisoire — Choix final et limites

**Explication simple :** Le candidat scientifique est choisi sur dev avant la lecture des métriques test.

**Résultats réellement disponibles :** [{'family': 'A', 'variant': 'char', 'dev_weighted_f1': 0.7682820176468543, 'dev_macro_f1': 0.7651389835258441, 'seed_std': 0.0, 'seeds': 3}]

**Tableau ou figure :** `results/selection_lock.json et figures/modele_final.png`.

**Problème :** Choisir après avoir vu test surestime la qualité.

**Solution implémentée ou testée :** Verrouiller variantes, règles et empreintes des checkpoints ; comparer cinq familles en laissant vides les familles non exécutées.

**Conclusion scientifique :** Le meilleur parmi les runs disponibles n'est pas le meilleur de toutes les méthodes possibles.

**Référence :** https://aclanthology.org/2023.semeval-1.315/

## Questions probables du professeur

- **Pourquoi ne pas mélanger train et test ?** Le test doit représenter des exemples non utilisés pour apprendre ou choisir.
- **Pourquoi Macro-F1 et Weighted-F1 ?** Le premier traite chaque classe également ; le second pondère par les effectifs réels.
- **Pourquoi trois graines identiques en TF-IDF ?** Le solveur peut être déterministe ; les réplications identiques ne constituent pas trois jeux de données indépendants.
- **Pourquoi pas de transfert algérien ?** La consigne finale demande exclusivement ary.
- **Une attention forte explique-t-elle la décision ?** Elle décrit le mécanisme interne ; une intervention contrôlée serait nécessaire pour une conclusion causale.
- **Pourquoi un score manque-t-il ?** Le run n'a pas été exécuté ; on ne le remplace jamais par une estimation présentée comme mesurée.
