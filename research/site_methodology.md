# Darija-Sentiment : dossier méthodologique pour le site

Ce document décrit le code de la campagne `campaign_v4`. Il ne contient aucun nouveau score de modèle : les scores de la présentation doivent être importés depuis les sorties Colab de cette campagne, puis reliés à leur fichier d’origine. Les anciens fichiers `results/` à la racine et les textes historiques ne sont pas une source interchangeable avec `experiments/campaign_v4/results/`.

## 1. Question de recherche

Comment reconnaître un sentiment négatif, neutre ou positif dans des tweets marocains écrits en arabe, en caractères latins ou dans un mélange des deux ? Le projet compare des représentations de complexité croissante : n-grammes TF-IDF, transfert de modèles publics, réseaux compacts, encodeur dialectal gelé et fine-tuning. Une sixième famille P étudie l’effet d’un nettoyage retirant emojis et ponctuation.

Les objectifs se distinguent : mesurer la qualité globale, examiner la généralisation entre écritures et caractériser les coûts d’inférence. Le modèle retenu pour la démonstration doit être celui sélectionné sur validation dans le verrou de la campagne. Un modèle donnant le meilleur score sur test ne remplace pas ce choix a posteriori.

## 2. Corpus et traçabilité

Le corpus principal est la partition marocaine `ary` d’AfriSenti-SemEval 2023. Les fichiers sont figés à la révision Git `5aec3cfcf87b59f885e98641f72e52182776429d` et contrôlés par SHA-256. Ils restent inchangés ; les tweets transformés sont calculés séparément. L’ordre des classes dans les probabilités est `negative`, `neutral`, `positive`.

Effectifs vérifiés directement dans les trois TSV locaux de cette révision :

| Partition | Négatif | Neutre | Positif | Total |
|---|---:|---:|---:|---:|
| Train | 1 664 | 2 161 | 1 758 | 5 583 |
| Validation (`dev`) | 138 | 189 | 167 | 494 |
| Test | 850 | 967 | 1 144 | 2 961 |

La distribution doit aussi être examinée par écriture : une bonne performance globale peut masquer un échec sur une sous-population. Le détecteur reconnaît les alphabets, pas les langues. Une phrase latine peut être en Arabizi, en français ou contenir les deux ; la présence de caractères arabes et latins donne la catégorie mixte. Mentions et URL sont ignorées pour cette détection afin d’éviter de créer artificiellement du latin.

L’audit distingue doublons bruts, doublons après nettoyage et collisions entre partitions. Les contradictions impliquant les labels test ne sont analysées qu’après verrouillage. Une analyse supplémentaire par similarité de n-grammes caractères repère des quasi-doublons ; elle ne modifie pas les partitions officielles.

**Sources locales :** `src/preprocess.py:18-29`, `src/preprocess.py:117-169`, `src/validation.py:12-43` ; effectifs comptés dans `data/ary/train.tsv`, `dev.tsv` et `test.tsv`.

## 3. Protocole expérimental

1. **Train apprend** : vocabulaire, TF-IDF, paramètres des classifieurs, embeddings fastText et poids des réseaux.
2. **Validation choisit** : variante, époque retenue et décalages de la règle de décision.
3. **Le verrou enregistre les choix** : candidat, représentants par famille, graines, empreintes des artefacts et liste des runs évalués.
4. **Test mesure** : les choix restent ceux fixés sur validation, même lorsqu’une autre variante obtient un meilleur score test.

Les réseaux entraînés utilisent les graines 42, 43 et 44. Le représentant est toujours le checkpoint de graine 42, jamais la meilleure graine sur test. Les méthodes déterministes A, B et D avec régression logistique ne sont exécutées qu’une fois. Les variantes réduites E2 et P utilisent également une seule graine, mais pour réduire leur budget : elles ne permettent donc pas d’estimer une variabilité sur trois entraînements.

Le test marocain avait déjà été consulté dans l’étude historique. La nouvelle campagne constitue ainsi une **réévaluation exploratoire**, même si son code bloque le nouvel accès aux labels test jusqu’au verrouillage. Une confirmation indépendante demande de nouveaux exemples annotés et un protocole de collecte distinct.

La configuration sauvegardée prévoit 128 tokens pour les réseaux à mots et BERT, 512 caractères pour le char-CNN, au plus 15 époques C, 10 époques D et 5 époques E1. Les manifestes et les historiques réels font foi pour connaître le nombre d’époques effectivement exécutées.

**Sources locales :** `src/campaign.py:25-78`, `src/train.py:28-38`, `src/train.py:57-81`, `src/evaluate.py:98-145`, `experiments/campaign_v4/campaign.json`.

## 4. Nettoyage : préserver les indices de sentiment

Le nettoyage `conservative_v2` normalise Unicode en NFC, retire URL et mentions, puis harmonise les espaces. Il conserve négation, accents, mots français, chiffres Arabizi, emojis et répétitions de lettres. La tokenisation des modèles à mots sépare les unités alphanumériques et les signes ; elle abaisse la casse latine. Les modèles BERT emploient leur propre tokenizer.

Trois transformations diagnostiques sont définies : retrait des emojis et émoticônes seuls, retrait de la ponctuation seule en préservant les séquences d’emojis reconnues, et retrait des deux. Elles ne sont pas supposées préserver le sentiment : « j’aime ça 🙂 » et « j’aime ça 🙄 » peuvent porter des intentions différentes.

La campagne réduite P réentraîne un TF-IDF union et, à l’origine, un DarijaBERT sur le texte sans emojis ni ponctuation. Elle ne réalise pas des ablations isolées de la négation, du français, des chiffres Arabizi ou des répétitions. Le site ne doit leur attribuer aucun effet chiffré non mesuré.

**Sources locales :** `src/preprocess.py:63-114`, `src/preprocess.py:172-185`, `scripts/reduce_remaining.py:31-39`.

## 5. Les six familles

### A — TF-IDF et régression logistique

Les baselines utilisent des n-grammes de mots 1–2, des n-grammes de caractères 2–5 ou leur union. Les vectoriseurs utilisent `min_df=2`, une fréquence sous-linéaire et respectivement 50 000 et 100 000 caractéristiques maximales. La régression logistique est pondérée par classe, avec au plus 2 000 itérations. L’union compare `C=0,25`, `1` et `4` sur validation. Une baseline « classe majoritaire » donne un point de comparaison trivial.

L’intérêt des caractères est de représenter des fragments partagés entre graphies variables sans exiger qu’un mot complet ait été vu. Cette intuition ne prouve pas un gain : seuls les résultats de la campagne le montrent. Les baselines apprennent leur vocabulaire uniquement sur train et conservent le texte complet, sans troncature à 128 tokens.

**Sources :** `src/models.py:40-51`, `src/train.py:90-112`, `src/campaign.py:126-140`.

### B — Transfert de modèles publics fixes

**B1, NLI :** `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`. Chaque tweet est la prémisse de trois paires, une avec une hypothèse négative, une neutre et une positive. Le code utilise les logits d’entailment et un softmax entre les trois sentiments. Trois formulations sont comparées, en arabe, français et anglais. La comparaison change donc à la fois la langue et la formulation ; elle n’isole pas la sensibilité à des paraphrases d’une même langue.

**B2, sentiment :** `CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment`. Le code vérifie la correspondance des labels et réordonne les probabilités dans l’ordre commun. Les poids restent fixes : aucun apprentissage local sur train ary. Les entrées sont limitées à 128 tokens.

« Sans entraînement local sur ary » ne signifie pas « sans supervision antérieure ». Le modèle de sentiment a déjà reçu une supervision sur d’autres corpus. Ni la carte du modèle ni l’audit des seules partitions ary ne permettent de certifier l’absence de contamination externe.

**Sources :** `src/models.py:20-37`, `src/models.py:285-321`, `src/train.py:290-302` ; références publiques listées dans `research/papers.md`.

### C — Réseaux compacts appris sur ary

Toutes les représentations de mots ont une dimension de 128. Le vocabulaire est construit sur train avec une fréquence minimale de 2 ; PAD vaut 0 et UNK vaut 1.

| Architecture | Définition implémentée |
|---|---|
| BiLSTM | Une couche bidirectionnelle, 64 unités par direction ; moyenne masquée des états |
| BiLSTM-attention | Même encodeur, attention apprise : projection 128→64, tanh, score scalaire et softmax masqué |
| Mini-Transformer | Deux couches, dimension 128, quatre têtes, feed-forward 256, dropout 0,1 ; moyenne masquée |
| Ablation des positions | Même Mini-Transformer, avec ou sans encodage positionnel sinusoïdal |
| Char-CNN | Embeddings caractères de dimension 128 ; 64 filtres pour chacune des largeurs 3, 5 et 7 ; ReLU et max-pooling masqué |

La tête des réseaux C utilise un dropout de 0,2 puis une projection vers trois classes. L’optimiseur commun est AdamW, taux `10⁻³`, weight decay 0,01, batch effectif 32, gradient clipping à 1 et patience de 3 époques. Le budget est d’au plus 15 époques. L’époque retenue maximise le Weighted-F1 brut de validation ; les décalages de décision sont réglés ensuite.

Les variantes à mots comparent embeddings aléatoires, AraVec et fastText. Le char-CNN utilise un vocabulaire de caractères et ne reçoit pas AraVec ou fastText.

**AraVec :** les vecteurs officiels sont projetés à la dimension 128 au moyen d’une matrice aléatoire fixée par la graine ; les mots absents gardent une initialisation aléatoire.

**fastText :** entraîné strictement sur train ary avec dimension 128, fenêtre 5, `min_count=2`, skip-gram, 10 époques, un worker et 20 000 buckets. Dans la campagne actuelle, les mots hors du vocabulaire du réseau peuvent recevoir un vecteur composé par sous-mots fastText, sans ajouter de mots au vocabulaire appris. Il faut distinguer OOV lexical, présence d’un vecteur utilisable et OOV du vocabulaire C.

Le meilleur couple architecture–embeddings est choisi par sa moyenne sur les trois graines. Ce même couple est ensuite comparé sans rééquilibrage, avec perte pondérée et avec suréchantillonnage. Le suréchantillonnage utilise uniquement train, après séparation des partitions, pour éviter des copies circulant vers validation ou test.

**Sources :** `src/models.py:54-111`, `src/models.py:114-189`, `src/train.py:128-155`, `src/train.py:228-284`, `src/campaign.py:145-170`.

### D — DarijaBERT gelé

L’encodeur `SI2M-Lab/DarijaBERT` ne reçoit aucun gradient. Trois représentations sont comparées : CLS suivi d’une régression logistique, moyenne masquée suivie d’une régression logistique, et états de tokens suivis d’un BiLSTM-attention. La moyenne inclut les tokens spéciaux et exclut le padding.

Les têtes logistiques sont pondérées et utilisent au plus 1 000 itérations. La tête séquentielle comprend un BiLSTM de 64 unités par direction, une attention et une projection vers trois classes ; AdamW à `10⁻³`, micro-batch 8, batch effectif 32, patience 3, au plus 10 époques.

Les représentations extraites sont mises en cache pour réutiliser l’encodeur entre têtes et graines. Leur coût partagé est enregistré séparément. Un faible nombre de paramètres entraînables ne signifie pas une inférence légère : le texte doit toujours passer par DarijaBERT. La latence CPU de G est mesurée sans réutiliser le cache de représentations.

**Sources :** `src/models.py:192-208`, `src/models.py:233-282`, `src/train.py:391-421`, `src/evaluate.py:307-338`.

### E — Fine-tuning dialectal et multi-tâche

**E1 :** DarijaBERT et DarijaBERT-Arabizi sont adaptés à ary, chacun avec les graines 42, 43 et 44. Le modèle utilise l’état CLS, un dropout de 0,1 et une tête linéaire à trois sorties. L’ensemble des poids entraînables est ajusté avec AdamW à `2×10⁻⁵`, weight decay 0,01, perte pondérée d’après train, batch effectif 32, micro-batch 2 et accumulation des gradients. Le warmup représente 10 % des pas. Le budget maximal est de 5 époques, avec au moins 3 époques avant arrêt anticipé et une patience de 2. La précision mixte CUDA et le checkpointing des gradients limitent la mémoire.

**E2 réduit :** l’encodeur est d’abord choisi sur E1/validation. Une seconde tête prédit le sentiment du train algérien `arq`, avec un encodeur partagé. La perte vaut `L_ary + λ L_arq`. Le train auxiliaire est cyclé afin que son poids par pas corresponde au λ déclaré. Les collisions avec les textes ary et les doublons/contradictions auxiliaires sont filtrés. Aucun test arq n’est utilisé.

Le plan réellement retenu remplace la grille initiale 0,1/0,3/0,7 par deux runs, λ=0,1 et λ=0,3, graine 42, trois époques maximales. Les règles de choix restent fondées sur ary/dev. La comparaison à E1 est exploratoire : budget plus court et une seule graine pour E2. Un éventuel gain ou recul ne doit pas être présenté comme une conclusion générale sur l’utilité de l’algérien.

**Sources :** `src/models.py:211-230`, `src/train.py:424-465`, `src/train.py:128-153`, `src/train.py:183-224`, `src/campaign.py:81-115`, `scripts/reduce_remaining.py:31-90`.

### P — Ablation du nettoyage

P compare un TF-IDF union sans emojis ni ponctuation à la référence A_union conservatrice. Un second run DarijaBERT devait être comparé à E_darijabert, avec une graine 42 et trois époques maximales.

Le fichier de poids de `P_darijabert_without_both_s42` est absent de la sauvegarde partagée. À la demande de l’utilisateur, ce run a été **exclu des évaluations F/G**. Ses résultats dev existants sont conservés. Il ne reçoit aucun score test inventé et ne participe pas à la sélection finale. L’exclusion figure dans `results/evaluation_exclusions.json` et dans le verrou de sélection. P reste représenté par son modèle TF-IDF union disponible ; les « six méthodes » du laboratoire correspondent aux six familles, et non à six réseaux BERT.

**Sources :** `scripts/reduce_remaining.py:31-39`, `scripts/skip_missing_p.py:6-48`.

## 6. Décision multiclasse et métriques

La règle brute choisit la classe de probabilité maximale. La règle ajustée choisit `argmax_c(log(p_c) + b_c)`. Le premier décalage est fixé à zéro ; les deux autres sont choisis sur validation parmi −0,6, −0,3, 0, 0,3 et 0,6, soit 25 couples. Le meilleur décalage est celui qui améliore le Weighted-F1 de validation. Ce réglage n’est pas une calibration probabiliste et les scores affichés dans le laboratoire ne doivent pas être décrits comme des probabilités de certitude vérifiées.

Le **Weighted-F1** est la moyenne des F1 par classe pondérée par leurs effectifs. C’est le critère principal de comparaison du benchmark ; il peut sous-représenter un échec sur une classe rare. Le **Macro-F1**, moyenne non pondérée, lui donne un poids égal. Précision et rappel pondérés complètent le tableau. Une mesure par écriture explique des disparités que le score global masque.

La moyenne et l’écart-type sur les graines décrivent la variabilité d’entraînement. L’intervalle bootstrap décrit une autre incertitude : le code rééchantillonne les tweets 1 000 fois, graine 42, et conserve les quantiles 2,5 % et 97,5 % du Weighted-F1, conditionnellement au checkpoint représentatif. Le test exact de McNemar compare les erreurs appariées de deux modèles choisis sur validation ; ce n’est pas un test direct d’écart de F1.

**Sources :** `src/evaluate.py:18-46`, `src/evaluate.py:98-111`, `src/evaluate.py:164-182`.

## 7. Ce que mesurent F et G

| Partie | Question | Preuves à afficher | Limite d’interprétation |
|---|---|---|---|
| F1 | La relation labels–écriture change-t-elle entre dev et test ? | Distributions globales/conditionnelles, courbes et grille des décalages | Une variation de proportions ne prouve pas un pur label shift |
| F1 post hoc | Quelle règle gagnerait sur une grille test ? | Oracle fini 21×21 sur [−1,5 ; 1,5] et règle dev figée | Diagnostic optimiste ; ne devient jamais la règle déployée |
| F2 | Que se passe-t-il sur une écriture absente du train ? | Union entraînée arabe→test latin et inverse, référence train complet | Taille et couverture du train changent ensemble ; l’alphabet n’est pas isolé causalement |
| F3 | Comment les scores et embeddings couvrent-ils les écritures ? | F1 par arabe/latin/mixte, OOV réseau/AraVec/fastText | Alphabet n’est pas langue ; faible effectif mixte possible |
| F4 | Que changent les emojis et la ponctuation ? | Trois transformations, baisse de F1, taux de bascule sur textes modifiés | Une transformation peut changer le vrai sentiment ; labels humains nécessaires |
| F5 | Les écarts sont-ils stables ? | Graines, bootstrap 95 %, McNemar | Variabilité d’entraînement et d’échantillonnage sont différentes |
| F6 | Où le candidat se trompe-t-il ? | Dossier diversifié de 30 erreurs, catégories et validation humaine | Suggestions heuristiques ; ne pas extrapoler leurs fréquences à toutes les erreurs |
| F7 humain | Généralise-t-il à des cas linguistiques et à une collecte indépendante ? | Exemples annotés, source, date, relecteur, audit de proximité | Aucun score humain sans validation ; aucun ensemble indépendant si collecte absente |
| G | Quel compromis qualité–coût retenir ? | Tableau final, confusion, latence CPU, poids, paramètres, apprentissage | Le candidat reste celui fixé sur validation ; le matériel influence les coûts |

La latence CPU utilise 30 inférences unitaires après chargement et échauffement. Elle décrit un débit unitaire sur le matériel de mesure, pas un temps de réponse garanti en production. La taille des poids inclut les ressources nécessaires à l’inférence, dont les encodeurs externes et fastText éventuels ; elle ne mesure pas la RAM maximale.

Les temps cumulés dédoublonnent les lignes brut/ajusté et les reprises. Ils additionnent les durées d’époques sauvegardées et la dernière session. Les époques perdues avant checkpoint, les interruptions réseau et les périodes hors exécution ne sont pas mesurables rétrospectivement.

**Sources :** `src/campaign.py:321-399`, `src/evaluate.py:189-249`, `src/evaluate.py:251-386`, `src/validation.py:62-158`, `scripts/continue_campaign.py:94-140`.

## 8. Interpréter le laboratoire des six familles

Pour chaque famille, le laboratoire doit charger le représentant choisi sur validation et afficher le nom exact du run. Le texte passe par le nettoyage enregistré dans ses métadonnées ; P doit donc appliquer sa transformation sans emojis ni ponctuation. Les trois probabilités du modèle et la décision ajustée utilisent l’ordre commun des labels et les décalages sauvegardés. Une divergence entre probabilités brutes et classe retenue est possible lorsque les décalages changent l’argmax : l’interface doit l’expliquer.

Une saisie de démonstration n’a pas de label humain établi. Les modèles proposent des prédictions ; leur accord n’est pas une preuve de vérité, et leur désaccord constitue une observation utile. Les phrases d’exemple doivent être explicitement pédagogiques, sans être ajoutées au train ni comptées comme une évaluation indépendante. Une méthode dont le poids manque doit apparaître indisponible ; une règle lexicale de secours ne doit jamais être présentée comme sa vraie prédiction.

**Sources :** `src/train.py:468-527`, `src/evaluate.py:25-35`, métadonnées des checkpoints sélectionnés.

## 9. Limites à rendre visibles

- Test ary déjà observé historiquement : étude exploratoire, pas confirmation indépendante.
- Validation de seulement 494 tweets : choix de variantes et décalages peuvent être sensibles à cet échantillon.
- Trois graines ne couvrent pas toutes les sources de variabilité ; E2/P réduit n’a qu’une graine.
- Contamination externe des modèles préentraînés non certifiable avec les seules données locales.
- Détecteur d’alphabet heuristique, non détecteur linguistique ; sarcasme et changement de cible restent difficiles.
- Troncature à 128 tokens ou 512 caractères susceptible de supprimer de l’information.
- Retirer des marqueurs peut modifier le sentiment ; les métriques F4 sur labels d’origine sont diagnostiques.
- Revue humaine inachevée : aucune « part de bruit d’annotation » établie à partir des suggestions automatiques.
- Exclusion documentée du P DarijaBERT sans poids ; aucun résultat test ne lui est attribué.
- Coûts matériels et latences ne se transposent pas directement à un autre PC ou service.

## 10. Fil conducteur pour l’oral Q1–Q12

Pour chaque réponse : **observation → mesure réelle → interprétation → limite**.

1. **Q1 :** distributions, majorité et choix Weighted-F1/Macro-F1.
2. **Q2 :** dépendance entre écriture et label, baseline fondée sur l’écriture, validité de généralisation.
3. **Q3 :** nettoyage conservateur, P et F4 ; signaler les ablations individuelles non effectuées.
4. **Q4 :** principe NLI, trois hypothèses, formulations arabe/français/anglais et scores mesurés.
5. **Q5 :** CAMeLBERT par écriture, adéquation de domaine et contamination non certifiée.
6. **Q6 :** trois stratégies de déséquilibre et séparation entre apprentissage et décision.
7. **Q7 :** architectures C, attention descriptive, ablation des positions et variabilité des graines.
8. **Q8 :** D gelé contre E1, deux encodeurs dialectaux, gains et coûts réellement mesurés.
9. **Q9 :** E2 réduit λ=0,1/0,3 contre E1 ; budget et nombre de graines différents.
10. **Q10 :** distributions dev/test, décision fixée et oracle post hoc séparés ; suivi et recalibration à valider sur de nouvelles données.
11. **Q11 :** transfert entre écritures F2, scores par écriture et compromis de déploiement.
12. **Q12 :** erreurs relues humainement, limites de l’annotation et choix qualité/coût/latence.

**Attention aux notes historiques :** la fin de `research/questions_officielles.md` indique encore que Q9 est exclue et que E n’a pas été exécutée. Ces mentions décrivent un ancien périmètre et contredisent la campagne actuelle. Ne pas les recopier dans le site ou le nouveau notebook. Le plan réduit, les manifestes et les sorties effectivement récupérées sont les références courantes.

## 11. Références scientifiques à relier dans le site

- AfriSenti : https://aclanthology.org/2023.emnlp-main.862/
- AfriSenti-SemEval : https://aclanthology.org/2023.semeval-1.315/
- DarijaBERT : https://doi.org/10.1007/s41060-023-00498-2
- fastText : https://aclanthology.org/Q17-1010/
- Transformer : https://arxiv.org/abs/1706.03762
- Char-CNN : https://arxiv.org/abs/1509.01626
- Données officielles : https://github.com/afrisenti-semeval/afrisent-semeval-2023

Ces liens proviennent du dossier documentaire local ; cette tâche de préparation de contenu n’a pas effectué de nouvelle vérification web. Les résultats publiés par leurs auteurs ne doivent pas être confondus avec ceux de cette campagne ary.
