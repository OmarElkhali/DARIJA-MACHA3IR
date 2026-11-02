# Réponses préparatoires aux douze questions officielles

Statut : dossier de contenu pour le plan de soutenance, avant approbation. Ce fichier n'est pas le script oral final. Les énoncés exacts sont conservés dans `../../research/questions_officielles.md`. « val » correspond à `dev`.

Les valeurs existantes proviennent des artefacts du benchmark verrouillé. Les compléments Q1/Q2, le diagnostic Q10 et le témoin Q11 ont été calculés séparément pour répondre aux questions. Ils sont conservés dans `evidence.draft.json`, sans modifier `results.csv`, `comparison.csv`, la sélection ni les checkpoints. Les nouveaux calculs sur le test sont rétrospectifs et ne créent pas un test indépendant.

## Q1 — Distributions, majorité et métrique

Effectifs des trois labels, dans l'ordre négatif / neutre / positif :

| Partition | Population | Négatif | Neutre | Positif | Total |
|---|---|---:|---:|---:|---:|
| Train | Ensemble | 1 664 | 2 161 | 1 758 | 5 583 |
| Val | Ensemble | 138 | 189 | 167 | 494 |
| Test | Ensemble | 850 | 967 | 1 144 | 2 961 |
| Train | Arabe | 1 137 | 954 | 1 174 | 3 265 |
| Val | Arabe | 103 | 75 | 99 | 277 |
| Test | Arabe | 308 | 494 | 565 | 1 367 |
| Train | Latin | 514 | 1 190 | 572 | 2 276 |
| Val | Latin | 34 | 112 | 67 | 213 |
| Test | Latin | 527 | 454 | 538 | 1 519 |

Le global inclut aussi les écritures mixtes et autres. Le regroupement par écriture est une heuristique d'alphabet, pas une annotation de langue.

La classe majoritaire est apprise sur train : **neutre**, conservée pour val et test. On ne choisit pas la majorité de chaque partition en lisant ses réponses.

| Partition | Accuracy | Weighted-F1 | Macro-F1 |
|---|---:|---:|---:|
| Train | 0,3871 | 0,2160 | 0,1860 |
| Val | 0,3826 | 0,2117 | 0,1845 |
| Test | 0,3266 | 0,1608 | 0,1641 |

Le critère principal est le **Weighted-F1**, conformément au protocole du projet. Il combine précision et rappel de chaque classe, pondérés par son effectif. On ajoute le **Macro-F1** pour donner le même poids aux trois sentiments et éviter de masquer une classe faible. Un F1 de 0,54 n'est pas une accuracy de 54 %.

Sources : partitions officielles, `results/f1_conditional_labels.csv`, compléments analytiques de `evidence.draft.json`. Le score majorité val correspond au run sauvegardé.

## Q2 — Raccourci d'écriture

Oui, la distribution dépend fortement de l'écriture. Sur train, le neutre représente **29,2 % des textes arabes**, contre **52,3 % des textes latins**. Sur val, ces parts sont respectivement **27,1 %** et **52,6 %**. Sur test, elles deviennent **36,1 %** et **29,9 %**.

La règle apprend sur train la classe majoritaire de chaque écriture : arabe = positif, latin = neutre, mixte = neutre. Le groupe « autre » contient un seul exemple train et ne permet aucune conclusion générale.

| Règle apprise sur train | Weighted-F1 val | Weighted-F1 test |
|---|---:|---:|
| Toujours neutre | 0,2117 | 0,1608 |
| Majorité selon l'écriture | 0,3656 | 0,2945 |

L'alphabet seul permet déjà un score supérieur à la majorité globale. Une partie des performances peut donc reposer sur des régularités de corpus. Cela n'invalide pas automatiquement toutes les prédictions, mais impose des analyses par écriture et une validation sur de nouvelles données. La règle script-only n'est pas proposée comme modèle de production.

Sources : `scripts/build_notebook.py`, calcul script-only reproduit dans le dossier de preuves, distributions officielles.

## Q3 — Prétraitements et ablation

Le nettoyage commun remplace URLs et mentions, harmonise les espaces et conserve chiffres Arabizi, négations, accents, répétitions, français et emojis. Il n'effectue pas de translittération, stemming ni suppression systématique de stopwords.

| Famille | Représentation et conséquences |
|---|---|
| A mots | TF-IDF mots 1–2, minuscules. Le tokenizer standard exclut la ponctuation, les emojis et les tokens d'un seul caractère. |
| A caractères | TF-IDF caractères 2–5, minuscules. Les motifs de ponctuation et d'emojis peuvent être représentés s'ils passent les filtres de fréquence. |
| B | Tokenizer officiel du checkpoint. Longueur maximale 128 sous-tokens, avec l'hypothèse dans le cas NLI. La conservation dans le texte n'assure pas une représentation adaptée par le tokenizer. |
| C à mots | Tokens regex en minuscules, ponctuation/emojis distincts, vocabulaire train min_freq=2, UNK, 64 tokens maximum. |
| C Character-CNN | Caractères, vocabulaire train min_freq=2, 256 caractères maximum. |
| D et E | Tokenizer officiel DarijaBERT, 64 sous-tokens maximum dans le protocole enregistré. |

Supprimer les négations, chiffres Arabizi ou mots français peut changer le sens. Réduire les lettres répétées peut enlever une intensité. Ces risques sont expliqués, mais leur effet individuel n'a pas été mesuré ici.

L'ablation F4 disponible retire **conjointement** les emojis et les séquences de ponctuation répétée, sans réentraînement, sur les tweets polarisés initialement bien classés et effectivement modifiés :

| Représentant | Exemples réellement modifiés | Décisions devenues incorrectes |
|---|---:|---:|
| A caractères | 102 | 9, soit 8,8 % |
| B CAMeLBERT | 89 | 11, soit 12,4 % |
| C attention-fastText | 107 | 24, soit 22,4 % |
| D états-attention | 134 | 39, soit 29,1 % |

Ces populations diffèrent entre modèles. Ce tableau ne permet ni d'isoler l'effet des emojis seuls ni d'établir un classement causal de robustesse. Aucune ablation individuelle de suppression du français, des chiffres Arabizi ou des lettres répétées n'a été exécutée. Cette partie de la question reste donc expérimentalement incomplète.

Sources : `src/preprocess.py`, `src/models.py`, `src/train.py`, `results/f4_*_perturbations.csv`, `results/comparison.csv`.

## Q4 — B1, formulation et attention entre les textes

Weighted-F1 sur **val**, sans apprentissage local des poids :

| Langue des hypothèses | Scores bruts, argmax | Décalages choisis sur val |
|---|---:|---:|
| Arabe | 0,3980 | 0,5010 |
| Français | 0,3980 | 0,4972 |
| Anglais | 0,3860 | 0,4634 |

L'écart brut maximal est d'environ 0,0120. L'écart après réglage vaut 0,0376 et inclut l'effet des règles de décision propres aux runs. Il ne faut pas l'attribuer uniquement à la formulation. Une seule formulation par langue a été testée : pas d'étude complète de paraphrases à langue constante. Les variantes NLI n'ont pas de score test principal, puisque CAMeLBERT a été retenu pour la famille B.

Le tweet sert de prémisse et chaque sentiment devient une hypothèse textuelle. Un modèle préalablement entraîné au NLI estime leur compatibilité. L'implémentation compare les logits d'entailment des trois hypothèses. L'encodeur traite la paire conjointement et son auto-attention permet des interactions entre tokens de la prémisse et de l'hypothèse. Cela aide à comparer leur sens, sans démontrer une interprétation causale des poids.

Sources : `results/results.csv`, `src/models.py`, [fiche du modèle NLI](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7), [Yin et al., 2019](https://arxiv.org/abs/1909.00161), [He et al., DeBERTa](https://arxiv.org/abs/2006.03654).

## Q5 — B2 par écriture et contamination

CAMeLBERT, représentant retenu, règle de décision fixée sur val, **test** :

| Écriture | Effectif | Weighted-F1 |
|---|---:|---:|
| Arabe | 1 367 | 0,5581 |
| Latin | 1 519 | 0,1442 |
| Mixte | 75 | 0,4923 |

L'écart arabe/latin est de **0,4139**, soit 41,39 points de F1. Le modèle spécialisé en arabe dialectal représente vraisemblablement moins bien l'Arabizi latin. Les différences de segmentation et de données constituent des explications possibles, non isolées causalement.

Son score ne peut pas être déclaré exempt de contamination : la fiche officielle mentionne un fine-tuning sur ASTD, ArSAS et SemEval. Un audit exhaustif des textes de préentraînement et des corpus externes n'a pas été effectué.

Sources : `results/B_camelbert_s42_test_tuned.csv`, `results/comparison.csv`, [fiche officielle CAMeLBERT](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment).

## Q6 — Pondération, suréchantillonnage et décision

Comparaison contrôlée sur l'architecture C attention à initialisation aléatoire. Moyenne **val** sur trois graines :

| Entraînement | Argmax brut | Décalages val | Gain descriptif du réglage |
|---|---:|---:|---:|
| Sans rééquilibrage | 0,7223 | 0,7370 | +0,0147 |
| Perte pondérée | 0,7273 | 0,7356 | +0,0083 |
| Suréchantillonnage train | 0,7241 | 0,7294 | +0,0054 |

La pondération et le suréchantillonnage améliorent légèrement l'argmax dans ces runs, mais ne dépassent pas la référence après réglage val. Les gains de décision sont observés sur la partition qui a servi à régler les décalages : ce ne sont pas des gains indépendants certifiés.

Il faut séparer train et val **avant** le suréchantillonnage. Sinon, une copie d'un même exemple peut se retrouver des deux côtés et gonfler le score. Ici, seul le DataLoader train rééchantillonne. Le jeu val garde sa distribution officielle.

Le « seuil » est ici une règle multiclasse : argmax de log(p_c)+b_c, avec deux décalages relatifs et le premier fixé à zéro.

Sources : `results/results.csv`, `src/train.py`, `src/evaluate.py`.

## Q7 — Architectures, attention et positions

Weighted-F1 moyen **val**, décision réglée sur val, trois graines :

| Architecture | Aléatoire | fastText | AraVec |
|---|---:|---:|---:|
| C1 BiLSTM | 0,7385 | 0,7427 | 0,7443 |
| C2 BiLSTM-attention | 0,7370 | 0,7476 | 0,7405 |
| C3 Transformer avec positions | 0,7068 | 0,7370 | 0,7211 |
| C4 Character-CNN | 0,6983 | Non applicable | Non applicable |

L'attention n'améliore pas uniformément le BiLSTM : son intérêt dépend de l'initialisation dans cette comparaison. Le meilleur C retenu est attention-fastText.

Effet des positions, mêmes familles d'initialisation :

| Initialisation Transformer | Sans positions | Avec positions |
|---|---:|---:|
| Aléatoire | 0,7210 | 0,7068 |
| fastText | 0,7271 | 0,7370 |
| AraVec | 0,7119 | 0,7211 |

Les positions aident les variantes fastText et AraVec dans cette étude, mais pas la variante aléatoire. On conserve ce résultat négatif.

Sur les cinq exemples sauvegardés de C, les maxima peuvent porter sur un nom propre (« بنكيران », environ 0,923), une ponctuation (apostrophe, environ 0,452), ou des termes latins (« tala3 », 0,451 ; « niveau », 0,181). Le français « niveau » reçoit donc un poids visible dans un exemple, mais les poids ne se concentrent pas exclusivement sur les mots d'opinion. Le corpus de cinq exemples ne permet pas de conclure à une préférence générale pour les emojis ou le français. Une quantification par catégories de tokens n'a pas été exécutée.

Sources : `results/results.csv`, `results/attention_C_5_examples.csv`, `results/attention_D_5_examples.csv`. Attention descriptive, sans conclusion causale.

## Q8 — Gelé, fine-tuné et variante Arabizi

Pour D retenu, le test Weighted-F1 vaut **0,5321**, avec **435 715 paramètres entraînables**. Le coût enregistré est **52,34 s pour le run de tête**, auquel s'ajoutent **486,77 s d'extraction train/dev partagée**, soit 539,11 s dans le comparatif. La latence CPU mesurée est de **88,21 ms/tweet**, chargement exclu.

Pour E1, la campagne complète manque. Le modèle standard avec sa tête comporte environ **146,9 millions de paramètres**. L'estimation de mémoire FP32 est au moins 2,74 Gio avant activations et temporaires, pour environ 2,34 Gio de RAM disponibles au relevé initial. Ce relevé de ressources est historique, pas une mesure de la RAM disponible aujourd'hui.

Les pas techniques enregistrés sur deux exemples ont pris environ 4,73 s pour le modèle standard et 4,50 s pour Arabizi. Ils ne représentent ni un temps d'entraînement complet ni un gain de qualité.

Il est donc impossible de chiffrer le gain D/E1 ou de déclarer un vainqueur DarijaBERT/DarijaBERT-Arabizi. La variante Arabizi constitue une hypothèse pertinente pour les textes latins, à vérifier avec les deux campagnes équitables et une évaluation par écriture.

Sources : `results/comparison.csv`, `results/pretrained_smoke.json`, `results/resource_estimate_E.json`, `results/execution_status.csv`.

## Q9 — Multi-tâche et λ, hors périmètre expérimental

**Aucune expérience arq ni E2 n'est exécutée ou réintroduite.** Le périmètre demandé par l'utilisateur reste ary uniquement. La réponse expérimentale est donc : amélioration et effet de λ non mesurés.

Explication théorique : un encodeur partagé et deux têtes pourraient minimiser L_ary + λ L_arq. λ=0 revient à ignorer l'auxiliaire, et une valeur plus élevée augmente sa contribution. Un partage utile peut régulariser les représentations, tandis que des tâches mal assorties peuvent provoquer un transfert négatif. Un gain n'est jamais garanti par la seule présence de plus de données.

Pour une future étude explicitement autorisée, comparer λ=0, 0,3, 0,5 et 1 avec budgets comparables, trois graines, sélection sur val ary et données d'évaluation indépendantes. Aucun graphique de gain fictif ne figurera dans le PPT.

Source théorique : [Ruder, 2017, An Overview of Multi-Task Learning](https://arxiv.org/html/1706.05098v1). Statut local : exclu du périmètre.

## Q10 — Décalage de prior et oracle a posteriori

| Distribution globale | Val | Test |
|---|---:|---:|
| Négatif | 27,94 % | 28,71 % |
| Neutre | 38,26 % | 32,66 % |
| Positif | 33,81 % | 38,64 % |

Le changement est plus marqué conditionnellement à l'écriture latine : le neutre passe de 52,58 % à 29,89 %. Cela ne prouve pas les hypothèses d'un pur label shift avec distributions conditionnelles des textes invariantes.

Diagnostic du candidat A_char_s42, mêmes scores test sauvegardés :

| Règle | Décalages négatif / neutre / positif | Weighted-F1 test |
|---|---|---:|
| Fixée sur val, résultat principal | 0 / +0,3 / 0 | 0,5365 |
| Argmax brut, diagnostic | 0 / 0 / 0 | 0,5536 |
| Meilleur point a posteriori de la grille originale | 0 / −0,6 / 0 | 0,5738 |

L'écart grille-test / décision-val vaut **0,03725**, soit **3,72 points de F1**. L'oracle correspond au meilleur des 25 points de la grille d'origine, pas à un optimum global continu. Il utilise les labels test et fournit un résultat optimiste de diagnostic. **Il ne remplace aucune règle verrouillée et n'est pas déployé.**

En production, je prévoirais le suivi des écritures, des taux de classes prédites et des scores, puis des métriques sur un flux fraîchement annoté. Une recalibration ou un changement de décision serait validé sur de nouvelles données réservées à cet usage, puis contrôlé sur une évaluation distincte. Ce plan reste proposé, non implémenté.

Sources : `results/f1_label_shift.csv`, `results/f1_conditional_labels.csv`, `results/A_char_s42_test_tuned.csv`, diagnostic séparé dans `evidence.draft.json`.

## Q11 — Écriture non vue et déploiement

F2 mesure les résultats des modèles TF-IDF union appris sur une seule écriture :

| Population test | Union train complet, témoin rétrospectif | Union entraîné sur l'autre écriture | Chute descriptive |
|---|---:|---:|---:|
| Arabe, n=1 367 | 0,7481 | 0,1932, train latin | 55,48 points |
| Latin, n=1 519 | 0,3461 | 0,2098, train arabe | 13,64 points |

Le témoin complet est A_union_s42, déjà entraîné et verrouillé avant test, restauré sans entraînement supplémentaire. Sa mesure sur test est une analyse rétrospective ajoutée pour répondre à Q11 et ne change pas le tableau principal ni le candidat. Les modèles partagent architecture et graine, mais la couverture d'écriture **et la taille de train** changent ensemble : l'écart n'isole donc pas l'effet causal de l'alphabet seul.

Pour un prototype traitant les deux écritures, je retiendrais le candidat préalablement choisi A caractères : il combine le meilleur résultat représentatif global disponible et un coût faible. Le F1 latin reste faible, et les 75 tweets mixtes test ne suffisent pas à garantir le comportement sur des commentaires réellement mixtes. Une validation sur le domaine de commentaires visé reste nécessaire avant un déploiement large.

Sources : `results/f2_cross_script.csv`, `models/A_union_s42`, `results/comparison.csv`, diagnostic séparé Q11 dans `evidence.draft.json`.

## Q12 — Erreurs, bruit d'annotation et choix pratique

Parmi les 30 erreurs sélectionnées : 8 « bruit d'annotation possible », 8 « confusion neutre/polarisé », 7 « ironie ou autre à examiner » et 7 « tokenisation Arabizi possible ».

L'échantillonnage a équilibré les catégories heuristiques : ces proportions **ne mesurent pas leur fréquence parmi toutes les erreurs**. Pour le candidat à caractères, « tokenisation Arabizi possible » désigne une hypothèse de couverture des graphies, pas un échec WordPiece démontré.

Les votes ont été retrouvés pour **18/30 erreurs**, avec **8 désaccords sur 18** (44,4 % des lignes rapprochées), soit 8/30 dans le dossier. Les 12 autres lignes n'ont pas de votes rapprochés. Aucune de ces 30 erreurs n'a encore de validation humaine complète. **La part réelle de labels faux reste inconnue** : désaccord et erreur d'annotation sont distincts.

Comparaison des représentants, mesurée sur cette machine CPU, chargement exclu pour la latence :

| Famille | Weighted-F1 test | Macro-F1 test | Latence CPU | Poids locaux, encodeur compris |
|---|---:|---:|---:|---:|
| A caractères | 0,5365 | 0,5272 | 3,62 ms/tweet | 5,39 Mo |
| B CAMeLBERT | 0,4209 | 0,4156 | 93,13 ms/tweet | 416,17 Mo |
| C attention-fastText | 0,4962 | 0,4852 | 1,85 ms/tweet | 4,17 Mo |
| D états-attention | 0,5321 | 0,5229 | 88,21 ms/tweet | 799,28 Mo |

A est le choix proposé pour un prototype : candidat fixé sur val, score représentatif global le plus élevé, inférence rapide et faible taille. C est plus rapide dans le relevé, mais son score représentatif est inférieur. A et D restent proches en qualité, et McNemar p=0,3231 ne montre pas de différence de taux d'erreur à 5 % ; il ne teste pas directement le F1.

La robustesse F4 d'A vaut 8,8 % de bascules sur ses 102 cas modifiés initialement corrects. On ne compare pas ce chiffre causalement aux autres familles, car leurs sous-ensembles diffèrent. Le gain A/D ne peut pas être annoncé comme une supériorité universelle. Le déploiement envisagé doit être accompagné du suivi des erreurs et d'une collecte annotée représentative, surtout sur le latin et le mélange des écritures.

Sources : `results/f6_30_erreurs.csv`, `results/error_analysis.md`, `results/comparison.csv`, `results/f5_mcnemar.json`, `evidence.draft.json`.
