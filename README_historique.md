# TP2 — Groupe 12 : Darija-Sentiment

Projet universitaire indépendant d'analyse de sentiment **marocain ary uniquement** : `negative`, `neutral`, `positive`. Le livrable principal est [le notebook pédagogique](notebooks/TP2_Darija.ipynb), avec **40 cellules de code** et des explications en français. Le périmètre demandé a été corrigé : aucun corpus algérien, aucune tête arq, aucun apprentissage multi-tâche entre pays.

## Démarrer en Conda local

Python 3.11 est utilisé. Le Python 3.14 présent sur cette machine n'est pas le kernel du projet.

```powershell
cd 'D:\SetupGame\Desktop\TP LLM\DARIJA SENTIEMENTS\Darija-Sentiment'
# Environnement existant vérifié :
& 'C:\Users\SetupGame\anaconda3\python.exe' -m pip install -r requirements.txt
& 'C:\Users\SetupGame\anaconda3\python.exe' -m ipykernel install --user --name darija-sentiment --display-name 'Darija-Sentiment (Conda)'
& 'C:\Users\SetupGame\anaconda3\python.exe' -m jupyter notebook notebooks/TP2_Darija.ipynb
```

Dans VS Code, sélectionner **Darija-Sentiment (Conda)**. Pour un nouvel environnement : `conda create -n darija-sentiment python=3.11`, puis `conda activate darija-sentiment` et `python -m pip install -r requirements.txt`. Ne pas installer toutes les dépendances dans le Python système 3.14.

Le notebook livré lit les résultats sauvegardés : les drapeaux `TRAIN_A` à `TRAIN_E` et `LOCK_AND_TEST` sont faux. Son exécution complète ne déclenche pas de campagne lourde. `python scripts/execute_notebook.py` exécute le notebook et régénère son HTML uniquement si l’exécution réussit. Les fichiers officiels sont déjà présents, intacts. La provenance et les hashes figurent dans `data/manifest.json`.

## Données et frontière dev/test

Révision officielle : `5aec3cfcf87b59f885e98641f72e52182776429d` du [dépôt AfriSenti](https://github.com/afrisenti-semeval/afrisent-semeval-2023).

| Partition ary | Effectif vérifié | Rôle |
|---|---:|---|
| train | 5 583 | Apprentissage uniquement |
| dev | 494 | Choix des modèles, époques et décalages |
| test | 2 961 | Évaluation finale après verrouillage |

L'audit a trouvé **20 doublons internes train**, **2 collisions textuelles train/dev**, aucun chevauchement train/test et **aucune contradiction de labels détectée**. Voir `results/audit/` (l'audit incluant les labels test est produit seulement après verrouillage). Les partitions n'ont pas été modifiées. Les votes individuels marocains sont fournis par le dépôt et conservés dans `data/provenance/`. Leur rapprochement textuel est conservateur ; des mentions anonymisées empêchent certains matchs.

`load_split('test')` renvoie les textes sans labels. L'ouverture des labels test exige un fichier de sélection ; l'évaluation finale vérifie les empreintes des checkpoints verrouillés. Cette protection est une aide au protocole, pas une barrière de sécurité contre un utilisateur qui ouvrirait manuellement le TSV public. L'audit ne règle jamais les modèles avec les labels test.

Toutes les variantes exécutées sont comparées sur **dev**. Le test évalue ensuite la variante retenue par famille sur ses trois graines (B fixe : une seule), ainsi que les expériences F2 prédéfinies. Les variantes écartées sur dev ne sont pas testées pour chercher un autre gagnant.

## Les cinq familles

| Famille | Variantes implémentées |
|---|---|
| A | Majorité ; TF-IDF mots 1–2 ; caractères 2–5 ; combinaison ; LR équilibrée ; décision dev |
| B | NLI mDeBERTa avec hypothèses ar/fr/en ; CAMeLBERT sentiment externe |
| C | BiLSTM ; BiLSTM-attention ; Transformer 2 couches/128D/4 têtes avec/sans position ; Character-CNN ; hasard/AraVec/fastText pour les réseaux à mots ; perte pondérée ; oversampling train |
| D | DarijaBERT gelé : CLS+LR ; mean+LR ; états+BiLSTM-attention |
| E | Fine-tuning DarijaBERT et DarijaBERT-Arabizi, uniquement ary |

Le module `src/preprocess.py` gère provenance, nettoyage et vocabulaire ; `src/models.py` les architectures ; `src/train.py` les boucles et checkpoints ; `src/evaluate.py` les décisions, mesures et F1–F6 ; `src/research.py` la bibliographie, les solutions et l'oral. Chaque run sauvegarde graine/configuration, historique si applicable, métriques et chemin du checkpoint.

Le vocabulaire C est construit sur train à fréquence minimale 2. Les mots hors vocabulaire restent UNK, y compris pour l'initialisation fastText : cette expérience compare des initialisations, pas une inférence ouverte par sous-mots sur tous les OOV. Gensim implémente FastText sans dépendre du paquet `fasttext` difficile à compiler sous Windows. AraVec 100D est projeté à 128D par une matrice aléatoire fixée et ajusté ensuite avec le réseau. Les poids AraVec éventuellement fournis sont une ressource externe de [l'auteur](https://github.com/bakrianoo/aravec), pas des résultats d'un autre TP.

## Reproduire les expériences

**Avant tout verrouillage**, dans une nouvelle copie de travail sans les anciens `results/` et checkpoints de runs :

```powershell
python -c "from src.preprocess import download_data; print(download_data())"
python tests/check_pipeline.py
python -m src.train --part A
python -m src.train --part C --aravec-path models/resources/aravec/full_uni_cbow_100_twitter.mdl
python -m src.train --part B
python -m src.train --part D
python -m src.train --part E --device cuda
python -m src.train --part status
# Une fois tous les choix dev terminés :
python -m src.train --part finalize
```

`--part quick` exécute A, les deux entraînements F2 (sélectionnés sans test), puis C avec cinq architectures aléatoires, attention-fastText, attention-perte pondérée et attention-oversampling, chacune sur 42/43/44. `--part C` ajoute les autres variantes fastText et AraVec si fourni. `--part all` enchaîne toutes les familles. Les téléchargements B/D/E peuvent être volumineux ; ils sont figés par révision Hugging Face.

E : AdamW, LR 2e-5, micro-batch 4, accumulation 8, batch effectif 32, trois époques, warmup 10 %, patience 2 après trois époques minimum. Pour passer à cinq époques ou changer le micro-batch, appeler explicitement `finetune_run(..., epochs=5, micro_batch=...)` **avant** verrouillage et pour les deux modèles aux mêmes conditions. Le CPU est accepté mais très lent. CUDA ne peut pas être activé par un simple paramètre si PyTorch est une build CPU.

Après verrouillage, les fonctions d'entraînement refusent une modification de l'étude. Une nouvelle étude doit être annoncée ; revoir un test déjà observé ne recrée pas un jeu indépendant. Le code n'optimise aucun hyperparamètre sur test.

## Résultats et limites

- `results/results.csv` : uniquement des métriques réellement calculées, par run, partition et règle ; **ne pas compter argmax et dev_offsets comme deux entraînements**.
- `results/execution_status.csv` : couverture réelle de toutes les variantes obligatoires, y compris les runs non exécutés.
- `results/comparison.csv` : cinq familles, représentant graine 42 choisi par variante moyenne dev ; scores test, latence CPU et taille locale, IC bootstrap et stabilité.
- `results/selection_lock.json` : candidat, variantes et empreintes figés avant test.
- `results/f1_label_shift.csv`, `f2_cross_script.csv`, `*_scripts.csv`, `f4_perturbations.csv`, `f5_mcnemar.json`, `f6_30_erreurs.csv` : artefacts F1–F6 lorsqu'ils ont été exécutés.
- `research/papers.md` : onze publications vérifiées, dates et portée des résultats.
- `research/solutions.md` : propositions implémentées et améliorations/dev réellement observées.
- `research/oral.md` : douze thèmes provisoires avec preuves disponibles. **Les énoncés officiels Q1–Q12 manquent** : aucune correspondance exacte n'est inventée.
- `results/generalization_analysis.md` : baisse dev/test, changement des relations écriture/labels, mentions et artefacts du corpus ; aucune correction choisie sur test.
- `results/error_analysis.md` : lecture assistée provisoire des 30 erreurs, avec hypothèses, français, ambiguïtés de cible et validation humaine encore nécessaire.

F5 utilise 1 000 bootstrap de tweets, IC percentile 95 %. Cet IC conditionne sur le checkpoint ; l'écart-type entre graines est distinct. McNemar exact compare les taux d'erreur de deux modèles appariés et ne teste pas directement le F1. F4 est conditionné aux tweets polarisés initialement corrects et effectivement modifiés. Les catégories F6 sont heuristiques et doivent être validées humainement ; le fichier fournit les colonnes de revue.

Les résultats de la littérature restent séparés des résultats locaux. CAMeLBERT a déjà été entraîné en sentiment externe (ASTD, ArSAS, SemEval) ; DarijaBERT a utilisé des tweets au préentraînement. L'absence de contamination externe ne peut pas être certifiée. Les 2 collisions officielles sont une autre limite. Latin ≠ Arabizi certain ; mixed ≠ français détecté.

Les temps D dépendent de la présence de caches. Dans le tableau G, `run_seconds` conserve le temps du run et `feature_extraction_seconds_shared` le coût partagé de préparation train/dev ; `train_seconds` additionne ces deux coûts pour la méthode représentative. `weight_mb` additionne les fichiers de poids du run et le checkpoint externe B/D réellement présent dans le cache ; `external_encoder_mb` indique sa contribution. Il s'agit de la taille des fichiers distribués, qui peuvent aussi contenir une tête de préentraînement inutilisée. La latence CPU inclut l'encodeur lorsqu'il est utilisé, après chargement/échauffement, sur 30 prédictions unitaires.

## Application Streamlit et vérifications

```powershell
python -m streamlit run app.py
python tests/check_pipeline.py
python tests/check_app.py
python tests/check_checkpoints.py
python scripts/execute_notebook.py
```

L’interface **Darija / Sentiment** comprend six espaces :

- **Vue d’ensemble** : scores dev/test, matrice interactive, qualité/coût et export du comparatif.
- **Laboratoire** : exemples arabe/Arabizi/mixte, comparaison des modèles, scores et règle de décision, historique de session et export CSV. Le mode avancé expose les 76 checkpoints disponibles.
- **Corpus** : recherche, filtres par écriture et sentiment, distributions, longueurs, tokens fréquents, audit et transfert d’un tweet vers le laboratoire. Le test reste sans labels dans cet explorateur.
- **Modèles** : explications des familles A–E, variantes, courbes d’apprentissage, paramètres et attention sur cinq exemples.
- **Robustesse** : neuf graphiques interactifs pour explorer F1–F6, les intervalles de confiance, les changements de distribution et les erreurs commentées.
- **Recherche** : fiches de publications, solutions testées, lexique et préparation de l’oral ; téléchargement des notebooks HTML et exécutable.

Le **mode soutenance** ajoute un parcours guidé avec précédent/suivant. Les graphiques permettent le survol, le zoom, l’activation des légendes et l’export PNG. Les données des graphiques proviennent uniquement des artefacts enregistrés. La direction visuelle et les maquettes sont documentées dans `design/README.md`.

Les modèles sont chargés **séquentiellement au clic**, puis libérés. Aucun checkpoint absent ne produit de prédiction. Les scores affichés sont bruts, non calibrés ; la décision affiche les décalages réellement choisis sur dev. L’historique conserve au plus huit analyses dans la session et n’altère jamais les résultats de benchmark. Les vérifications de l’interface sont enregistrées dans `results/ui_verification.json`.

Le brouillon, les modèles choisis et le mode avancé restent conservés lorsqu’on change de page. Si le texte ou les modèles changent, l’ancienne analyse est signalée jusqu’au prochain clic. Chaque résultat de la comparaison peut être examiné séparément ; les exports incluent la phrase analysée. Le corpus possède une remise à zéro des filtres.

La restauration des checkpoints B/D est **strictement locale**, avec la révision enregistrée dans chaque run : aucune requête HTTP n’est nécessaire pour tester une phrase une fois le cache préparé. Un vocabulaire, tokenizer ou encodeur manquant rend le checkpoint indisponible ; une métadonnée illisible est signalée sans bloquer les autres modèles. Les limites de longueur sont affichées dans le laboratoire : la fin des textes trop longs est tronquée par les modèles à longueur fixe. Des scores invalides ou non normalisés sont refusés au lieu de produire un sentiment trompeur.

Le fichier `results/verification.json` et le bilan `results/EXECUTION.md` précisent les vérifications effectuées et les expériences non exécutées. La présence d'une architecture ou d'une commande n'est jamais présentée comme une preuve d'entraînement.
