# Audit de livraison — périmètre ary

Le périmètre initial est conservé, sauf l'exclusion explicite de `arq` et du multi-tâche entre pays demandée ensuite. **L'objectif global reste incomplet tant que les énoncés officiels Q1–Q12 ne sont pas fournis et intégrés.**

| Exigence | État constaté | Preuve actuelle |
|---|---|---|
| Projet indépendant et arborescence demandée | Présent | README, requirements, cinq modules src, data/ary, notebooks, research, results, models, app.py |
| Exclusivement ary | Vérifié | `load_split` refuse une autre langue ; aucun dossier data/arq ; configs de runs |
| Trois sentiments | Vérifié | Ordre canonical negative/neutral/positive dans preprocess.py et scores sauvegardés |
| Partitions officielles 5583/494/2961 | Vérifié | Révision Git et SHA-256 figés ; data/manifest.json ; results/audit/counts.csv |
| Votes individuels | Présents et exploités avec prudence | CSV public marocain dans data/provenance ; jointure exacte en F6 |
| Doublons, collisions, contradictions | Audités | audit/counts.csv, collisions.csv, conflicts.csv, conflicts_including_test_final.csv |
| Aucun mélange train/test ni transformation apprise sur dev/test | Vérifié dans les voies d'apprentissage | train.py : vectoriseurs/vocabulaire/fastText ajustés sur train ; modèles préentraînés figés par révision |
| Choix sur dev, test après verrouillage | Vérifié | selection_lock.json, hashes des 294 artefacts, select_on_dev et final_evaluation |
| Graines, configurations, provenance, checkpoints | Présents | 76 dossiers de runs avec meta.json, poids/référence externe et prédictions dev |
| Métriques authentiques | Présentes | results.csv : 184 lignes, 76 runs, aucune métrique E complète inventée |
| Huit publications initiales | Fiches présentes | research/papers.md ; auteurs, année, problème, solution, résultats des auteurs, application, expérience et source |
| Meilleures équipes et recherche récente | Présentes | NLNDE, UCAS-IIE-NLP, Atlas-Chat ; onze fiches au total ; recherche non exhaustive déclarée |
| Google Scholar | Tentative non aboutie déclarée | papers.md distingue l'accès échoué des sources primaires effectivement consultées |
| Famille A et variantes | Exécutées | Majorité, mots 1–2, caractères 2–5, union, LR balanced ; graines 42/43/44 ; décisions dev et raccourci d'écriture |
| Famille B et variantes | Exécutées | NLI ar/fr/en et CAMeLBERT ; métriques dev, décomposition par écriture et test du représentant fixé |
| Limites Arabizi et contamination externe | Explicitées | Notebook B, papers.md, généralisation ; aucune absence de contamination certifiée |
| Famille C et variantes | Exécutées | 15 variantes × trois graines : BiLSTM, attention, Transformer 2 couches/128D/4 têtes ± positions, char-CNN ; embeddings et ablations |
| Vocabulaire train, min_freq=2, OOV | Vérifié | build_vocab, encode_texts, oov.csv ; plafond du vocabulaire fermé explicitement décrit |
| AraVec et fastText | Exécutés | Couverture, provenance AraVec et checkpoints ; fastText appris sur train uniquement |
| Perte pondérée, oversampling train, décisions dev | Exécutés | Variantes C correspondantes et meta.json |
| Attention C sur cinq exemples | Vérifiée | attention_C_5_examples.csv : cinq séquences, poids normalisés ; figure et affichage RTL |
| Famille D | Exécutée | CLS+LR, mean+LR, états+BiLSTM-attention, trois graines ; encodeur requires_grad=False |
| Attention D, temps et paramètres | Vérifiés | attention_D_5_examples.csv, historiques, comparison.csv ; extraction partagée signalée |
| Famille E sur ary seulement | Implémentée et contrôlée techniquement | Deux vrais checkpoints : forward/backward/pas AdamW sur deux exemples, sans métrique de benchmark |
| Campagne E complète | Non exécutée, explicitement signalée | execution_status.csv, resource_estimate_E.json, pretrained_smoke.json et EXECUTION.md |
| Protocole E | Présent | AdamW 2e-5, batch effectif 32, warmup 10 %, 3–5 époques autorisées, early stopping et trois graines |
| E2 arq/multi-tâche | Exclu par la consigne ultérieure | Une seule tête sentiment ary dans FineTunedSentiment |
| F1 | Exécuté après verrouillage | Distributions globales/conditionnelles, règles brute/dev, corpus_markers et generalization_analysis.md |
| F2 | Exécuté | f2_cross_script.csv : arabe→latin et latin→arabe, trois graines |
| F3 | Exécuté ; translittération facultative non exécutée | Scores par écriture et OOV ; absence d'expérience de translittération annoncée |
| F4 | Exécuté | f4_A/B/C/D_perturbations.csv ; sous-ensembles et limites documentés |
| F5 | Exécuté | 1000 bootstrap, IC 95 %, variabilité des graines et McNemar exact ; B fixe non répliqué artificiellement |
| F6 | Analyse provisoire présente | Trente erreurs uniques, votes, catégories automatiques et lecture assistée de chaque erreur ; validation humaine encore à faire |
| G : cinq familles et tous les champs | Présent avec E non mesuré | comparison.csv ; F1, précision/rappel, temps, latence CPU, poids, paramètres, écritures, robustesse et graines |
| Matrice finale 3×3 | Présente et inspectée | figures/modele_final.png, correspondant au candidat fixé sur dev |
| Notebook pédagogique de 35–45 cellules de code | Vérifié | 40 cellules de code exécutées sans erreur ; 93 cellules au total ; un affichage par cellule |
| Objectif, pourquoi, principe, exemple et recherche avant le code | Vérifiés | Contrôle des cellules Markdown précédant chacune des 40 cellules de code |
| Mini-résumés A–E | Présents | Cinq cellules avec les six rubriques demandées |
| Résultats et interprétations sans gains inventés | Présents | Tableaux réels, interprétation conditionnelle et résultats négatifs conservés |
| Figures réelles enregistrées, arabe lisible | Vérifiés | Quinze PNG, tableaux/attention en HTML avec direction automatique ; export HTML du notebook |
| Tableau recherche/solutions et états | Présent | research/solutions.md et results/research_solutions.csv |
| Application et présentation interactive | Vérifiée | Six espaces, fonctions initiales conservées ; ui_verification.json : corpus sans labels test, prédictions A/B/C/D, filtres, erreurs, historique et soutenance |
| Chargement progressif et tous les checkpoints disponibles | Vérifié | Chargement séquentiel ; 76 rechargements comparés aux scores dev sauvegardés dans checkpoint_verification.json |
| Présentation personnelle et questions probables du professeur | Présentes | research/oral.md, mais douze thèmes encore provisoires |
| Réponses exactes Q1–Q12 du sujet officiel | **Manquantes** | Aucun sujet/énoncé officiel fourni ; l'utilisateur a indiqué qu'il allait les fournir |

Les restrictions conditionnelles d'exécution sont décrites, sans les transformer en résultats. L'absence des questions officielles reste une information indispensable ; aucun thème inventé n'est déclaré être la question officielle correspondante.
