# DARIJA · MACHA3IR

Analyse du sentiment en darija marocaine : six familles de méthodes, leurs résultats mesurés et leurs limites. Travail sur les partitions officielles `ary` d’AfriSenti : **5 583 tweets train, 494 validation, 2 961 test**.

**[Ouvrir le site](https://omarelkhali.github.io/DARIJA-MACHA3IR/)** · **[Lire le notebook en HTML](https://omarelkhali.github.io/DARIJA-MACHA3IR/downloads/TP2_Darija_complet_resultats.html)** · **[Télécharger le notebook final](notebooks/TP2_Darija_complet_resultats.ipynb)** · **[Soutenance illustrée — 38 diapositives](presentation/Darija_Soutenance_38_Slides_QR/Darija_Soutenance_38_Slides_QR_Finale.pptx)** · [Notes pour l’oral](presentation/Darija_Soutenance_38_Slides_QR/speech.md)

## Résultats de la campagne

Le représentant de chaque famille est choisi sur validation, puis évalué sur test avec la graine 42 et les décalages de décision figés. La sélection E repose sur la moyenne de validation, **0,811256**. Les 33 variantes et les 71 évaluations finales sont documentées dans les sorties enregistrées. Le modèle P DarijaBERT sans signes est exclu du test, conformément à la décision de la campagne.

| Famille | Représentant choisi sur validation | Weighted-F1 test | Macro-F1 test |
|---|---|---:|---:|
| A | TF-IDF caractères + régression logistique | 0,553509 | 0,543395 |
| B | CAMeLBERT, transfert fixe | 0,420555 | 0,415300 |
| C | BiLSTM + AraVec | 0,493340 | 0,484702 |
| D | DarijaBERT gelé, séquence d’états | 0,571593 | 0,565359 |
| E | DarijaBERT-Arabizi ajusté | **0,608138** | **0,600620** |
| P | TF-IDF union, sans emojis ni ponctuation | 0,567396 | 0,557839 |

IC bootstrap à 95 % du F1 test de E : **[0,590655 ; 0,624099]**, 1 000 rééchantillonnages conditionnels au checkpoint. Les chiffres sont transcrits à la précision affichée dans les notebooks, sans inventer les colonnes tronquées.

Le test avait déjà été consulté au cours du travail historique : ces résultats restent exploratoires. E2 utilise seulement λ=0,1 et 0,3, graine 42, trois époques maximum. La revue humaine est en attente ; aucun score indépendant n’est revendiqué. Les temps cumulés excluent les doubles comptes raw/tuned et reprises ; les époques perdues avant checkpoint et coupures réseau ne sont pas mesurables.

## Récupérer le projet complet

Les fichiers de modèles disponibles et les ressources AraVec sont versionnés avec **Git LFS**. Installer Git LFS, puis cloner le dépôt avec les fichiers binaires :

```powershell
git lfs install
git clone https://github.com/OmarElkhali/DARIJA-MACHA3IR.git
cd DARIJA-MACHA3IR
git lfs pull
```

Le projet local synchronisé représente environ **1,78 Gio**, hors environnements virtuels et caches. `project_manifest.json` donne les fichiers, tailles, empreintes SHA-256 et exclusions d’exécution. Le checkpoint E absent localement ne peut pas être ajouté ; aucune copie n’est inventée.

L’application Streamlit d’origine (`app.py`, `assets/`, `.streamlit/`), les notebooks historiques, les résultats locaux et les anciennes présentations sont également conservés. Les dossiers racine `models/` et `results/` et la soutenance originale constituent des archives de campagnes antérieures : leurs scores ne remplacent pas ceux de `campaign_v4` et du notebook final indiqué en tête de ce document.

## Notebook de présentation

`notebooks/TP2_Darija_complet_resultats.ipynb` rassemble le code A/B/C/D/E/P, les analyses F/G, les tableaux et figures réellement enregistrés, avec une narration en français et le code source en annexe. **Il a été assemblé sans exécuter les notebooks.** Les sorties sont importées avec leur provenance ; les compteurs d’exécution sont réinitialisés. L’entraînement et l’évaluation finale ne sont pas déclenchés par défaut dans ce livrable.

Les deux notebooks d’origine sont conservés comme sources. Pour reconstruire uniquement les livrables, sans entraînement :

```powershell
python -m pip install -r requirements-notebook.txt -r requirements-presentation.txt
python scripts/create_final_notebook.py
python scripts/build_snapshot_site.py
python scripts/build_results_presentation.py
```

## Site HTML, CSS et JavaScript

Le site statique propose le corpus, le protocole, les six méthodes, les 33 variantes filtrables, les graphiques, les analyses F/G, les limites et douze questions pour l’oral. `site/dist/` est la version locale ; `docs/` est la copie publiée par GitHub Pages. Aucune compilation JavaScript n’est nécessaire.

GitHub Pages héberge la présentation et les résultats. **Les prédictions nécessitent le moteur Python local**, car les checkpoints PyTorch ne s’exécutent pas dans le navigateur. Le site permet d’y connecter le laboratoire ; l’interface complète fonctionne également sur le serveur local.

```powershell
python -m pip install -r requirements-notebook.txt
python scripts/cache_pretrained.py
python site/server.py
```

Ouvrir **http://127.0.0.1:8765**, puis écrire une phrase. Les poids sélectionnés de A/C/D/P et le marqueur préentraîné de B sont fournis. La commande `cache_pretrained.py` prépare les encodeurs publics B/D dans le cache Hugging Face ; le serveur les charge ensuite hors ligne. Ce téléchargement peut être volumineux. Le serveur vérifie les empreintes SHA-256 des artefacts locaux. Les phrases ne sont pas enregistrées. Utiliser seulement des checkpoints de confiance.

**Cinq méthodes ont été testées avec de véritables prédictions. Le checkpoint ajusté E `E_arabizi_s42/weights.pt` manque dans les fichiers locaux** : son résultat scientifique existe dans les sorties enregistrées, mais le laboratoire affiche son indisponibilité. Aucun score de phrase n’est simulé. Pour réactiver E, restaurer le fichier original correspondant au manifeste dans `experiments/campaign_v4/models/E_arabizi_s42/weights.pt`, puis redémarrer le serveur. Le checkpoint P DarijaBERT exclu reste hors F/G.

## Organisation et vérification

- `src/` : préparation, modèles, apprentissage, évaluation et analyses.
- `experiments/campaign_v4/` : paramètres, métadonnées, historiques, résultats et checkpoints disponibles. Les binaires sont suivis avec Git LFS ; les caches et états temporaires de reprise sont exclus.
- `research/notebook_tables_snapshot.json` : tableaux extraits avec cellule et sortie d’origine ; `research/notebook_figures/` : figures enregistrées.
- `site/dist/downloads/` : tableaux CSV, notebook et archive de résultats transcrits. Cette archive ne prétend pas remplacer les CSV bruts masqués par l’affichage tronqué.
- `presentation/Darija_Soutenance_38_Slides_QR/` : soutenance finale en 38 pages, notes intégrées, comparaison scientifique et 12 questions-réponses avec renvois. Anciennes versions supprimées.

Validation effectuée : structure et compilation des cellules du notebook sans exécution, export HTML, 38 diapositives illustrées inspectées, provenance de génération et notes intégrées vérifiées, graphiques issus des mesures, navigation/filtres et affichage mobile du site, absence d’erreur JavaScript, validation des requêtes de l’API et cinq prédictions réelles. Rapport : `research/web_delivery_checks.json`. Le script `scripts/check_web_delivery.py` utilise Playwright en mode headless ; il ne contrôle pas le bureau de l’utilisateur.

## Références

[AfriSenti](https://aclanthology.org/2023.emnlp-main.862/) · [AfriSenti-SemEval](https://aclanthology.org/2023.semeval-1.315/) · [DarijaBERT](https://doi.org/10.1007/s41060-023-00498-2) · [fastText](https://aclanthology.org/Q17-1010/) · [Données officielles](https://github.com/afrisenti-semeval/afrisent-semeval-2023).

Les données et modèles tiers restent soumis aux licences de leurs auteurs. Aucune licence générale n’est appliquée à ces ressources par ce dépôt.
