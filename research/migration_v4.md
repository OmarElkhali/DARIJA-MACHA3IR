# Passage de l'exécution observée à la campagne v4

## Sauvegarde effectuée localement

Avant modification, copie des notebooks, résultats, modèles locaux, sources, scripts et dépendances
dans `../../backups/before_improvements_2026-11-01/`, avec contrôle SHA-256 de chaque fichier.
Le manifeste est `snapshot_manifest.json`. Les modèles/résultats locaux historiques du 29 octobre
ne sont pas présentés comme les nouveaux poids Colab du 1er novembre.

Le notebook local contenait 28 cellules exécutées et 117 sorties. Il est conservé sans modification
dans `notebooks/TP2_Darija_observe_avant_v4.ipynb`, ainsi que dans la sauvegarde complète.
`notebook_measured_dev.csv` et `notebook_run_log.txt` dans la sauvegarde reprennent les mesures et
les logs réellement affichés. Les scores extraits sont arrondis à quatre décimales, pas reconstruits
comme des probabilités ou des poids. Le run E_darijabert_s42 affiche Weighted-F1 dev 0.7911,
Macro-F1 dev 0.7900 et 822.3 s. L'exécution suivante E_darijabert_s43 a été interrompue par l'utilisateur.
Cette interruption ne prouve aucune erreur du modèle. Aucun score E2 ni test v3 n'est inventé.

## Sauvegarder les fichiers encore sur Colab

Les poids de la campagne Colab n'existent pas dans les fichiers locaux reçus. **Avant de supprimer
ou réinitialiser le runtime existant**, importer le nouveau notebook dans la même session et exécuter
1.1 à 1.4. `USE_GOOGLE_DRIVE=True` monte votre Drive avec votre autorisation, puis copie les dossiers
`experiments` des anciens `/content/Darija-Sentiment-runtime-*`, avec code et scripts associés, vers
`MyDrive/Darija-Sentiment/archives/`. Chaque copie est vérifiée par SHA-256. Un échec de copie arrête
la cellule avant pip. Une relance ne recopie pas une source inchangée ; une source modifiée reçoit
une nouvelle archive. Cette procédure est implémentée et testée sur un système de fichiers simulant
Colab, mais n'a pas été exécutée sur votre compte Google par l'agent.

Après le message de sauvegarde réussie à 1.4, poursuivre à 1.5–1.6. La configuration recharge
automatiquement les modules Python du projet si l'ancien runtime, le code ou la sortie diffèrent.
Un redémarrage reste nécessaire si pip remplace des bibliothèques déjà chargées ; reprendre alors
depuis 1.1, la copie inchangée est reconnue. Une session déjà détruite
ne peut pas être récupérée à partir des seules sorties du notebook.

## Nouvelle campagne et reprise

La nouvelle campagne se nomme `campaign_v4`. Elle n'écrase ni n'importe comme nouvelles mesures
les anciens résultats. Le code, le plan de graines et les nettoyages ont changé ; une reprise directe
de la campagne v3 sous son ancien nom serait scientifiquement incohérente et reste refusée.
Le nouveau système reprend les runs v4 après leur dernière époque sauvegardée. Il ne peut pas
créer rétroactivement un checkpoint d'époque pour le run v3 interrompu.

Une seule exécution pour A et les têtes D+LR déterministes ; les réseaux gardent trois graines.
P ajoute les quatre nettoyages avec TF-IDF, deux baselines de raccourcis et trois adaptations BERT
supplémentaires sur trois graines. E1 fournit la référence conservatrice BERT sans doublon.
Les baselines de raccourcis ne participent pas au choix du candidat principal.

## Contrôles et limites

- Réparation des caches incomplets, sélection vide et CSV F4 corrigés.
- Reprise avec états modèle/optimiseur/scheduler/scaler/aléatoires ; écritures temporaires puis remplacement.
- 21 allers-retours entraînement/sauvegarde/rechargement passent localement sur CPU avec de petits
  encodeurs aléatoires. Cela ne valide pas les vrais poids ni la mémoire T4.
- Une interruption CPU après l'époque 1, pendant l'époque 2, puis une reprise donnent les mêmes
  pertes et poids qu'une exécution continue. L'identité numérique n'est pas garantie sur CUDA.
- Audit lexical réel au seuil fixé 0.9 : 9 textes dev proches du train, 16 test proches du train,
  11 test proches de dev, avec ce normaliseur et ce plus proche voisin. Ce sont des signaux à examiner,
  pas des preuves de paraphrase ; les partitions officielles restent intactes.
- F4 utilise la même population et sépare les suppressions. Les scores utilisant les labels originaux
  sont diagnostiques ; une transformation peut changer le sentiment. La revue humaine est distincte.
- Les 12 cas linguistiques sont des propositions construites à annoter. L'ensemble indépendant est
  un fichier vide à collecter/annoter humainement, jamais un test artificiellement déclaré indépendant.
- Les évaluations humaines restent séparées et marquées en attente tant qu'aucun label n'est validé.

Le nouveau notebook est préparé pour exécution ; aucun gain de benchmark v4 n'est revendiqué.
