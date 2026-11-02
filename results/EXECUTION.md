# Bilan réel d'exécution — 29 octobre 2026

Périmètre : **ary exclusivement**. Exécution Conda locale, CPU, sans CUDA. Les énoncés officiels Q1–Q12 sont absents ; les douze thèmes d'oral ne sont pas leur remplacement certifié.

## Expériences exécutées

76 runs distincts ; 184 lignes de métriques (dev/test × argmax/règle dev), sans scores inventés. Répartition : {'A': 18, 'B': 4, 'C': 45, 'D': 9}. Les modèles B sont fixes : un run par variante, pas trois répétitions artificielles. Les familles entraînées disposent de trois graines par variante. F2 possède ses propres sous-ensembles train, conformément à sa définition.

```text
family                            status            variant  weighted_f1  macro_f1  n_seeds  cpu_ms_per_tweet
     A                           exécuté               char     0.536523  0.527248      3.0          3.623907
     B                           exécuté          camelbert     0.420902  0.415620      1.0         93.126153
     C                           exécuté attention_fasttext     0.496214  0.485216      3.0          1.846203
     D                           exécuté             states     0.532093  0.522852      3.0         88.206060
     E non exécuté — aucun résultat test                NaN          NaN       NaN      NaN               NaN
```

## Variantes non exécutées

```text
family    variant        seeds                    status
     E darijabert [42, 43, 44] non exécuté (0/3 graines)
     E    arabizi [42, 43, 44] non exécuté (0/3 graines)
```

La campagne E complète n'a pas été exécutée : pas de GPU, budget RAM réduit et coût CPU élevé. Le modèle standard comporte environ 147 millions de paramètres avec sa tête ; poids FP32 + gradients + deux moments AdamW + copie du meilleur état demandent au moins 2,74 Gio, avant activations et temporaires. Voir resource_estimate_E.json. Un pas technique sur deux exemples, lorsqu'il est présent dans pretrained_smoke.json, n'est pas un entraînement complet et ne crée aucune métrique E dans results.csv.

## Robustesse

F1 : proportions dev/test et comparaison a posteriori des décisions sans réoptimiser. F2 : arabe→latin et latin→arabe avec graines 42/43/44. F3 : F1 par écriture, OOV train/dev/test ; translittération facultative non exécutée. F4 : perturbations emojis/ponctuation sur les tweets polarisés initialement corrects de chaque famille sélectionnée. F5 : 1 000 bootstrap, IC 95 %, variabilité entre graines pour A/C/D, McNemar exact apparié. F6 : 30 erreurs avec catégories **provisoires**, votes disponibles et colonnes de validation humaine ; la validation interprétative reste à faire.

## Vérifications

- Notebook : 40 cellules de code exécutées, aucune erreur, un affichage par cellule ; export HTML avec sorties.
- Pipeline : formes, gradients, masques, vocabulaire, FastText, tête séquentielle et boucle de fine-tuning sur petit encodeur de test.
- Streamlit : six espaces de présentation, corpus sans labels test, prédictions réelles A/B/C/D, filtres, erreurs de saisie/checkpoint, historique et mode soutenance. Voir ui_verification.json ; la vérification visuelle par navigateur automatisé n’a pas été menée à terme.
- Sélection : empreintes des checkpoints verrouillés vérifiées avant test ; aucune nouvelle sélection à partir du test.

## Limites scientifiques

20 doublons train, 2 collisions entre partitions officielles conservées ; absence de contamination externe non certifiée ; petit effectif mixte ; couleurs d'attention descriptives et non causales ; catégories F6 et indices de français à valider ; E complet absent. Temps D hors extraction lorsque le cache est utilisé : coût d'extraction partagé dans frozen_extraction_time.json.

- Checkpoints : 76 rechargements sur un exemple dev de chaque ecriture ; ecart maximal aux scores sauvegardes 1.2e-07. Voir checkpoint_verification.json.

## Corrections et améliorations du 29 octobre 2026

- Brouillon, mode avancé et choix des modèles conservés entre les pages ; analyse précédente signalée si la phrase ou les modèles changent.
- Chaque modèle comparé peut être examiné séparément ; exports avec la phrase et limites de longueur visibles.
- Réinitialisation des filtres du corpus ; seules les graines effectivement exécutées sont proposées pour les courbes.
- Restauration des modèles depuis le cache uniquement, avec la révision du run ; détection des fichiers locaux manquants et métadonnées illisibles. Import du moteur différé au clic.
- Validation commune des probabilités et décalages : les scores non finis ou non normalisés ne produisent plus de décision.
- Le script d’exécution synchronise le HTML après réussite et compte les cellules réellement exécutées ; export dupliqué supprimé du vérificateur.

Contrôles après correction : pipeline OK ; six pages Streamlit sans exception ; prédictions réelles A/B/C/D avec requêtes HTTP interdites pendant le contrôle ; 76 checkpoints fidèles aux scores dev sauvegardés (écart maximal 1.2e-07) ; 40 cellules de code exécutées sans erreur et HTML régénéré ; service HTTP 200. Les empreintes du benchmark restent intactes. Rendu visuel navigateur non revérifié.
