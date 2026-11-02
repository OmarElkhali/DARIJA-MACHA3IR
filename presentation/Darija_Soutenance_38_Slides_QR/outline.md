# Plan approuvé — soutenance illustrée et comparaison scientifique

**Plan validé par l’auteur ; production sans autre confirmation.**

36 diapositives : les 35 pages de la version détaillée et une nouvelle comparaison scientifique après les résultats test.
23 pages originales conservées. Les 12 nouvelles pages seront refaites, plus la page de comparaison, soit 13 pages à produire.

## Direction visuelle demandée

Même style que les anciennes diapositives : papier ivoire, vert forêt, sauge, terre cuite, grands titres sérif, schémas et illustrations explicatives. Compositions variées et densité comparable aux pages d’origine.
Production souhaitée avec le skill codex-ppt : pages illustrées complètes, texte et données inclus dans les images, puis assemblage PowerPoint avec notes orales.
Outil proposé : génération d’images intégrée, déjà disponible. Après validation du plan et de cet outil, une page témoin sera soumise avant la production des autres pages.

## Plan et affectation des images

### 1. Analyse de sentiment en darija marocaine
Page d’origine 1 conservée.
Image stricte conservée : `presentation/Darija_Sentiment_Soutenance_Resultats/origin_image/slide_01.png`.

### 2. Une langue, plusieurs graphies
Page d’origine 2 conservée.
Image stricte conservée : `presentation/Darija_Sentiment_Soutenance_Resultats/origin_image/slide_02.png`.

### 3. Train apprend, validation choisit, test mesure
Page d’origine 3 conservée.
Image stricte conservée : `presentation/Darija_Sentiment_Soutenance_Resultats/origin_image/slide_03.png`.

### 4. Lire les distributions avant les scores
Page d’origine 4 conservée.
Image stricte conservée : `presentation/Darija_Sentiment_Soutenance_Resultats/origin_image/slide_04.png`.

### 5. L’écriture peut devenir un raccourci
Page d’origine 5 conservée.
Image stricte conservée : `presentation/Darija_Sentiment_Soutenance_Resultats/origin_image/slide_05.png`.

### 6. A — Comprendre TF-IDF et les caractères
Page d’origine 7 conservée.
Image stricte conservée : `presentation/Darija_Sentiment_Soutenance_Resultats/origin_image/slide_07.png`.

### 7. A — Résultats et temps d’entraînement
Page à produire dans le style illustré d’origine.
- TF-IDF mots : 1,79 s. Caractères : 8,39 s. Union : 9,06 s, graine 42.
- A caractères : F1 test 0,553509. Huit runs A : 49,15 s cumulées.
- Visuel : mécanisme TF-IDF, cartes de fragments et chronomètre, avec les valeurs intégrées dans la page.
- Références de style uniquement : pages originales 7 (TF-IDF), 10 (déséquilibre), 13 (D/E), 22 (erreurs). Leurs anciens textes ne remplacent pas les chiffres ci-dessus.

### 8. B1 — Reformuler le sentiment comme une hypothèse
Page d’origine 8 conservée.
Image stricte conservée : `presentation/Darija_Sentiment_Soutenance_Resultats/origin_image/slide_08.png`.

### 9. B — Transfert fixe et coût local
Page à produire dans le style illustré d’origine.
- Aucun entraînement local sur ary : 0 s.
- CAMeLBERT : validation 0,602343 et test 0,420555. Arabe / latin : 0,558 / 0,144.
- Visuel : encodeur verrouillé, tweets arabe/latin, trois sentiments et coût d’inférence.
- Références de style uniquement : pages originales 7 (TF-IDF), 10 (déséquilibre), 13 (D/E), 22 (erreurs). Leurs anciens textes ne remplacent pas les chiffres ci-dessus.

### 10. Déséquilibre : agir sur l’apprentissage et la décision
Page d’origine 10 conservée.
Image stricte conservée : `presentation/Darija_Sentiment_Soutenance_Resultats/origin_image/slide_10.png`.

### 11. C — Des embeddings à la décision
Page d’origine 11 conservée.
Image stricte conservée : `presentation/Darija_Sentiment_Soutenance_Resultats/origin_image/slide_11.png`.

### 12. C — Apprentissage, résultats et durée
Page à produire dans le style illustré d’origine.
- BiLSTM + AraVec : run de 43,43 s, six époques observées.
- Validation moyenne 0,753140, test graine 42 de 0,493340. Total C : 20 min 08 s.
- Figure stricte : courbes C originales. Compléter le pied de la page sans inventer de traces.
- Image stricte : `site/dist/figures/curves_C.png`, courbes enregistrées, à conserver fidèlement.
- Références de style uniquement : pages originales 7 (TF-IDF), 10 (déséquilibre), 13 (D/E), 22 (erreurs). Leurs anciens textes ne remplacent pas les chiffres ci-dessus.

### 13. D — DarijaBERT comme encodeur gelé
Page à produire dans le style illustré d’origine.
- DarijaBERT gelé. Variantes CLS, moyenne et séquence states.
- La tête states apprend 435 715 paramètres.
- Visuel : architecture pédagogique avec cadenas, états des tokens et tête apprenable.
- Références de style uniquement : pages originales 7 (TF-IDF), 10 (déséquilibre), 13 (D/E), 22 (erreurs). Leurs anciens textes ne remplacent pas les chiffres ci-dessus.

### 14. D — Résultats et durée de calcul
Page à produire dans le style illustré d’origine.
- Runs D : CLS 1 min 34 s, moyenne 1 min 11 s, states 42 de 1 min 59 s.
- States 42 inclut 64,18 s d’extraction train + validation. F1 test 0,571593.
- Total D : 6 min 43 s. Visuel : extraction, cache et apprentissage de la tête.
- Références de style uniquement : pages originales 7 (TF-IDF), 10 (déséquilibre), 13 (D/E), 22 (erreurs). Leurs anciens textes ne remplacent pas les chiffres ci-dessus.

### 15. E — Ajuster DarijaBERT sur le sentiment
Page à produire dans le style illustré d’origine.
- DarijaBERT-Arabizi : encodeur et tête ajustés, environ 170 M de paramètres.
- AdamW, taux 2 × 10⁻⁵, micro-batch 2, batch effectif 32, 128 tokens et cinq époques maximum.
- Visuel : architecture détaillée avec propagation des gradients et poids modifiés.
- Références de style uniquement : pages originales 7 (TF-IDF), 10 (déséquilibre), 13 (D/E), 22 (erreurs). Leurs anciens textes ne remplacent pas les chiffres ci-dessus.

### 16. E — Le fine-tuning concentre le temps
Page à produire dans le style illustré d’origine.
- Arabizi : 36 min 14 s (42), 37 min 02 s (43), 26 min 32 s (44, avec reprise).
- E1 standard + Arabizi : 2 h 51 min 10 s. F1 test du représentant : 0,608138.
- Visuel : entraînements parallèles, chronomètres et reprise d’un checkpoint.
- Références de style uniquement : pages originales 7 (TF-IDF), 10 (déséquilibre), 13 (D/E), 22 (erreurs). Leurs anciens textes ne remplacent pas les chiffres ci-dessus.

### 17. E2 — Une tâche auxiliaire ne garantit pas un gain
Page d’origine 14 conservée.
Image stricte conservée : `presentation/Darija_Sentiment_Soutenance_Resultats/origin_image/slide_14.png`.

### 18. E2 — Budget réduit et suite souhaitée
Page à produire dans le style illustré d’origine.
- λ 0,1 : trois époques, 23 min 08 s, F1 test 0,605137.
- λ 0,3 : trois époques, 24 min 19 s, F1 test 0,604073.
- Souhait futur : prolonger E2 avec arrêt anticipé et plusieurs graines. λ 0,7 reste à tester.
- Références de style uniquement : pages originales 7 (TF-IDF), 10 (déséquilibre), 13 (D/E), 22 (erreurs). Leurs anciens textes ne remplacent pas les chiffres ci-dessus.

### 19. Nettoyer sans effacer les indices : expérience P
Page d’origine 6 conservée.
Image stricte conservée : `presentation/Darija_Sentiment_Soutenance_Resultats/origin_image/slide_06.png`.

### 20. P — Temps mesurés et étude incomplète
Page à produire dans le style illustré d’origine.
- TF-IDF union sans signes : 14,98 s, F1 test 0,567396.
- DarijaBERT sans signes : 20 min 13 s, validation 0,761560, exclu du test.
- Visuel : nettoyage illustré et chronomètres. Étendre l’étude à tous les modèles reste un objectif.
- Références de style uniquement : pages originales 7 (TF-IDF), 10 (déséquilibre), 13 (D/E), 22 (erreurs). Leurs anciens textes ne remplacent pas les chiffres ci-dessus.

### 21. La sélection se fait avant la comparaison finale
Page d’origine 15 conservée.
Image stricte conservée : `presentation/Darija_Sentiment_Soutenance_Resultats/origin_image/slide_15.png`.

### 22. Résultats sur le test officiel
Page d’origine 16 conservée.
Image stricte conservée : `presentation/Darija_Sentiment_Soutenance_Resultats/origin_image/slide_16.png`.

### 23. Nos résultats face aux publications sur AfriSenti–ary
Page à produire dans le style illustré d’origine.
- Même corpus de référence : AfriSenti, darija marocaine, tâche A monolingue. Métrique : Weighted-F1 test en %.
- NLNDE, Wang et al. (2023) : 64,82. ABCD, Thin et al. (2023) : 61,54. UM6P, El Mahdaouy et al. (2023) : 60,15.
- Notre E DarijaBERT-Arabizi : 60,81 %, graine 42, sélection sur validation.
- Comparaison indicative : validation locale 494 exemples contre 1 216 annoncés dans le bilan AfriSenti, préparation et agrégation différentes. Aucun classement direct ni supériorité statistique.
- Visuel : tableau éditorial lisible, papiers scientifiques et encadré de prudence. Aucun graphique de valeurs synthétiques.
- Références de style uniquement : pages originales 7 (TF-IDF), 10 (déséquilibre), 13 (D/E), 22 (erreurs). Leurs anciens textes ne remplacent pas les chiffres ci-dessus.

### 24. Lire les erreurs du candidat, classe par classe
Page d’origine 17 conservée.
Image stricte conservée : `presentation/Darija_Sentiment_Soutenance_Resultats/origin_image/slide_17.png`.

### 25. Validation → test : une baisse à expliquer
Page d’origine 18 conservée.
Image stricte conservée : `presentation/Darija_Sentiment_Soutenance_Resultats/origin_image/slide_18.png`.

### 26. Écriture non vue : le transfert reste difficile
Page d’origine 19 conservée.
Image stricte conservée : `presentation/Darija_Sentiment_Soutenance_Resultats/origin_image/slide_19.png`.

### 27. Incertitude : situer le score, pas seulement le classer
Page d’origine 20 conservée.
Image stricte conservée : `presentation/Darija_Sentiment_Soutenance_Resultats/origin_image/slide_20.png`.

### 28. G — Qualité, latence CPU et taille des poids
Page d’origine 21 conservée.
Image stricte conservée : `presentation/Darija_Sentiment_Soutenance_Resultats/origin_image/slide_21.png`.

### 29. Temps enregistrés par famille
Page à produire dans le style illustré d’origine.
- 72 runs distincts, quatre transferts fixes B inclus.
- A 49 s, B 0 s local, C 20 min 08 s, D 6 min 43 s, E 3 h 38 min 37 s, P 20 min 28 s.
- Total enregistré 4 h 26 min 46 s. Coupures, téléchargements et calculs perdus non quantifiés.
- Références de style uniquement : pages originales 7 (TF-IDF), 10 (déséquilibre), 13 (D/E), 22 (erreurs). Leurs anciens textes ne remplacent pas les chiffres ci-dessus.

### 30. Quatre changements de session Colab
Page à produire dans le style illustré d’origine.
- Quatre changements de session Colab confirmés par l’auteur.
- Les sessions se terminaient avant la fin des calculs : relances, restauration et reprise.
- Exemple tracé : E Arabizi graine 44. Le temps perdu n’est pas mesuré.
- Visuel : session interrompue, câble coupé, checkpoint sauvegardé et cycle de reprise.
- Références de style uniquement : pages originales 7 (TF-IDF), 10 (déséquilibre), 13 (D/E), 22 (erreurs). Leurs anciens textes ne remplacent pas les chiffres ci-dessus.

### 31. Améliorations souhaitées
Page à produire dans le style illustré d’origine.
- Tous les modèles : comparer texte intact, sans emojis, sans ponctuation, sans les deux.
- A/C/D/E/E2 : réentraîner à budget comparable. B : évaluer les entrées perturbées, poids fixes.
- E2 : budget plus long, arrêt anticipé, plusieurs graines. Sauvegardes persistantes et reprise complète.
- Ces expériences restent à réaliser. Aucun score futur ni gain n’est inventé.
- Références de style uniquement : pages originales 7 (TF-IDF), 10 (déséquilibre), 13 (D/E), 22 (erreurs). Leurs anciens textes ne remplacent pas les chiffres ci-dessus.

### 32. Trente erreurs : comprendre avant de compter
Page d’origine 22 conservée.
Image stricte conservée : `presentation/Darija_Sentiment_Soutenance_Resultats/origin_image/slide_22.png`.

### 33. Du benchmark au laboratoire interactif
Page d’origine 23 conservée.
Image stricte conservée : `presentation/Darija_Sentiment_Soutenance_Resultats/origin_image/slide_23.png`.

### 34. Ce que nous retenons du projet
Page d’origine 24 conservée.
Image stricte conservée : `presentation/Darija_Sentiment_Soutenance_Resultats/origin_image/slide_24.png`.

### 35. Annexe — Lire les courbes de fine-tuning
Page d’origine 25 conservée.
Image stricte conservée : `presentation/Darija_Sentiment_Soutenance_Resultats/origin_image/slide_25.png`.

### 36. Références et guide pour les questions
Page d’origine 26 conservée.
Image stricte conservée : `presentation/Darija_Sentiment_Soutenance_Resultats/origin_image/slide_26.png`.

## Sources de la nouvelle comparaison

- Wang et al., NLNDE, tableau 2, colonne ma, page PDF 4 : https://aclanthology.org/2023.semeval-1.68.pdf
- Thin et al., ABCD, tableau 1, Track 7: Moroccan, page PDF 4 : https://aclanthology.org/2023.semeval-1.44.pdf
- El Mahdaouy et al., UM6P, tableau 4, colonne ma, page PDF 5 : https://aclanthology.org/2023.semeval-1.138.pdf
- Muhammad et al., bilan AfriSenti-SemEval, tableaux 1 et 3 : https://aclanthology.org/2023.semeval-1.315.pdf
- Remarque de traçabilité : le bilan officiel indique 64,83 pour NLNDE, alors que le papier de l’équipe donne 64,82. La page utilisera la valeur du papier, avec l’écart documenté dans les notes.
- Notre score exact : 0,608138. Conversion en pourcentage et arrondi : 60,81 %. Aucune nouvelle expérience ni aucun nouvel entraînement ne sera déclenché.


## Ajout demandé : questions et réponses

38 diapositives au total. Pages 1 à 36 conservées. Pages 37 et 38 : six questions et réponses par page, fondées sur les énoncés officiels de research/questions_officielles.md et les résultats actuels (sans reprendre les anciennes restrictions de périmètre).

- Q1 — Quelle métrique pour des classes déséquilibrées ?
  F1 pondéré pour le benchmark, Macro-F1 pour équilibrer les classes. Majorité seule : 0,160795 sur test. Détails : p. 3–4.

- Q2 — L’écriture crée-t-elle un raccourci ?
  Neutres latins : 52,3 % sur train, 29,9 % sur test. Cette corrélation change ; elle menace la généralisation. Détails : p. 5, 25–26.

- Q3 — Faut-il supprimer emojis et ponctuation ?
  Conserver les indices par défaut. P union sans signes : F1 test 0,567396. Ablation sur tous les modèles encore à réaliser. Détails : p. 19–20, 31.

- Q4 — Comment le NLI classe-t-il sans train local ?
  Comparer l’implication de trois hypothèses. F1 validation : arabe 0,501026 ; français 0,497245 ; anglais 0,463444. Supervision externe préalable. Détails : p. 8.

- Q5 — Pourquoi B2 diffère-t-il selon l’écriture ?
  CAMeLBERT : F1 test arabe 0,558 ; latin 0,144. Couverture de l’Arabizi limitée, cause non isolée ; contamination externe non certifiée. Détails : p. 9.

- Q6 — Comment traiter le déséquilibre ?
  Pondération, suréchantillonnage et décalages de décision sur validation. Suréchantillonner uniquement train après séparation, pour éviter les fuites. Détails : p. 10, 21.

- Q7 — Quel réseau C et que prouve l’attention ?
  BiLSTM + AraVec retenu : F1 test 0,493340. L’attention ne prouve pas une cause ; les courbes décrivent les époques observées. Détails : p. 11–12.

- Q8 — Quel gain et quel coût pour le fine-tuning ?
  D : 0,571593 ; E Arabizi : 0,608138, soit +3,65 points. Environ 170 M de paramètres ajustés ; run 42 : 36 min 14 s. Détails : p. 13–16, 21–22.

- Q9 — La tâche auxiliaire améliore-t-elle E2 ?
  F1 : λ 0,1 → 0,605137 ; λ 0,3 → 0,604073. Aucun gain établi face à E1 : une graine, trois époques, budgets différents. Détails : p. 17–18, 27.

- Q10 — Comment gérer le décalage validation–test ?
  Décision figée sur validation. L’optimum trouvé sur test reste un diagnostic. En production : suivi des dérives et recalibration sur de nouvelles données. Détails : p. 3, 21, 25.

- Q11 — Que se passe-t-il sur une écriture non vue ?
  TF-IDF : arabe → latin 0,2254 ; latin → arabe 0,2110. Couvrir les deux écritures et valider le domaine cible avant déploiement. Détails : p. 26, 34.

- Q12 — Quel modèle déployer et quelles limites ?
  E : meilleure validation, F1 test 0,608138 ; A : option légère à valider. Trente erreurs à relire ; bruit d’annotation non quantifié, checkpoint E à restaurer. Détails : p. 21–22, 28–29, 32–34.
