# Direction visuelle — Darija / Sentiment

L’interface conserve Streamlit et les modèles existants. Elle remplace la présentation utilitaire initiale par une composition éditoriale classique, adaptée à un projet universitaire. Les trois références ont été créées avec l’outil **image_gen intégré** avant l’implémentation ; les prompts exacts sont dans [prompts.json](prompts.json).

## Références

- [Vue d’ensemble](overview-reference.png) : priorité au titre, à la lecture des résultats et au parcours scientifique.
- [Laboratoire](laboratory-reference.png) : formulaire à gauche, sentiment et scores à droite, puis comparaison et lecture des tokens.
- [Modèles](models-reference.png) : une explication associée à une architecture, aux mesures dev et aux courbes.

Ces images sont des maquettes, pas des preuves expérimentales. Leurs libellés ou valeurs illustratives ne remplacent jamais les CSV du projet. L’interface utilise les vrais noms de modèles et les résultats enregistrés, notamment lorsque la maquette emploie un exemple approximatif.

## Analyse et traduction en interface

**Hiérarchie.** Les références opposent une grande typographie de titre à des textes courts et des légendes discrètes. La page d’accueil garde un seul message principal, deux actions et une composition asymétrique avec le mot arabe « مشاعر ». Le laboratoire privilégie d’abord l’entrée et la prédiction, avant les détails.

**Typographie.** Georgia pour les titres et les grands nombres ; Segoe UI/Arial pour les textes, contrôles et graphiques ; Tahoma pour le motif arabe. Les polices système évitent un téléchargement externe et gardent un rendu lisible hors ligne. Titres adaptatifs autour de 42–75 px, textes courants autour de 16 px, gouttières de 16–42 px selon l’écran.

**Couleurs.** Ivoire `#F6F3EC`, vert profond `#173D36`, sauge `#91A58C`, filets gris chauds `#D9D9CE`. La terre cuite sert aux sentiments négatifs et à certaines comparaisons sémantiques. Aucun score n’est codé par une couleur sans libellé.

**Quatre éléments distinctifs.** Grand titre avec motif arabe ; registre de quatre indicateurs séparés par des filets ; graphiques ouverts associés à une progression scientifique ; laboratoire partagé entre saisie et lecture des scores. Les surfaces de formulaire et de résultat ont une fonction ; les autres sections restent ouvertes, sans empilement de boîtes.

**Deux signes de mouvement.** Apparition légère des introductions et retour tactile des boutons/navigation. Les graphiques ajoutent une interaction utile au survol et au zoom. `prefers-reduced-motion` désactive les transitions et animations décoratives.

**Interaction.** Navigation supérieure à six espaces, recherche et filtres, aperçus contextuels, exemples réutilisables, prédictions persistantes dans la session, chargement modèle par modèle, gestion des erreurs, téléchargements effectifs et parcours de soutenance. Les graines restent distinctes dans les graphiques comparatifs.

**Accessibilité.** Contrôles Streamlit natifs avec libellés, focus visible, navigation clavier, textes de sentiment explicites, HTML échappé pour les phrases, direction automatique pour l’arabe, adaptation des colonnes/espacements sur petit écran.

## Vérification

`tests/check_app.py` vérifie les six espaces, les cinq fiches de familles, les textes test sans labels, les filtres, le passage corpus→laboratoire, les prédictions réelles A/B/C/D, les erreurs de saisie/checkpoint, l’historique et le mode soutenance. Les modèles et le verrouillage du benchmark restent inchangés.

La vérification visuelle dans le navigateur n’a pas pu être menée à terme : le contrôle navigateur intégré était indisponible, puis le contrôle Windows s’est arrêté faute de pouvoir identifier l’URL avec assez de certitude. Aucun autre outil d’automatisation de navigateur n’a été utilisé pour contourner cet arrêt. Les tests AppTest et le service HTTP ont été vérifiés séparément.
