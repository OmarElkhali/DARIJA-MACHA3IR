# Darija-Sentiment — soutenance

La présentation comporte 24 diapositives, dont une annexe. Elle explique le protocole, les familles A–E, les résultats et les limites, avec les réponses officielles Q1–Q12 intégrées au récit.

## Utiliser la présentation

Ouvrir `Darija_Sentiment_Soutenance.pptx` dans PowerPoint ou un lecteur compatible. Les notes d'oral de chaque page sont intégrées au fichier et conservées dans `speech.md`. Les images complètes des pages se trouvent dans `origin_image/`.

Les diapositives sont des images générées en 16:9, selon le workflow codex-ppt demandé : les textes et graphiques ne sont pas individuellement éditables. Les notes restent éditables. Pour modifier un graphique, il faut régénérer l'image concernée à partir de ses sources et vérifier ses valeurs.

## Méthode pour l'oral

- Annoncer le problème avant de nommer les architectures.
- Pour chaque graphique, dire la partition, la mesure et la population, puis expliquer le constat et sa limite.
- Expliquer une méthode par une phrase simple, puis donner son résultat utile. Les détails de paramètres restent dans les notes.
- Réserver environ 20 à 25 minutes au parcours principal. Les notes complètes fournissent environ 3 200 mots ; les effectifs détaillés peuvent être gardés pour les questions du jury. La dernière page est une annexe.
- Préparer la démonstration de l'application à l'avance. Une phrase réussie reste une démonstration, pas une nouvelle mesure de benchmark.
- Terminer par le choix conditionnel d'un prototype et les validations encore nécessaires.

## Lire les réponses scientifiques

`research/questions_officielles.md`, à la racine du projet, conserve les énoncés. `answers.draft.md` contient le dossier préparatoire détaillé, et `evidence.draft.json` les chiffres qui ont servi à préparer les pages. `outline.md` indique où chaque question est traitée.

La campagne E complète n'est pas exécutée. Q9 reste théorique car le périmètre demandé est ary uniquement. Les catégories F6 sont provisoires et les désaccords entre annotateurs ne mesurent pas un taux de labels faux. Les analyses a posteriori de Q10 et le témoin de Q11 n'ont pas changé le candidat ni les checkpoints verrouillés.

## Production et traçabilité

- Direction visuelle : ivoire, vert forêt, sauge et terre cuite, typographie classique et schémas pédagogiques.
- Backend : outil de génération d'images intégré, sans fallback API/CLI.
- Autorisation : `approvals.json`, réalisation complète sans validations intermédiaires supplémentaires.
- Référence visuelle : diapositive 16, inspectée avant la production des autres pages.
- Contenu et prompts : `deck_spec.json`, `prompts/slide_XX.json`, `sample_prompt.txt`.
- Dispatch et provenance : `slide_jobs.json`, `slide_run_state.json`, enregistrements par les scripts fournis par codex-ppt.
- Contrôles finaux : `qa_report.json` après assemblage.

Les slides sont vérifiées visuellement en tant qu'images et le fichier PPTX est contrôlé après assemblage. Ne pas assimiler ces contrôles à une inspection native dans PowerPoint si elle n'a pas été réalisée.

Ce style peut être conservé dans une bibliothèque personnelle si l'utilisateur le demande ; aucun enregistrement personnel n'est effectué automatiquement.
