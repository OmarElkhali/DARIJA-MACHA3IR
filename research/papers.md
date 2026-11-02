# Recherche documentaire — TP2 Darija

Vérification : 29 octobre 2026. Recherche avant implémentation sur ACL Anthology, arXiv, Springer, ScienceDirect, Hugging Face et le dépôt officiel. Google Scholar a été interrogé mais son accès automatisé a échoué ; aucune consultation réussie n'est revendiquée. Recherche ciblée 2024–2026, non exhaustive ; aucune référence 2026 directement pertinente retenue après vérification. Les synthèses reposent sur les résumés officiels et métadonnées accessibles ; pas sur une relecture intégrale de chaque PDF.

Le préprint DarijaBERT date de 2023, la publication en ligne de 2024 et le volume de 2025. Ces dates ne sont pas contradictoires.

## AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages

**Auteurs et année :** Shamsuddeen Hassan Muhammad et al. — 2023

**Problème :** Manque de corpus de sentiment pour les langues africaines.

**Solution :** Tweets annotés par des locuteurs natifs, trois sentiments, partitions de référence.

**Résultat scientifique des auteurs :** Benchmark de plus de 110 000 tweets dans 14 langues, dont ary et arq ; baselines propres à ces corpus.

**Application au TP :** Employer exclusivement les partitions marocaines ary et mesurer les divergences entre écritures.

**Expérience proposée :** Comparer les cinq familles sur ary sans déplacer de tweets entre partitions.

**Source primaire :** https://arxiv.org/abs/2302.08956

## SemEval-2023 Task 12: Sentiment Analysis for African Languages (AfriSenti-SemEval)

**Auteurs et année :** Shamsuddeen Hassan Muhammad, Idris Abdulmumin, Seid Muhie Yimam, David Ifeoluwa Adelani, Ibrahim Said Ahmad, Nedjma Ousidhoum, Abinew Ali Ayele, Saif Mohammad, Meriem Beloucif et Sebastian Ruder — 2023

**Problème :** Comparer des systèmes monolingues, multilingues et de transfert sans ressources.

**Solution :** Évaluation partagée avec Weighted-F1.

**Résultat scientifique des auteurs :** NLNDE premier en A/B et UCAS-IIE-NLP premier en C ; classements agrégés, pas une promesse de F1 marocain.

**Application au TP :** Critère principal Weighted-F1, Macro-F1 complémentaire pour les classes moins représentées.

**Expérience proposée :** Choisir sur dev, verrouiller puis seulement évaluer test.

**Source primaire :** https://aclanthology.org/2023.semeval-1.315/

## DarijaBERT: a step forward in NLP for the written Moroccan dialect

**Auteurs et année :** Kamel Gaanoun, Abdou Mohamed Naira, Anass Allak et Imade Benelallam — 2024 en ligne ; volume 2025 ; prépublication 2023

**Problème :** Les modèles arabes représentent peu la darija latine.

**Solution :** Préentraînement dialectal avec variantes arabe, Arabizi et mixte.

**Résultat scientifique des auteurs :** Comparaison sur quatre tâches en darija, dont la classification thématique MTCD ; les conclusions du papier ne constituent pas notre résultat AfriSenti.

**Application au TP :** Comparer les deux checkpoints demandés avec mêmes graines, batch et budget.

**Expérience proposée :** D : encodeur gelé ; E : fine-tuning uniquement marocain ary.

**Source primaire :** https://link.springer.com/article/10.1007/s41060-023-00498-2

## Arabizi Language Models for Sentiment Analysis

**Auteurs et année :** Gaétan Baert, Souhir Gahbiche, Guillaume Gadek et Alexandre Pauchet — 2020

**Problème :** Graphies Arabizi variables, peu de modèles adaptés.

**Solution :** BAERT préentraîné sur LAD, corpus de 7,7 millions de tweets ; évaluation sur SALAD annoté.

**Résultat scientifique des auteurs :** BAERT dépasse mBERT sur SALAD. Ce corpus n'est pas le test marocain AfriSenti.

**Application au TP :** L'adéquation de l'écriture est une hypothèse expérimentale, pas une garantie.

**Expérience proposée :** F3 : décomposer F1 arabe/latin/mixte ; E1 : tester DarijaBERT-Arabizi.

**Source primaire :** https://aclanthology.org/2020.coling-main.51/

## Enriching Word Vectors with Subword Information

**Auteurs et année :** Piotr Bojanowski, Edouard Grave, Armand Joulin et Tomas Mikolov — 2017

**Problème :** Un vecteur indépendant par mot traite mal les mots rares et inconnus.

**Solution :** Représenter un mot par des n-grammes de caractères appris par skip-gram.

**Résultat scientifique des auteurs :** Évaluations de similarité et analogies sur neuf langues ; pas une validation directe de sentiment en darija.

**Application au TP :** Entraîner FastText (Gensim) uniquement sur les tweets train, incluant Arabizi.

**Expérience proposée :** Comparer embeddings aléatoires/AraVec/fastText ; mesurer OOV et couverture.

**Source primaire :** https://aclanthology.org/Q17-1010/

## Character-level Convolutional Networks for Text Classification

**Auteurs et année :** Xiang Zhang, Junbo Zhao et Yann LeCun — 2015

**Problème :** Dépendance à la segmentation en mots dans la classification de textes.

**Solution :** Convolutions directement sur les caractères.

**Résultat scientifique des auteurs :** Résultats compétitifs sur de grands corpus de classification ; pas de preuve spécifique sur les petits corpus darija.

**Application au TP :** C4 compact adapté au TP, inspiré du principe et non une reproduction exacte du réseau publié.

**Expérience proposée :** Comparer char-CNN et TF-IDF caractères avec les mêmes partitions.

**Source primaire :** https://arxiv.org/abs/1509.01626

## Attention Is All You Need

**Auteurs et année :** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser et Illia Polosukhin — 2017

**Problème :** Calcul séquentiel coûteux des architectures récurrentes.

**Solution :** Auto-attention multi-têtes et information de position.

**Résultat scientifique des auteurs :** Expériences de traduction WMT 2014 anglais-allemand et anglais-français ; pas un score de sentiment.

**Application au TP :** Mini-Transformer deux couches, dimension 128, quatre têtes.

**Expérience proposée :** Ablation avec/sans position, sans modifier le reste du protocole.

**Source primaire :** https://arxiv.org/abs/1706.03762

## A systematic literature review of Arabic dialect sentiment analysis

**Auteurs et année :** Yassir Matrane, Faouzia Benabbou et Nawal Sael — 2023

**Problème :** Hétérogénéité des dialectes, ressources, nettoyages et protocoles.

**Solution :** Revue de 75 études publiées entre 2012 et 2022.

**Résultat scientifique des auteurs :** Synthèse des difficultés de normalisation et de comparaison entre jeux différents ; pas un nouveau modèle gagnant universel.

**Application au TP :** Conserver négation, chiffres Arabizi et français ; tester plutôt que supposer l'utilité du nettoyage.

**Expérience proposée :** F4 : retirer emojis et ponctuation répétée après verrouillage ; rapporter les changements.

**Source primaire :** https://doi.org/10.1016/j.jksuci.2023.101570

## NLNDE at SemEval-2023 Task 12: Adaptive Pretraining and Source Language Selection for Low-Resource Multilingual Sentiment Analysis

**Auteurs et année :** Mingyang Wang, Heike Adel, Lukas Lange, Jannik Strötgen et Hinrich Schütze — 2023

**Problème :** Le transfert peut dégrader une langue cible si les langues sources sont mal choisies.

**Solution :** Préentraînement adapté et sélection explicite des langues sources, ensembles de graines.

**Résultat scientifique des auteurs :** Huit pistes sur quinze remportées, meilleur système multilingue AfriSenti ; le bénéfice d'une autre langue n'est pas universel.

**Application au TP :** Contexte scientifique seulement : le projet demandé est exclusivement ary.

**Expérience proposée :** Aucun transfert entre pays ; F2 mesure uniquement le transfert entre écritures au sein de ary.

**Source primaire :** https://aclanthology.org/2023.semeval-1.68/

## UCAS-IIE-NLP at SemEval-2023 Task 12: Enhancing Generalization of Multilingual BERT for Low-resource Sentiment Analysis

**Auteurs et année :** Dou Hu, Lingwei Wei, Yaxin Liu, Wei Zhou et Songlin Hu — 2023

**Problème :** Faible généralisation vers des langues peu annotées.

**Solution :** SACL-XLMR, adaptation lexicale et apprentissage contrastif adversarial supervisé.

**Résultat scientifique des auteurs :** Première place de la sous-tâche zero-shot AfriSenti ; système différent du NLI B1.

**Application au TP :** Le zero-shot doit être évalué par langue/écriture, sans confondre transfert et absence de supervision antérieure.

**Expérience proposée :** B1 trois langues d'hypothèses ; F2 transfert entre écritures.

**Source primaire :** https://aclanthology.org/2023.semeval-1.255/

## Atlas-Chat: Adapting Large Language Models for Low-Resource Moroccan Arabic Dialect

**Auteurs et année :** Guokan Shang, Hadi Abdine, Yousef Khoubrane, Amr Mohamed, Yassine Abbahaddou, Sofiane Ennadir, Imane Momayiz, Xuguang Ren, Eric Moulines, Preslav Nakov, Michalis Vazirgiannis et Eric Xing — 2025

**Problème :** Faible suivi des instructions en darija par les grands modèles.

**Solution :** Instruction-tuning spécifique et suite d'évaluation en darija.

**Résultat scientifique des auteurs :** Modèles 2B/9B/27B évalués sur des tâches discriminatives et génératives ; résultats non interchangeables avec notre protocole.

**Application au TP :** Contexte récent sur l'adaptation dialectale, sans ajouter une sixième famille.

**Expérience proposée :** Perspective hors benchmark : audit de contamination avant toute comparaison LLM.

**Source primaire :** https://aclanthology.org/2025.loreslm-1.2/

## Cartes des modèles et ressources

- NLI : https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
- CAMeLBERT : https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment
- DarijaBERT : https://huggingface.co/SI2M-Lab/DarijaBERT
- Arabizi : https://huggingface.co/SI2M-Lab/DarijaBERT-arabizi
- AraVec (format Gensim ; téléchargement manuel volumineux) : https://github.com/bakrianoo/aravec
- Données et votes : https://github.com/afrisenti-semeval/afrisent-semeval-2023

CAMeLBERT est déjà supervisé sur ASTD, ArSAS et SemEval : zero-shot **sur ary local**, pas modèle sans supervision sentiment. Sans identifiants/textes complets des corpus de préentraînement, l'absence de contamination ne peut pas être certifiée. DarijaBERT a notamment vu des tweets au préentraînement. Un audit des seules partitions locales ne prouve pas l'absence de contamination externe.
