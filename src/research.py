"""Bibliographie vérifiée le 29 octobre 2026 ; aucun score emprunté au TP."""
from .preprocess import ROOT
import pandas as pd

# Les résultats ci-dessous sont ceux des auteurs. Aucun n'est un résultat local.
PAPERS = [
    dict(id="afrisenti", title="AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages",
         authors="Shamsuddeen Hassan Muhammad et al.", year="2023", source="https://arxiv.org/abs/2302.08956",
         problem="Manque de corpus de sentiment pour les langues africaines.",
         solution="Tweets annotés par des locuteurs natifs, trois sentiments, partitions de référence.",
         result="Benchmark de plus de 110 000 tweets dans 14 langues, dont ary et arq ; baselines propres à ces corpus.",
         application="Employer exclusivement les partitions marocaines ary et mesurer les divergences entre écritures.",
         experiment="Comparer les cinq familles sur ary sans déplacer de tweets entre partitions."),
    dict(id="semeval", title="SemEval-2023 Task 12: Sentiment Analysis for African Languages (AfriSenti-SemEval)",
         authors="Shamsuddeen Hassan Muhammad, Idris Abdulmumin, Seid Muhie Yimam, David Ifeoluwa Adelani, Ibrahim Said Ahmad, Nedjma Ousidhoum, Abinew Ali Ayele, Saif Mohammad, Meriem Beloucif et Sebastian Ruder", year="2023", source="https://aclanthology.org/2023.semeval-1.315/",
         problem="Comparer des systèmes monolingues, multilingues et de transfert sans ressources.",
         solution="Évaluation partagée avec Weighted-F1.",
         result="NLNDE premier en A/B et UCAS-IIE-NLP premier en C ; classements agrégés, pas une promesse de F1 marocain.",
         application="Critère principal Weighted-F1, Macro-F1 complémentaire pour les classes moins représentées.",
         experiment="Choisir sur dev, verrouiller puis seulement évaluer test."),
    dict(id="darijabert", title="DarijaBERT: a step forward in NLP for the written Moroccan dialect",
         authors="Kamel Gaanoun, Abdou Mohamed Naira, Anass Allak et Imade Benelallam", year="2024 en ligne ; volume 2025 ; prépublication 2023", source="https://link.springer.com/article/10.1007/s41060-023-00498-2",
         problem="Les modèles arabes représentent peu la darija latine.",
         solution="Préentraînement dialectal avec variantes arabe, Arabizi et mixte.",
         result="Comparaison sur quatre tâches en darija, dont la classification thématique MTCD ; les conclusions du papier ne constituent pas notre résultat AfriSenti.",
         application="Comparer les deux checkpoints demandés avec mêmes graines, batch et budget.",
         experiment="D : encodeur gelé ; E : fine-tuning uniquement marocain ary."),
    dict(id="baert", title="Arabizi Language Models for Sentiment Analysis",
         authors="Gaétan Baert, Souhir Gahbiche, Guillaume Gadek et Alexandre Pauchet", year="2020", source="https://aclanthology.org/2020.coling-main.51/",
         problem="Graphies Arabizi variables, peu de modèles adaptés.",
         solution="BAERT préentraîné sur LAD, corpus de 7,7 millions de tweets ; évaluation sur SALAD annoté.",
         result="BAERT dépasse mBERT sur SALAD. Ce corpus n'est pas le test marocain AfriSenti.",
         application="L'adéquation de l'écriture est une hypothèse expérimentale, pas une garantie.",
         experiment="F3 : décomposer F1 arabe/latin/mixte ; E1 : tester DarijaBERT-Arabizi."),
    dict(id="fasttext", title="Enriching Word Vectors with Subword Information",
         authors="Piotr Bojanowski, Edouard Grave, Armand Joulin et Tomas Mikolov", year="2017", source="https://aclanthology.org/Q17-1010/",
         problem="Un vecteur indépendant par mot traite mal les mots rares et inconnus.",
         solution="Représenter un mot par des n-grammes de caractères appris par skip-gram.",
         result="Évaluations de similarité et analogies sur neuf langues ; pas une validation directe de sentiment en darija.",
         application="Entraîner FastText (Gensim) uniquement sur les tweets train, incluant Arabizi.",
         experiment="Comparer embeddings aléatoires/AraVec/fastText ; mesurer OOV et couverture."),
    dict(id="charcnn", title="Character-level Convolutional Networks for Text Classification",
         authors="Xiang Zhang, Junbo Zhao et Yann LeCun", year="2015", source="https://arxiv.org/abs/1509.01626",
         problem="Dépendance à la segmentation en mots dans la classification de textes.",
         solution="Convolutions directement sur les caractères.",
         result="Résultats compétitifs sur de grands corpus de classification ; pas de preuve spécifique sur les petits corpus darija.",
         application="C4 compact adapté au TP, inspiré du principe et non une reproduction exacte du réseau publié.",
         experiment="Comparer char-CNN et TF-IDF caractères avec les mêmes partitions."),
    dict(id="transformer", title="Attention Is All You Need",
         authors="Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser et Illia Polosukhin", year="2017", source="https://arxiv.org/abs/1706.03762",
         problem="Calcul séquentiel coûteux des architectures récurrentes.",
         solution="Auto-attention multi-têtes et information de position.",
         result="Expériences de traduction WMT 2014 anglais-allemand et anglais-français ; pas un score de sentiment.",
         application="Mini-Transformer deux couches, dimension 128, quatre têtes.",
         experiment="Ablation avec/sans position, sans modifier le reste du protocole."),
    dict(id="review", title="A systematic literature review of Arabic dialect sentiment analysis",
         authors="Yassir Matrane, Faouzia Benabbou et Nawal Sael", year="2023", source="https://doi.org/10.1016/j.jksuci.2023.101570",
         problem="Hétérogénéité des dialectes, ressources, nettoyages et protocoles.",
         solution="Revue de 75 études publiées entre 2012 et 2022.",
         result="Synthèse des difficultés de normalisation et de comparaison entre jeux différents ; pas un nouveau modèle gagnant universel.",
         application="Conserver négation, chiffres Arabizi et français ; tester plutôt que supposer l'utilité du nettoyage.",
         experiment="F4 : retirer emojis et ponctuation répétée après verrouillage ; rapporter les changements."),
    dict(id="nlnde", title="NLNDE at SemEval-2023 Task 12: Adaptive Pretraining and Source Language Selection for Low-Resource Multilingual Sentiment Analysis",
         authors="Mingyang Wang, Heike Adel, Lukas Lange, Jannik Strötgen et Hinrich Schütze", year="2023", source="https://aclanthology.org/2023.semeval-1.68/",
         problem="Le transfert peut dégrader une langue cible si les langues sources sont mal choisies.",
         solution="Préentraînement adapté et sélection explicite des langues sources, ensembles de graines.",
         result="Huit pistes sur quinze remportées, meilleur système multilingue AfriSenti ; le bénéfice d'une autre langue n'est pas universel.",
         application="Contexte scientifique seulement : le projet demandé est exclusivement ary.",
         experiment="Aucun transfert entre pays ; F2 mesure uniquement le transfert entre écritures au sein de ary."),
    dict(id="ucas", title="UCAS-IIE-NLP at SemEval-2023 Task 12: Enhancing Generalization of Multilingual BERT for Low-resource Sentiment Analysis",
         authors="Dou Hu, Lingwei Wei, Yaxin Liu, Wei Zhou et Songlin Hu", year="2023", source="https://aclanthology.org/2023.semeval-1.255/",
         problem="Faible généralisation vers des langues peu annotées.",
         solution="SACL-XLMR, adaptation lexicale et apprentissage contrastif adversarial supervisé.",
         result="Première place de la sous-tâche zero-shot AfriSenti ; système différent du NLI B1.",
         application="Le zero-shot doit être évalué par langue/écriture, sans confondre transfert et absence de supervision antérieure.",
         experiment="B1 trois langues d'hypothèses ; F2 transfert entre écritures."),
    dict(id="atlas", title="Atlas-Chat: Adapting Large Language Models for Low-Resource Moroccan Arabic Dialect",
         authors="Guokan Shang, Hadi Abdine, Yousef Khoubrane, Amr Mohamed, Yassine Abbahaddou, Sofiane Ennadir, Imane Momayiz, Xuguang Ren, Eric Moulines, Preslav Nakov, Michalis Vazirgiannis et Eric Xing", year="2025", source="https://aclanthology.org/2025.loreslm-1.2/",
         problem="Faible suivi des instructions en darija par les grands modèles.",
         solution="Instruction-tuning spécifique et suite d'évaluation en darija.",
         result="Modèles 2B/9B/27B évalués sur des tâches discriminatives et génératives ; résultats non interchangeables avec notre protocole.",
         application="Contexte récent sur l'adaptation dialectale, sans ajouter une sixième famille.",
         experiment="Perspective hors benchmark : audit de contamination avant toute comparaison LLM."),
]


def papers_table():
    return pd.DataFrame(PAPERS)


def write_research():
    folder = ROOT / "research"
    folder.mkdir(exist_ok=True)
    intro = "# Recherche documentaire — TP2 Darija\n\nVérification : 29 octobre 2026. Recherche avant implémentation sur ACL Anthology, arXiv, Springer, ScienceDirect, Hugging Face et le dépôt officiel. Google Scholar a été interrogé mais son accès automatisé a échoué ; aucune consultation réussie n'est revendiquée. Recherche ciblée 2024–2026, non exhaustive ; aucune référence 2026 directement pertinente retenue après vérification. Les synthèses reposent sur les résumés officiels et métadonnées accessibles ; pas sur une relecture intégrale de chaque PDF.\n\nLe préprint DarijaBERT date de 2023, la publication en ligne de 2024 et le volume de 2025. Ces dates ne sont pas contradictoires.\n\n"
    for p in PAPERS:
        intro += f"## {p['title']}\n\n"
        for label, key in [("Auteurs et année", "authors"), ("Problème", "problem"), ("Solution", "solution"), ("Résultat scientifique des auteurs", "result"), ("Application au TP", "application"), ("Expérience proposée", "experiment"), ("Source primaire", "source")]:
            intro += f"**{label} :** {p[key]}{' — ' + p['year'] if key == 'authors' else ''}\n\n"
    intro += "## Cartes des modèles et ressources\n\n- NLI : https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7\n- CAMeLBERT : https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment\n- DarijaBERT : https://huggingface.co/SI2M-Lab/DarijaBERT\n- Arabizi : https://huggingface.co/SI2M-Lab/DarijaBERT-arabizi\n- AraVec (format Gensim ; téléchargement manuel volumineux) : https://github.com/bakrianoo/aravec\n- Données et votes : https://github.com/afrisenti-semeval/afrisent-semeval-2023\n\nCAMeLBERT est déjà supervisé sur ASTD, ArSAS et SemEval : zero-shot **sur ary local**, pas modèle sans supervision sentiment. Sans identifiants/textes complets des corpus de préentraînement, l'absence de contamination ne peut pas être certifiée. DarijaBERT a notamment vu des tweets au préentraînement. Un audit des seules partitions locales ne prouve pas l'absence de contamination externe.\n"
    (folder / "papers.md").write_text(intro, encoding="utf-8")
    return papers_table()


def solutions_table():
    from .evaluate import results_table
    from .preprocess import load_split
    results = results_table()
    rows = []
    train = load_split("train")
    specs = [
        ("Graphies variables", f"{train.text.nunique()} textes nettoyés distincts sur {len(train)}", "fasttext", "A", "word", "char", "TF-IDF caractères"),
        ("Mots rares / OOV", "OOV mesurés dans le notebook avec vocabulaire train", "fasttext", "C", "attention_random", "attention_fasttext", "Embeddings fastText train"),
        ("Déséquilibre", str(train.label.value_counts().to_dict()), "review", "C", "attention_random", "attention_random_weighted", "Perte pondérée"),
        ("Rééchantillonnage", "Aucun exemple dev/test ajouté au train", "review", "C", "attention_random", "attention_random_oversampling", "Oversampling train uniquement"),
        ("Ordre des mots", "Architecture 128, 4 têtes, 2 couches", "transformer", "C", "transformer_random_nopos", "transformer_random", "Encodage positionnel"),
        ("Adaptation Arabizi", str(train.script.value_counts().to_dict()), "darijabert", "E", "darijabert", "arabizi", "Fine-tuning deux checkpoints"),
    ]
    for problem, observation, paper, family, reference, candidate, solution in specs:
        subset = results.loc[(results.family == family) & (results.split == "dev") & (results.decision == "dev_offsets")] if not results.empty else results
        base = subset.loc[subset.variant == reference, "weighted_f1"]
        trial = subset.loc[subset.variant == candidate, "weighted_f1"]
        if len(base) and len(trial):
            delta = trial.mean() - base.mean()
            status = "Testée avec amélioration" if delta > 0 else "Testée sans amélioration"
            result = f"Dev : {base.mean():.4f} → {trial.mean():.4f}, delta={delta:+.4f} ; graines={len(base)}/{len(trial)}"
        else:
            status, result = "Déjà implémentée ; non exécutée ou comparaison incomplète", "Non mesuré"
        source = next(p["source"] for p in PAPERS if p["id"] == paper)
        rows.append({"Problème": problem, "Observation mesurée": observation, "Publication": source, "Solution proposée": solution, "Résultat après essai": result, "Statut": status})
    rows.append({"Problème": "Mélange arabe/français", "Observation mesurée": "Écriture mixte ≠ français identifié ; validation humaine nécessaire", "Publication": next(p["source"] for p in PAPERS if p["id"] == "review"), "Solution proposée": "Nettoyage conservateur", "Résultat après essai": "Implémenté, sans ablation causale de langue", "Statut": "Déjà implémentée"})
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "results/research_solutions.csv", index=False)
    header = "| " + " | ".join(table.columns) + " |\n| " + " | ".join(["---"] * len(table.columns)) + " |\n"
    body = "\n".join("| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in row) + " |" for row in table.itertuples(index=False, name=None))
    (ROOT / "research/solutions.md").write_text("# Problèmes et solutions inspirées de la recherche\n\nComparaisons descriptives sur dev ; ni test causal ni garantie de gain sur test. Les valeurs sont calculées à partir des runs disponibles.\n\n" + header + body, encoding="utf-8")
    return table


def oral_report():
    """Douze thèmes pédagogiques provisoires, sans inventer les questions officielles."""
    from .evaluate import results_table, select_on_dev
    from .train import plan_table
    from .preprocess import load_split
    table = results_table()
    dev = table.loc[(table.split == "dev") & (table.decision == "dev_offsets")] if len(table) else table
    best = select_on_dev().head(1).to_dict("records") if len(dev) else []
    train = load_split("train")
    themes = [
        ("Données et protocole", "Une partition est un rôle expérimental : train apprend, dev choisit, test évalue après verrouillage.", f"Train={len(train)}, dev={len(load_split('dev'))}, test textes={len(load_split('test'))}.", "results/audit/counts.csv", "Fuites entre partitions et doublons.", "Partitions officielles inchangées, audit texte et empreintes SHA-256.", "Un audit local ne garantit pas l'absence de contamination externe.", "afrisenti"),
        ("Écritures et nettoyage", "Latin, arabe et mixte décrivent l'alphabet, pas la langue exacte.", str(train.script.value_counts().to_dict()), "figures/scripts.png", "Arabizi et français peuvent se ressembler.", "Conserver chiffres, accents, négations et emojis ; signaler les heuristiques.", "Une annotation humaine est nécessaire pour identifier le code-switching français.", "baert"),
        ("TF-IDF et n-grammes", "TF-IDF donne plus de poids aux fragments discriminants ; les caractères partagent des fragments entre graphies.", str(dev.loc[dev.family == 'A', ['variant', 'weighted_f1']].groupby('variant').mean().round(4).to_dict()) if len(dev) else "Non mesuré.", "results.csv : A / dev", "Orthographe variable.", "Comparer mots, caractères et combinaison avec régression logistique pondérée.", "Un écart dev ne prouve pas une amélioration générale.", "review"),
        ("Zero-shot", "Le NLI compare le tweet à trois hypothèses ; CAMeLBERT connaît déjà une tâche sentiment externe.", str(plan_table().query("family == 'B'")[["variant", "status"]].to_dict('records')), "results.csv : B / dev", "Effet de la langue de l'hypothèse et contamination possible.", "Comparer arabe, français, anglais ; scores par écriture.", "Sans run B, aucune performance B ne peut être annoncée.", "ucas"),
        ("Embeddings et OOV", "Un embedding représente un mot par des nombres ; OOV signifie absent du vocabulaire appris sur train.", "Voir oov.csv et embedding_coverage.json ; absence de fichier = non mesuré.", "results/oov.csv", "Mots rares, formes inédites.", "Comparer hasard, AraVec et fastText entraîné exclusivement sur train.", "Le vocabulaire min_freq=2 garde UNK pour les tokens hors vocabulaire, même si fastText sait composer un vecteur ; cette version compare l'initialisation des mots du vocabulaire.", "fasttext"),
        ("BiLSTM et attention", "BiLSTM lit les deux directions ; l'attention apprend une pondération des états.", str(plan_table().query("family == 'C' and variant.str.startswith('attention')", engine='python')[["variant", "status"]].to_dict('records')), "figures/attention_C.png", "Padding et interprétation abusive des poids.", "Masquer le padding et visualiser cinq vrais exemples dev.", "Les poids d'attention sont descriptifs, pas une preuve causale.", "review"),
        ("Mini-Transformer et CNN", "Le Transformer met les mots en relation ; le CNN détecte des motifs locaux de caractères.", str(plan_table().query("family == 'C' and (variant.str.startswith('transformer') or variant.str.startswith('charcnn'))", engine='python')[["variant", "status"]].to_dict('records')), "figures/curves_C.png", "Perte d'ordre sans positions et bruit orthographique.", "Deux couches, dimension 128, quatre têtes, ablation des positions et char-CNN.", "La taille du corpus peut favoriser les modèles simples.", "transformer"),
        ("Encodeur gelé", "On extrait des représentations sans changer les paramètres BERT ; seule la tête apprend.", str(plan_table().query("family == 'D'")[["variant", "status"]].to_dict('records')), "results.csv : D / dev", "Coût d'extraction et mémoire des états cachés.", "Comparer CLS, moyenne masquée et BiLSTM-attention ; caches identifiés.", "Le nombre de paramètres entraînables n'est pas le nombre total de paramètres utilisés.", "darijabert"),
        ("Fine-tuning sur ary", "Le fine-tuning ajuste aussi l'encodeur aux labels marocains.", str(plan_table().query("family == 'E'")[["variant", "status"]].to_dict('records')), "results.csv : E / dev", "Coût CPU et équité entre checkpoints.", "AdamW 2e-5, batch effectif 32, 3 époques, warmup 10 %, mêmes graines ; aucun corpus d'un autre pays.", "Un test de forme ne remplace pas un entraînement complet.", "darijabert"),
        ("Déséquilibre et décision", "Une règle de décision transforme les scores en classe ; ses réglages appartiennent à dev.", "Décalages réels dans models/*/meta.json ; ablations dans research_solutions.csv.", "results/research_solutions.csv", "Optimiser le Weighted-F1 peut masquer une classe faible.", "Rapporter Macro-F1, perte pondérée, oversampling train et deux décalages relatifs dev.", "Les scores bruts ne sont pas des probabilités calibrées ; le test ne règle pas les décalages.", "review"),
        ("Robustesse et incertitude", "Bootstrap mesure la variabilité d'échantillonnage ; trois graines mesurent celle de l'entraînement.", "Résultats disponibles dans comparison.csv, f2_cross_script.csv, f4_perturbations.csv et f5_mcnemar.json ; fichier absent = non exécuté.", "results/comparison.csv", "Changement de distribution, emojis et erreurs humaines.", "F1–F6, 1 000 rééchantillonnages, IC 95 %, McNemar exact et 30 erreurs à valider.", "McNemar porte sur les taux d'erreur appariés ; un IC bootstrap conditionne sur le modèle entraîné.", "nlnde"),
        ("Choix final et limites", "Le candidat scientifique est choisi sur dev avant la lecture des métriques test.", str(best), "results/selection_lock.json et figures/modele_final.png", "Choisir après avoir vu test surestime la qualité.", "Verrouiller variantes, règles et empreintes des checkpoints ; comparer cinq familles en laissant vides les familles non exécutées.", "Le meilleur parmi les runs disponibles n'est pas le meilleur de toutes les méthodes possibles.", "semeval"),
    ]
    text = "# Préparation à l'oral — Groupe 12\n\n**Les énoncés officiels Q1 à Q12 ne sont pas fournis. Les douze thèmes ci-dessous sont provisoires et ne prétendent pas reproduire les questions officielles. Fournir le sujet permettra la correspondance exacte.**\n\n## Présentation personnelle proposée\n\nDans ce projet, j'étudie trois sentiments sur des tweets marocains en écritures arabe, latine et mixte. Je compare cinq familles, du TF-IDF au fine-tuning. Je sépare apprentissage, validation et test ; je distingue les résultats réellement calculés des expériences encore non exécutées. Le projet utilise uniquement ary, conformément à la consigne finale.\n\n"
    for i, (title, explanation, measured, artifact, problem, solution, conclusion, paper) in enumerate(themes, 1):
        source = next(p["source"] for p in PAPERS if p["id"] == paper)
        text += f"## Q{i} provisoire — {title}\n\n**Explication simple :** {explanation}\n\n**Résultats réellement disponibles :** {measured}\n\n**Tableau ou figure :** `{artifact}`.\n\n**Problème :** {problem}\n\n**Solution implémentée ou testée :** {solution}\n\n**Conclusion scientifique :** {conclusion}\n\n**Référence :** {source}\n\n"
    text += "## Questions probables du professeur\n\n- **Pourquoi ne pas mélanger train et test ?** Le test doit représenter des exemples non utilisés pour apprendre ou choisir.\n- **Pourquoi Macro-F1 et Weighted-F1 ?** Le premier traite chaque classe également ; le second pondère par les effectifs réels.\n- **Pourquoi trois graines identiques en TF-IDF ?** Le solveur peut être déterministe ; les réplications identiques ne constituent pas trois jeux de données indépendants.\n- **Pourquoi pas de transfert algérien ?** La consigne finale demande exclusivement ary.\n- **Une attention forte explique-t-elle la décision ?** Elle décrit le mécanisme interne ; une intervention contrôlée serait nécessaire pour une conclusion causale.\n- **Pourquoi un score manque-t-il ?** Le run n'a pas été exécuté ; on ne le remplace jamais par une estimation présentée comme mesurée.\n"
    (ROOT / "research/oral.md").write_text(text, encoding="utf-8")
    return text
