"""Textes pédagogiques de l'interface, indépendants des métriques enregistrées."""
FAMILIES = {
    "A": {
        "name": "ML classique", "title": "Des fragments de texte, un premier repère.",
        "intro": "TF-IDF décrit les fragments présents dans un tweet. Une régression logistique apprend ensuite lesquels sont associés à chaque sentiment.",
        "objective": "Établir une référence simple, rapide et explicable avant les modèles plus coûteux.",
        "principle": "Comparer la majorité, les mots, les caractères et leur combinaison. Le vocabulaire et les poids TF-IDF sont appris uniquement sur train.",
        "example": "« zwin », « zwina » et « zwiin » peuvent partager des fragments de caractères, même lorsque les mots complets diffèrent.",
        "limit": "Le modèle exploite des associations statistiques ; un alphabet ou un sujet peut devenir un raccourci trompeur.",
        "flow": ["Tweet", "TF-IDF mots / caractères", "Régression logistique", "3 sentiments"],
        "question": "Pourquoi commencer par une méthode simple ?", "answer": "Elle fournit une référence mesurable. Un modèle plus complexe doit justifier son coût par ses résultats, pas par son nom.",
        "paper": "review",
    },
    "B": {
        "name": "Zero-shot", "title": "Transférer ce qu'un modèle connaît déjà.",
        "intro": "Le NLI confronte un tweet à des hypothèses de sentiment. CAMeLBERT utilise une tête sentiment déjà entraînée sur d'autres corpus.",
        "objective": "Mesurer ce que des modèles externes savent faire sans entraîner leurs paramètres sur ary.",
        "principle": "Comparer des hypothèses en arabe, français et anglais, puis CAMeLBERT. Les scores bruts et la décision ajustée sur dev restent distincts.",
        "example": "Pour « hadchi zwin », le NLI compare trois hypothèses : sentiment positif, négatif et neutre.",
        "limit": "Zero-shot sur ary ne signifie pas absence de supervision antérieure. La couverture Arabizi et la contamination externe restent des questions ouvertes.",
        "flow": ["Tweet + hypothèses", "NLI multilingue / CAMeLBERT", "Scores réordonnés", "Décision"],
        "question": "CAMeLBERT est-il non supervisé ?", "answer": "Non. Il a appris le sentiment sur des corpus externes. Le transfert est sans apprentissage local de ses poids sur ary.",
        "paper": "ucas",
    },
    "C": {
        "name": "Deep Learning", "title": "Apprendre les régularités du langage.",
        "intro": "Les réseaux transforment les tokens en vecteurs, puis apprennent à combiner leur contexte. Ici, leurs paramètres sont appris sur le train marocain.",
        "objective": "Comparer BiLSTM, attention, Mini-Transformer et Character-CNN à conditions déclarées.",
        "principle": "Initialiser les embeddings au hasard, avec AraVec ou avec fastText appris sur train. Tester séparément positions, pondération de la perte et oversampling.",
        "example": "Un BiLSTM lit les deux directions. L'attention apprend une pondération des états ; les positions de remplissage sont masquées.",
        "limit": "Le vocabulaire est fermé à train avec une fréquence minimale de 2. Les OOV restent UNK, y compris pour l'initialisation fastText.",
        "flow": ["Tokens", "Embeddings", "BiLSTM / Transformer / CNN", "3 sentiments"],
        "question": "Un poids d'attention explique-t-il une décision ?", "answer": "Il décrit une pondération interne. Il ne suffit pas à démontrer une relation causale ; les cinq visualisations restent descriptives.",
        "paper": "fasttext",
    },
    "D": {
        "name": "Encodeur gelé", "title": "Garder les représentations, apprendre la lecture.",
        "intro": "DarijaBERT produit des représentations sans modifier ses poids. Seule une tête de classification apprend le sentiment local.",
        "objective": "Comparer CLS + régression logistique, moyenne + régression et états cachés + BiLSTM-attention.",
        "principle": "Désactiver les gradients et le dropout de l'encodeur. Réutiliser ses extractions entre les graines, en déclarant le coût de préparation.",
        "example": "Mean pooling moyenne les états non-PAD ; CLS prend le premier état. D2 conserve la séquence pour sa tête attentive.",
        "limit": "Peu de paramètres entraînables ne signifie pas une inférence légère : l'encodeur complet reste utilisé.",
        "flow": ["Tweet", "DarijaBERT gelé", "CLS / moyenne / séquence", "Tête sentiment"],
        "question": "Qu'est-ce qui apprend lorsque BERT est gelé ?", "answer": "La régression ou la tête BiLSTM-attention. Les poids de l'encodeur ne reçoivent aucun gradient.",
        "paper": "darijabert",
    },
    "E": {
        "name": "Fine-tuning", "title": "Adapter aussi les représentations.",
        "intro": "Le fine-tuning ajuste l'encodeur et une seule tête sentiment ary. Les variantes DarijaBERT et DarijaBERT-Arabizi sont prévues dans un protocole équivalent.",
        "objective": "Comparer l'adaptation des deux checkpoints au sentiment marocain, sans corpus arq.",
        "principle": "AdamW à 2e-5, batch effectif 32, warmup 10 %, trois à cinq époques, early stopping et graines 42, 43, 44.",
        "example": "Un micro-batch de 4 répété huit fois par accumulation donne un batch effectif de 32.",
        "limit": "La campagne complète n'a pas été exécutée sur cette machine CPU. Les tests techniques sur deux tweets ne constituent pas des résultats de benchmark.",
        "flow": ["Tweet ary", "DarijaBERT / Arabizi ajustable", "Tête sentiment ary", "Validation dev"],
        "question": "Pourquoi les scores E sont-ils absents ?", "answer": "Les architectures et un pas AdamW réel sont vérifiés, mais pas la campagne complète. Aucun score ni checkpoint sentiment n'est inventé.",
        "paper": "darijabert",
    },
}

GLOSSARY = {
    "Weighted-F1": "Moyenne des F1 des classes pondérée par leurs effectifs. C'est le critère principal déclaré du TP.",
    "Macro-F1": "Moyenne simple des F1 des trois classes : chaque sentiment compte autant, même si ses exemples sont moins nombreux.",
    "Précision": "Parmi les tweets prédits dans une classe, proportion dont le label officiel correspond à cette classe.",
    "Rappel": "Parmi les tweets d'une classe, proportion retrouvée correctement par le modèle.",
    "OOV": "Token absent du vocabulaire construit sur train. min_freq=2 exclut aussi certains tokens rares du train.",
    "Embedding": "Vecteur numérique représentant un token. Sa proximité avec d'autres vecteurs peut représenter des régularités apprises.",
    "Attention": "Pondération apprise des états d'une séquence. Visualiser ses poids ne prouve pas une explication causale.",
    "Graine": "Valeur fixant des générateurs aléatoires. Comparer plusieurs graines aide à observer la variabilité de l'apprentissage.",
    "Bootstrap": "Rééchantillonnage des tweets avec remise. Les 1 000 répétitions estiment ici un intervalle conditionnel au modèle déjà entraîné.",
    "McNemar": "Test apparié des désaccords de justesse entre deux modèles. Il ne teste pas directement une différence de F1.",
    "Arabizi": "Écriture de l'arabe avec des lettres latines et parfois des chiffres. Une écriture latine ne prouve pas qu'un texte est en Arabizi.",
    "Dev / test": "Dev sert à choisir. Test sert à mesurer après verrouillage, sans changer les choix selon ses résultats.",
}

VARIANTS = {
    "majority": "Classe majoritaire", "word": "TF-IDF mots", "char": "TF-IDF caractères", "union": "TF-IDF mots + caractères",
    "camelbert": "CAMeLBERT sentiment", "nli_ar": "NLI · hypothèses arabes", "nli_fr": "NLI · hypothèses françaises", "nli_en": "NLI · hypothèses anglaises",
    "cls": "CLS + régression", "mean": "Moyenne + régression", "states": "États + BiLSTM-attention",
    "darijabert": "DarijaBERT ajusté", "arabizi": "DarijaBERT-Arabizi ajusté",
}


def variant_name(variant):
    if variant in VARIANTS:
        return VARIANTS[variant]
    text = str(variant)
    replacements = [("union_train_arabic", "TF-IDF · train arabe"), ("union_train_latin", "TF-IDF · train latin"),
                    ("charcnn", "Character-CNN"), ("bilstm", "BiLSTM"), ("transformer", "Transformer"), ("attention", "Attention"),
                    ("fasttext", "fastText"), ("aravec", "AraVec"), ("random", "aléatoire"), ("nopos", "sans positions"),
                    ("weighted", "perte pondérée"), ("oversampling", "suréchantillonnage")]
    for old, new in replacements:
        text = text.replace(old, new)
    return text.replace("_", " · ")
