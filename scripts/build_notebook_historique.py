"""Construit le support pédagogique (40 cellules de code principales)."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def lesson(title, objective, why, principle, example, reference, code):
    md(f"## {title}\n\n**Objectif :** {objective}\n\n**Pourquoi ?** {why}\n\n**Principe :** {principle}\n\n**Exemple pédagogique :** {example}\n\n**Recherche scientifique :** {reference}")
    cells.append(nbf.v4.new_code_cell(code.strip()))


def recap(family, done, learned, difficulties, solutions, conclusion, question):
    md(f"### Mini-résumé {family}\n\n**Ce que j'ai fait :** {done} L'état réel des exécutions est indiqué dans les tableaux ci-dessus.\n\n**Ce que j'ai appris :** {learned}\n\n**Les difficultés rencontrées :** {difficulties}\n\n**Les solutions testées :** {solutions} Une ligne « non exécuté » signifie que la solution est seulement implémentée.\n\n**Ce que les résultats permettent de conclure :** {conclusion}\n\n**Question probable du professeur et réponse :** {question}")


md("""# TP2 — Groupe 12 : Analyse de sentiment en Darija marocaine

### Du tweet au raisonnement scientifique · Master Big Data

**Périmètre final : exclusivement le corpus marocain `ary`.** Aucun corpus algérien ni apprentissage multi-tâche entre pays. Nous conservons exactement cinq familles : **A** ML classique, **B** transfert sans entraînement local, **C** réseaux entraînés sur train, **D** DarijaBERT gelé, **E** fine-tuning sur ary.

Ce notebook est le document principal. Lire les explications, exécuter les cellules dans l'ordre puis interpréter les **sorties réelles**. Les modules `src/` regroupent les boucles répétitives ; leur rôle est expliqué au fil du texte. Les exemples inventés pour expliquer une notion sont explicitement pédagogiques et ne sont jamais ajoutés au benchmark.

**Parcours :** 1. environnement/recherche → 2. données et audit → A–E. modèles → F. robustesse → G. comparaison et oral.

**Deux modes :** lecture des résultats sauvegardés (par défaut, sans réentraînement) ou nouvelle étude avant verrouillage. Les tableaux vides sont remplacés par une explication « non exécuté ». La lecture de ce notebook n'exige aucun téléchargement de modèle lourd.

**Protocole :** train apprend les paramètres, dev choisit architecture/époque/décalages, test évalue une fois les choix verrouillés. La disponibilité publique du test ne donne pas le droit de l'utiliser pour choisir. Les labels test et les votes associés ne sont analysés qu'en F/G. Les collisions textuelles peuvent être auditées avant, sans leurs labels.

**Questions officielles :** les énoncés Q1–Q12 n'ont pas été fournis. La fin contient douze thèmes d'oral provisoires ; ils ne sont pas présentés comme les réponses certifiées au sujet absent.
""")

lesson("1.1 — Ouvrir le projet et fixer les choix", "Préparer un chemin stable et des commandes visibles.", "Un notebook ouvert depuis notebooks/ doit retrouver ses modules et ses données.", "Utiliser pathlib et les fonctions courtes des cinq modules.", "Une phrase telle que « had lfilm زوين » traverse le même nettoyage dans le notebook et l'application.", "[AfriSenti](https://arxiv.org/abs/2302.08956) motive un protocole reproductible.", r'''
from pathlib import Path
import sys, json, html
PROJECT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src/preprocess.py").exists())
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML, Image
from src.preprocess import ROOT, LABELS, download_data, load_split, audit_data, clean_text, tokens, build_vocab, encode_texts, oov_table, save_json
from src.models import HYPOTHESES, TextNetwork, MODEL_REVISIONS
from src.train import runtime_info, plan_table, run_campaign, pretrained_run, frozen_run, finetune_run, load_predictor, seed_everything
from src.evaluate import results_table, interpret, by_script, decide, metric_scores, select_on_dev, lock_selection, final_evaluation, attention_examples, plot_histories, posthoc_diagnostics
from src.research import papers_table, solutions_table, oral_report
plt.rcParams.update({"figure.figsize": (8, 4), "axes.spines.top": False, "axes.spines.right": False})
COLORS = ["#c95b5b", "#6b7b90", "#168a82"]
TRAIN_A = TRAIN_B = TRAIN_C = TRAIN_D = TRAIN_E = False
LOCK_AND_TEST = False  # True uniquement après toutes les comparaisons dev souhaitées.
SEEDS = [42, 43, 44]
DEVICE = "cpu"  # Le runtime livré est CPU ; modifier explicitement si CUDA est disponible.
ARAVEC_PATH = ROOT / "models/resources/aravec/full_uni_cbow_100_twitter.mdl"
def show_csv(name, limit=30):
    path = ROOT / "results" / name
    if not path.exists():
        print(f"Non exécuté : {name}")
        return pd.DataFrame()
    frame = pd.read_csv(path)
    display(frame.head(limit))
    return frame
def family_rows(family, split="dev"):
    frame = results_table()
    return frame.loc[(frame.family == family) & (frame.split == split) & (frame.decision == "dev_offsets")] if len(frame) else frame
def show_family(family):
    frame = family_rows(family)
    display(frame)
    print(interpret(frame))
    return frame
def show_figure(name):
    path = ROOT / "results/figures" / name
    if path.exists():
        display(Image(filename=str(path)))
    else:
        print(f"Figure non disponible : {name} (aucune donnée simulée).")
def show_attention_tokens(table):
    for example, group in table.groupby("example", sort=False):
        maximum = max(float(group.attention.max()), 1e-12)
        colored = " ".join(f'<span title="poids={row.attention:.4f}" style="background:rgba(22,138,130,{0.08+0.65*row.attention/maximum:.3f});padding:3px;border-radius:3px">{html.escape(row.token)}</span>' for row in group.itertuples())
        display(HTML(f'<p><b>{html.escape(example)}</b></p><p dir="auto" style="font:18px Arial;line-height:2.3">{colored}</p>'))
print("Projet :", ROOT)
display(pd.DataFrame({"action": ["A", "B", "C", "D", "E", "verrouillage/test"], "exécuter": [TRAIN_A, TRAIN_B, TRAIN_C, TRAIN_D, TRAIN_E, LOCK_AND_TEST]}))
''')

lesson("1.2 — Ressources et reproductibilité", "Mesurer les ressources actuelles.", "Un GPU absent change le temps disponible, pas le nombre de tweets officiel.", "Afficher les versions, CPU, RAM et CUDA ; garder trois graines.", "Un micro-batch de 4 accumulé huit fois donne un batch effectif de 32.", "[DarijaBERT](https://link.springer.com/article/10.1007/s41060-023-00498-2) fournit des encodeurs coûteux à ajuster.", '''
runtime = runtime_info()
display(pd.DataFrame([runtime]).drop(columns="packages"))
display(pd.Series(runtime["packages"], name="version").to_frame())
print("Graines prévues :", SEEDS, "— aucun changement silencieux du protocole.")
''')

lesson("1.3 — Ce que dit la recherche", "Relier chaque méthode à une source vérifiée.", "Un gain publié sur un autre corpus n'est pas notre résultat.", "Lire les fiches : problème, méthode, résultat des auteurs et expérience locale proposée.", "BAERT dépasse mBERT sur SALAD : cela motive une comparaison Arabizi, sans prédire notre F1.", "[Baert et al.](https://aclanthology.org/2020.coling-main.51/) ; bibliographie complète dans research/papers.md.", '''
display(papers_table()[["id", "title", "authors", "year", "source"]])
display(Markdown((ROOT / "research/papers.md").read_text(encoding="utf-8")))
print("Recherche ciblée vérifiée au 29/10/2026. Accès Google Scholar tenté mais non disponible ; sources primaires consultées.")
''')

lesson("2.1 — Télécharger les seules données ary", "Conserver les trois fichiers officiels et leur provenance.", "Changer les partitions rendrait le benchmark difficile à comparer.", "Téléchargement à une révision Git immuable, empreintes SHA-256, aucune génération de tweets.", "train.tsv contient un texte et son label officiel.", "[Dépôt officiel AfriSenti](https://github.com/afrisenti-semeval/afrisent-semeval-2023).", '''
manifest = download_data()
train, dev = load_split("train"), load_split("dev")
test_texts = load_split("test")  # Sans colonne label.
display(manifest[["path", "sha256"]])
display(pd.DataFrame({"split": ["train", "dev", "test"], "n": [len(train), len(dev), len(test_texts)], "labels_utilisés": [True, True, False]}))
assert "label" not in test_texts
print("Effectifs attendus : 5583 / 494 / 2961 ; vérifier le tableau avant d'entraîner.")
''')

lesson("2.2 — Doublons, contradictions et collisions", "Auditer les textes avant de comparer les modèles.", "Un tweet déjà vu peut gonfler l'évaluation.", "Détecter doublons exacts/après nettoyage et labels contradictoires train/dev ; collisions test sans labels.", "Deux textes identiques avec des labels train différents seraient une contradiction.", "[AfriSenti](https://arxiv.org/abs/2302.08956), annotation et limites des ressources.", '''
counts, collisions, conflicts = audit_data()
display(counts)
display(collisions)
display(conflicts)
print(f"Audit : {int(counts.raw_duplicates.sum())} doublons internes, {len(collisions)} collisions entre partitions, {len(conflicts)} contradictions train/dev détectées.")
print("Les fichiers officiels restent intacts. Ces observations sont des limites, pas une permission de déplacer des exemples.")
''')

lesson("2.3 — Lire cinq tweets et leurs écritures", "Comprendre ce que contient le corpus.", "Une statistique ne remplace pas la lecture des exemples.", "Afficher cinq textes train échappés en HTML, direction automatique pour l'arabe.", "« زوين بزاف » est un exemple pédagogique de sentiment positif selon son contexte.", "[Revue des dialectes arabes](https://doi.org/10.1016/j.jksuci.2023.101570).", '''
display(train[["row_id", "tweet", "label", "script"]].head())
for text in train.tweet.head():
    display(HTML(f'<p dir="auto" style="font-family:Arial;font-size:17px">{html.escape(text)}</p>'))
print("Les cinq lignes sont de vrais exemples train ; script désigne l'alphabet détecté.")
''')

lesson("2.4 — Distribution des sentiments train/dev", "Mesurer le déséquilibre sans ouvrir les labels test.", "Un score peut sembler bon en privilégiant la classe la plus fréquente.", "Comparer effectifs train et proportions train/dev ; ne pas corriger artificiellement dev.", "Toujours prédire neutre peut avoir un rappel neutre élevé mais échouer sur les autres classes.", "[SemEval-2023 tâche 12](https://aclanthology.org/2023.semeval-1.315/) utilise Weighted-F1.", '''
distribution = pd.DataFrame({"train": train.label.value_counts().reindex(LABELS, fill_value=0), "dev": dev.label.value_counts().reindex(LABELS, fill_value=0)})
display(distribution)
ax = distribution.div(distribution.sum()).plot.bar(color=[COLORS[2], COLORS[1]])
ax.set(ylabel="Proportion", title="Sentiments — train et dev uniquement")
plt.tight_layout(); plt.savefig(ROOT / "results/figures/labels_train_dev.png", dpi=140); plt.show()
print("Weighted-F1 pondère chaque F1 par l'effectif de sa classe ; Macro-F1 est leur moyenne simple.")
''')

lesson("2.5 — Écritures et raccourcis possibles", "Voir si les sentiments dépendent de l'écriture.", "Un modèle peut exploiter un indice d'alphabet plutôt que comprendre le sentiment.", "Croiser script et label sur train, puis mesurer une baseline script-only sur dev en A.", "Une majorité neutre parmi les textes latins peut créer un raccourci.", "[Baert et al.](https://aclanthology.org/2020.coling-main.51/) souligne la spécificité de l'Arabizi.", '''
scripts = pd.DataFrame({"train": train.script.value_counts(), "dev": dev.script.value_counts(), "test_textes": test_texts.script.value_counts()}).fillna(0).astype(int)
display(scripts)
display(pd.crosstab(train.script, train.label, normalize="index").reindex(columns=LABELS))
french_pattern = r"\b(?:mais|avec|pour|merci|vraiment|bonjour|parce)\b"
french_hints = pd.DataFrame([{"split": name, "indices_français_possibles": int(frame.text.str.contains(french_pattern, case=False, regex=True).sum()), "tweets": len(frame)} for name, frame in [("train", train), ("dev", dev)]])
display(french_hints)
display(train.loc[train.text.str.contains(french_pattern, case=False, regex=True), ["text", "script"]].head(5))
scripts.plot.bar(color=[COLORS[2], COLORS[1], COLORS[0]])
plt.ylabel("Tweets"); plt.title("Écritures détectées — aucun label test")
plt.tight_layout(); plt.savefig(ROOT / "results/figures/scripts.png", dpi=140); plt.show()
print("L'écriture latine peut contenir Arabizi, français ou les deux. Ce détecteur ne classe pas les langues.")
print("Les indices français sont une heuristique lexicale provisoire, pas une annotation de langue ni une preuve de code-switching.")
''')

lesson("2.6 — Nettoyage conservateur", "Uniformiser URLs, mentions et espaces sans supprimer le sentiment.", "Les chiffres 3/7/9 et les répétitions sont informatifs en Arabizi.", "Nettoyage déterministe sans dictionnaire appris, sans suppression des stopwords ni des négations.", "« ma3jbnich 😡!!! » garde sa négation, son emoji et son intensité.", "[Matrane et al.](https://doi.org/10.1016/j.jksuci.2023.101570) synthétise les choix de prétraitement.", '''
examples = train[["tweet", "text"]].head(5).rename(columns={"tweet": "avant", "text": "après"})
display(examples)
display(pd.DataFrame({"pédagogique_avant": ["ma3jbnich 😡!!! https://exemple.ma @ami"], "après": [clean_text("ma3jbnich 😡!!! https://exemple.ma @ami")]}))
print("Ces transformations sont identiques à l'entraînement et à l'inférence de l'application.")
''')

lesson("2.7 — Tokens, vocabulaire, OOV et tenseurs", "Passer des textes à des matrices d'entiers.", "Le réseau n'accepte pas directement une chaîne de caractères.", "Vocabulaire appris sur train uniquement, fréquence minimale 2 ; PAD=0, UNK=1 ; séquence tronquée à 64 tokens.", "Un token absent du vocabulaire devient UNK ; PAD complète une phrase courte.", "[fastText](https://aclanthology.org/Q17-1010/) motive les sous-mots ; notre vocabulaire reste limité à train.", '''
vocab = build_vocab(train.text, min_freq=2)
encoded = encode_texts(train.text.head(5).tolist(), vocab, max_length=64)
display(pd.DataFrame({"texte": train.text.head(), "tokens": train.text.head().map(tokens)}))
display(pd.DataFrame(encoded[:, :12]))
oov = oov_table({"train": train, "dev": dev}, vocab)
display(oov)
print("Vocabulaire :", len(vocab), "— tenseur de cinq exemples :", encoded.shape)
print("OOV test sera affiché en F3. FastText initialise les mots du vocabulaire ; les tokens hors vocabulaire restent UNK dans cette implémentation.")
print("OOV est calculé sur les tokens regex des textes complets, ponctuation/emojis inclus. Le train peut avoir des OOV : les tokens de fréquence 1 sont écartés par min_freq=2.")
''')

md("""# Partie A — Machine Learning classique

Le **TF-IDF** transforme chaque tweet en un vecteur creux : une composante par mot, paire de mots ou fragment de caractères. La fréquence inverse de document réduit le poids des fragments présents presque partout. La **régression logistique multiclasse** apprend trois scores. `class_weight='balanced'` augmente la contribution des classes moins fréquentes pendant l'apprentissage.

Pour la classe $c$, $F1_c=2P_cR_c/(P_c+R_c)$. Le Macro-F1 moyenne les trois valeurs à poids égal ; le Weighted-F1 les pondère par les effectifs. Leur désaccord est instructif.
""")
lesson("A.1 — Entraîner les baselines", "Comparer majorité, mots 1–2, caractères 2–5 et combinaison.", "Les méthodes simples sont des références nécessaires avant les Transformers.", "Chaque vectoriseur est ajusté sur train ; C=1 est fixé ; trois graines sont enregistrées.", "Les variantes « zwina » et « zwin » partagent des fragments de caractères.", "[Revue de Matrane et al.](https://doi.org/10.1016/j.jksuci.2023.101570).", '''
if TRAIN_A:
    run_campaign("A")
display(plan_table().query("family == 'A'"))
print("TRAIN_A=False : lecture des résultats sauvegardés, sans nouvel entraînement.")
''')

lesson("A.2 — Comparer les mesures dev", "Identifier la meilleure représentation parmi les runs disponibles.", "Le nombre de paramètres ou le nom du modèle ne prédit pas son F1.", "Moyennes par variante et variabilité entre graines ; aucune lecture test.", "Une différence mots/caractères sur dev reste à vérifier sur test après verrouillage.", "[AfriSenti](https://arxiv.org/abs/2302.08956).", '''
a_results = show_family("A")
benchmark_a = a_results.loc[a_results.purpose == "benchmark"] if len(a_results) else a_results
if len(benchmark_a):
    a_summary = benchmark_a.groupby("variant")[["weighted_f1", "macro_f1"]].mean()
    display(a_summary)
    a_summary.plot.bar(color=[COLORS[2], COLORS[1]], ylim=(0, 1))
    plt.title("A — validation officielle dev"); plt.ylabel("F1")
    plt.tight_layout(); plt.savefig(ROOT / "results/figures/A_dev.png", dpi=140); plt.show()
print("Des graines identiques en TF-IDF sont possibles avec un solveur déterministe : ce ne sont pas trois corpus indépendants.")
''')

lesson("A.3 — Règle de décision optimisée sur dev", "Comparer argmax simple et décalages multiclasse.", "Le F1 n'est pas directement la fonction de perte logistique.", "Prédire argmax(log(p)+b), avec b_negative=0 et une grille fixée de 25 couples neutral/positive sur dev.", "Un décalage neutre positif augmente la préférence pour neutre sans réentraîner le modèle.", "[SemEval](https://aclanthology.org/2023.semeval-1.315/) fournit le critère ; la grille est notre choix expérimental.", '''
all_results = results_table()
if len(all_results):
    display(all_results.query("family == 'A' and split == 'dev'")[["variant", "seed", "decision", "weighted_f1", "macro_f1"]])
    offset_rows = []
    for file in sorted((ROOT / "models").glob("A_*/meta.json")):
        meta = json.loads(file.read_text(encoding="utf-8"))
        offset_rows.append({"run": meta["run_id"], "décalages_dev": meta["offsets"]})
    display(pd.DataFrame(offset_rows))
else:
    print("Non exécuté : aucun score A.")
print("Limite : cette sélection utilise une petite validation de 494 tweets ; il faut examiner sa stabilité sur test sans réoptimiser.")
''')

lesson("A.4 — Raccourci d'écriture et matrice dev", "Quantifier ce que l'alphabet seul permet de prédire.", "Une corrélation d'écriture peut masquer une faiblesse linguistique.", "Choisir la classe majoritaire de chaque script sur train, évaluer sur dev ; inspecter la meilleure baseline A sur dev.", "Le script latin n'est pas un sentiment : cette baseline mesure un raccourci possible.", "[Baert et al.](https://aclanthology.org/2020.coling-main.51/).", '''
from src.evaluate import save_confusion
script_majority = train.groupby("script").label.agg(lambda values: values.value_counts().idxmax()).to_dict()
script_prediction = dev.script.map(script_majority).fillna(train.label.mode().iloc[0])
display(pd.DataFrame([metric_scores(dev.label, script_prediction)]).assign(modèle="écriture seulement, appris sur train"))
if len(benchmark_a):
    selected_a = benchmark_a.groupby("variant").weighted_f1.mean().idxmax()
    row = benchmark_a.query("variant == @selected_a and seed == 42").iloc[0]
    folder = ROOT / row.checkpoint
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    prediction_a = load_predictor(folder)(dev.text.tolist())
    display(by_script(dev, prediction_a, meta["offsets"]))
    path = save_confusion(dev.label, decide(prediction_a, meta["offsets"]), "A_dev_confusion")
    display(Image(filename=str(path)))
print("Une bonne baseline script-only indiquerait un biais exploitable ; elle ne démontre pas que les autres modèles l'utilisent exclusivement.")
''')
recap("A", "Implémenté quatre baselines et une règle de décision dev.", "Les caractères partagent des fragments entre graphies ; le F1 se lit par classe et par écriture.", "Faible validation, classes inégales et corrélations de script.", "TF-IDF pondéré, décalages dev et diagnostic script-only.", "Les tableaux donnent un classement dev des seules expériences réalisées, pas encore un choix fondé sur test.", "Pourquoi TF-IDF peut-il battre un réseau ? Sur un petit corpus, ses fragments discriminants peuvent être appris plus efficacement.")

md("# Partie B — Zero-shot sur ary\n\nB1 est un modèle NLI multilingue : il estime la compatibilité entre le tweet et une hypothèse de sentiment. B2 a déjà été supervisé en sentiment sur des corpus externes. Ici, « zero-shot » signifie **sans apprentissage des paramètres sur ary**. La langue d'hypothèse et les éventuels décalages sont choisis sur dev ; la variante optimisée est donc une adaptation de décision sur dev. Les scores argmax bruts restent enregistrés séparément.")
lesson("B.1 — NLI et trois formulations", "Comparer les hypothèses arabe, française et anglaise.", "Une traduction d'hypothèse peut modifier les scores NLI.", "Trois paires tweet/hypothèse ; softmax des logits d'entailment entre les trois sentiments.", "Le tweet « hadchi zwin » est comparé à positif, négatif et neutre dans chaque langue.", "[Carte mDeBERTa NLI](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7).", '''
display(pd.DataFrame(HYPOTHESES, index=LABELS))
if TRAIN_B:
    for language in ("ar", "fr", "en"):
        pretrained_run("nli", language)
display(plan_table().query("family == 'B'"))
print("Le modèle est fixé à une révision. Aucun label test ne choisit la langue d'hypothèse.")
''')
lesson("B.2 — CAMeLBERT et contamination externe", "Évaluer un classifieur sentiment arabe externe.", "Sa couverture de l'Arabizi et son historique de corpus diffèrent de ceux du NLI.", "Lire id2label du checkpoint plutôt que supposer l'ordre des classes.", "Un texte latin peut être fragmenté par le tokenizer arabe et perdre des indices.", "[Carte CAMeLBERT](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment) : ASTD, ArSAS et SemEval.", '''
if TRAIN_B:
    pretrained_run("camel")
display(pd.DataFrame({"modèle": list(MODEL_REVISIONS), "révision": list(MODEL_REVISIONS.values())}))
print("Corpus externes CAMeLBERT : ASTD, ArSAS, SemEval. L'absence de contamination externe n'est pas certifiée.")
''')
lesson("B.3 — Scores réels et décomposition par écriture", "Lire les résultats B sans attribuer des performances aux runs absents.", "Une moyenne peut cacher l'échec sur les textes latins.", "Utiliser les prédictions dev sauvegardées et regrouper par script.", "Un meilleur score arabe que latin serait une observation, pas une propriété garantie.", "[UCAS-IIE-NLP](https://aclanthology.org/2023.semeval-1.255/) traite la généralisation low-resource avec une autre méthode.", '''
b_results = show_family("B")
for row in b_results.itertuples():
    file = ROOT / "results" / f"{row.run_id}_dev_tuned.csv"
    predictions = pd.read_csv(file)
    script_metrics = []
    for script, group in predictions.groupby("script"):
        script_metrics.append({"modèle": row.variant, "script": script, "n": len(group), **metric_scores(group.label, group.prediction)})
    display(pd.DataFrame(script_metrics))
print("Les effectifs mixtes sont faibles : un F1 mixte peut être très instable.")
''')
recap("B", "Implémenté NLI dans trois langues et CAMeLBERT.", "Zero-shot local n'efface pas la supervision externe.", "Langue d'hypothèse, segmentation Arabizi et corpus antérieurs incomplets.", "Comparaison de formulations, mapping explicite et scores par écriture.", "Seules les lignes calculées autorisent une comparaison ; aucune absence de contamination n'est affirmée.", "CAMeLBERT est-il non supervisé ? Non, il a appris le sentiment ailleurs ; il ne reçoit ici aucun entraînement de paramètres sur ary.")

md("# Partie C — Réseaux entraînés sur train ary\n\nUn **embedding** est un vecteur appris pour un token. Un **BiLSTM** agrège le contexte dans les deux sens. L'**attention personnalisée** calcule un poids par état, après masque des PAD. Le **Mini-Transformer** comporte exactement deux couches, dimension 128 et quatre têtes ; on teste l'effet des positions. Le **Character-CNN** reçoit des caractères et apprend des motifs locaux, sans vocabulaire de mots. Les embeddings de mots AraVec/fastText ne s'appliquent donc pas directement au CNN caractères.")
lesson("C.1 — Tester les dimensions avant d'entraîner", "Vérifier les sorties des quatre architectures.", "Une erreur de forme ou de masque peut invalider une expérience longue.", "Cinq vrais tweets train, tenseurs entiers, trois logits par exemple.", "Une entrée (5,64) devient une sortie (5,3).", "[Zhang et al.](https://arxiv.org/abs/1509.01626) ; [Vaswani et al.](https://arxiv.org/abs/1706.03762).", '''
import torch
seed_everything(42)
shape_rows = []
for kind in ("bilstm", "attention", "transformer", "charcnn"):
    character = kind == "charcnn"
    current_vocab = build_vocab(train.text, character=character)
    inputs = torch.tensor(encode_texts(train.text.head(5).tolist(), current_vocab, 256 if character else 64, character))
    network = TextNetwork(len(current_vocab), kind)
    outputs = network(inputs)
    assert outputs.shape == (5, 3) and torch.isfinite(outputs).all()
    shape_rows.append({"architecture": kind, "entrée": tuple(inputs.shape), "sortie": tuple(outputs.shape), "paramètres": sum(p.numel() for p in network.parameters())})
display(pd.DataFrame(shape_rows))
print("Poids initiaux : vérification de formes seulement, aucun score de benchmark calculé ici.")
''')
lesson("C.2 — Embeddings aléatoires, AraVec et fastText", "Comparer trois initialisations de mots de dimension commune 128.", "AraVec connaît surtout l'arabe ; fastText train peut apprendre des fragments Arabizi.", "FastText Gensim skip-gram est appris sur train uniquement ; AraVec 100D est projeté par une matrice aléatoire fixée vers 128D. Les embeddings restent ajustables pendant C.", "Les formes proches peuvent partager des sous-mots, mais les mots hors vocabulaire du classifieur restent UNK.", "[Bojanowski et al.](https://aclanthology.org/Q17-1010/) ; [AraVec officiel](https://github.com/bakrianoo/aravec).", '''
coverage_rows = []
for file in sorted((ROOT / "models").glob("C_*/embedding_coverage.json")):
    coverage_rows.append({"run": file.parent.name, **json.loads(file.read_text(encoding="utf-8"))})
display(pd.DataFrame(coverage_rows))
print("AraVec local disponible :", ARAVEC_PATH.exists())
print("Une couverture du vocabulaire n'est pas une mesure de F1. Le coût d'apprentissage des embeddings est inclus dans C.")
''')
lesson("C.3 — Entraînements et configurations", "Exécuter les architectures et leurs variantes déclarées.", "Trois graines aident à séparer un gain régulier d'un hasard d'initialisation.", "AdamW, 3 époques max, LR=1e-3, batch 32 ; meilleure époque sur Weighted-F1 dev avant réglage de décision.", "Une perte dev qui stagne peut déclencher l'early stopping ; les époques effectuées restent dans history.csv.", "[DarijaBERT](https://link.springer.com/article/10.1007/s41060-023-00498-2) motive une comparaison avec des modèles de capacité moindre.", '''
if TRAIN_C:
    run_campaign("C", aravec_path=ARAVEC_PATH if ARAVEC_PATH.exists() else None, device=DEVICE)
display(plan_table().query("family == 'C'"))
print("Si AraVec manque, ses lignes restent non exécutées. Le nombre de tweets n'est jamais réduit pour le benchmark.")
''')
lesson("C.4 — Courbes et comparaison", "Observer loss train et Weighted-F1 dev.", "Une baisse de loss ne garantit pas une amélioration de généralisation.", "Courbes des runs graine 42 et tableaux de toutes les graines ; figures enregistrées seulement si les données existent.", "Une loss qui baisse avec un F1 dev stable suggère un risque de surapprentissage.", "[Revue SA dialectale](https://doi.org/10.1016/j.jksuci.2023.101570).", '''
c_results = show_family("C")
if len(c_results):
    display(c_results.groupby("variant").agg(dev_f1=("weighted_f1", "mean"), dev_macro=("macro_f1", "mean"), std_graines=("weighted_f1", "std"), graines=("seed", "nunique")))
curve = plot_histories("C")
if curve:
    display(Image(filename=str(curve)))
else:
    print("Courbes non exécutées.")
''')
lesson("C.5 — Ablations : position, pondération et oversampling", "Mesurer l'effet des solutions envisagées.", "Une idée issue de la recherche peut dégrader ce corpus.", "Changer un facteur à la fois avec mêmes graines ; oversampling strictement train ; comparer les moyennes dev.", "La perte pondérée augmente la contribution des classes rares, sans augmenter leurs effectifs dev.", "[Vaswani et al.](https://arxiv.org/abs/1706.03762) pour les positions ; [revue](https://doi.org/10.1016/j.jksuci.2023.101570) pour les défis.", '''
solutions = solutions_table()
display(solutions.loc[solutions["Problème"].isin(["Mots rares / OOV", "Déséquilibre", "Rééchantillonnage", "Ordre des mots"])])
print("Chaque statut vient des mesures disponibles. Une différence positive sur dev n'est pas une preuve de gain sur test.")
''')
lesson("C.6 — Attention sur cinq exemples dev", "Visualiser les poids réellement appris.", "Voir les mots pondérés aide à discuter les limites, sans transformer l'attention en justification certaine.", "Restaurer C2 graine 42, calculer les poids après masque PAD, afficher les cinq séquences et une figure.", "Un poids élevé sur une négation est descriptif ; son effet causal demanderait une intervention.", "[Revue des méthodes](https://doi.org/10.1016/j.jksuci.2023.101570).", '''
attention_file_c = ROOT / "results/attention_C_5_examples.csv"
attention_c = pd.read_csv(attention_file_c) if attention_file_c.exists() and not TRAIN_C else attention_examples("C", n=5)
display(attention_c)
show_attention_tokens(attention_c)
show_figure("attention_C.png")
print("Poids descriptifs uniquement ; cinq exemples choisis dans dev, pas dans test pour choisir le modèle.")
''')
recap("C", "Implémenté BiLSTM, BiLSTM-attention, Mini-Transformer et Character-CNN.", "Les représentations, l'ordre et l'équilibre des classes sont des facteurs distincts.", "OOV, masques PAD, variance des graines et coût CPU.", "Trois initialisations de mots, ablation positionnelle, perte pondérée, oversampling et décalages dev.", "La couverture des variantes figure dans le tableau ; le statut ne se déduit pas de la présence du code.", "Pourquoi masquer PAD ? Les positions de remplissage ne doivent ni influencer l'attention ni la moyenne.")

md("# Partie D — DarijaBERT gelé\n\n**Gelé** signifie `requires_grad=False` et encodeur en mode évaluation : ses poids et son dropout ne changent pas pendant l'extraction. **CLS** utilise l'état du premier token ; **mean pooling** moyenne les états non-PAD (tokens spéciaux inclus, convention fixée). D2 entraîne une tête BiLSTM-attention sur les états cachés gelés. Le calcul de ces états peut rester coûteux et leur cache prend de la place.")
lesson("D.1 — CLS et mean pooling", "Entraîner seulement une régression logistique sur les représentations BERT.", "Séparer la qualité des représentations du coût du fine-tuning.", "Extraction figée sur train/dev puis classifieur pondéré ; cache associé à la révision et au hash du fichier.", "Chaque tweet devient un vecteur de la dimension cachée du checkpoint.", "[Gaanoun et al.](https://link.springer.com/article/10.1007/s41060-023-00498-2).", '''
if TRAIN_D:
    for pooling in ("cls", "mean"):
        for seed in SEEDS:
            frozen_run(pooling, seed, DEVICE)
display(plan_table().query("family == 'D'"))
extraction_file = ROOT / "results/frozen_extraction_time.json"
if extraction_file.exists():
    display(pd.DataFrame(json.loads(extraction_file.read_text(encoding="utf-8"))))
print("L'encodeur est gelé ; les paramètres de la régression sont entraînables. Le premier run inclut le coût d'extraction ; les suivants peuvent réutiliser le cache.")
''')
lesson("D.2 — États cachés, BiLSTM et attention", "Entraîner une tête séquentielle sans modifier BERT.", "La moyenne peut perdre des relations utiles entre tokens.", "États gelés → BiLSTM → attention masquée → trois scores.", "Une séquence de 64 vecteurs BERT devient trois logits.", "[DarijaBERT](https://link.springer.com/article/10.1007/s41060-023-00498-2).", '''
if TRAIN_D:
    for seed in SEEDS:
        frozen_run("states", seed, DEVICE)
display(plan_table().query("family == 'D' and variant == 'states'"))
print("Cache états float16 sur disque ; conversion float32 pour l'apprentissage. Cette variante peut manquer de mémoire sur la machine locale.")
''')
lesson("D.3 — Attention D2 sur cinq exemples", "Vérifier une visualisation issue d'un vrai checkpoint D2.", "Un encodeur gelé ne rend pas automatiquement la tête interprétable.", "Restaurer les paramètres appris de la tête et calculer les poids sur dev.", "Les sous-tokens BERT ne correspondent pas toujours à des mots entiers.", "[DarijaBERT](https://huggingface.co/SI2M-Lab/DarijaBERT).", '''
attention_file_d = ROOT / "results/attention_D_5_examples.csv"
attention_d = pd.read_csv(attention_file_d) if attention_file_d.exists() and not TRAIN_D else attention_examples("D", n=5)
display(attention_d)
show_attention_tokens(attention_d)
show_figure("attention_D.png")
print("Si D2 n'a pas été entraîné, aucune attention factice n'est produite.")
''')
lesson("D.4 — Temps et paramètres entraînables", "Comparer la précision et le coût des trois têtes.", "Un faible nombre de paramètres entraînables n'élimine pas le coût de BERT à l'inférence.", "Reporter temps réel, F1 et taille des têtes ; G mesurera la latence CPU.", "CLS+LR ajuste peu de poids mais utilise quand même tout l'encodeur.", "[Gaanoun et al.](https://link.springer.com/article/10.1007/s41060-023-00498-2).", '''
d_results = show_family("D")
if len(d_results):
    display(d_results[["variant", "seed", "weighted_f1", "macro_f1", "train_seconds", "trainable_parameters"]])
curve_d = plot_histories("D")
if curve_d:
    display(Image(filename=str(curve_d)))
''')
recap("D", "Implémenté CLS+LR, mean+LR et états+BiLSTM-attention.", "Geler l'encodeur réduit l'apprentissage, pas nécessairement l'inférence.", "Mémoire des états et temps d'extraction, convention de pooling.", "Caches identifiés, moyenne masquée et tête compacte.", "Les temps dépendent du cache ; ils ne doivent pas être comparés sans ce contexte.", "Qu'est-ce qui apprend ? La LR ou la tête séquentielle ; les paramètres de DarijaBERT ne reçoivent aucun gradient.")

md("# Partie E — Fine-tuning exclusivement marocain\n\nOn ajuste les poids de l'encodeur et d'une seule tête sentiment **ary**. Le modèle standard et la variante Arabizi utilisent les mêmes données, graines et budget. La demande finale exclut la tête arq, le corpus algérien et les coefficients multi-tâches. Ils ne font pas partie de ce projet.")
lesson("E.1 — Verrouiller un protocole équitable", "Rendre explicites les hyperparamètres des deux checkpoints.", "Réduire discrètement les tweets ou les époques empêcherait une comparaison scientifique.", "AdamW 2e-5, batch effectif 32, trois époques, warmup 10 %, arrêt anticipé après au moins trois époques ; trois graines.", "4 tweets par micro-batch × 8 accumulations = 32, dernier batch traité avec son effectif réel.", "[DarijaBERT et Arabizi](https://link.springer.com/article/10.1007/s41060-023-00498-2).", '''
e_protocol = {"optimiseur": "AdamW", "lr": 2e-5, "batch_effectif": 32, "micro_batch": 4, "époques_max": 3, "warmup": "10 %", "early_stopping": "patience 2, minimum 3 époques", "graines": SEEDS, "sélection": "Weighted-F1 dev ary", "corpus": "ary uniquement"}
display(pd.Series(e_protocol, name="valeur").to_frame())
resource_file = ROOT / "results/resource_estimate_E.json"
if resource_file.exists():
    display(pd.Series(json.loads(resource_file.read_text(encoding="utf-8")), name="Ressources E").to_frame())
from transformers import AutoTokenizer
tokenizer_examples = []
for model_id in ["SI2M-Lab/DarijaBERT", "SI2M-Lab/DarijaBERT-arabizi"]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=MODEL_REVISIONS[model_id], local_files_only=True)
        tokenizer_examples.append({"modèle": model_id, "exemple_pédagogique": "had lfilm zwin بزاف", "sous_tokens_réels": tokenizer.tokenize("had lfilm zwin بزاف")})
    except OSError:
        print("Tokeniseur absent du cache local :", model_id, "— aucun téléchargement automatique ici.")
display(pd.DataFrame(tokenizer_examples))
print("Sur CPU sans CUDA, les entraînements complets E peuvent rester non exécutés ; cela doit figurer dans le bilan.")
''')
lesson("E.2 — Ajuster les deux modèles", "Exécuter une comparaison standard/Arabizi à conditions égales.", "L'initialisation adaptée à une écriture peut aider ou nuire aux autres.", "Une boucle par checkpoint et graine ; sauvegarde de l'encodeur, tête, tokenizer, configuration et historique.", "Une phrase latine peut être mieux tokenisée par Arabizi ; cela reste à mesurer.", "[Baert et al.](https://aclanthology.org/2020.coling-main.51/) et [Gaanoun et al.](https://link.springer.com/article/10.1007/s41060-023-00498-2).", '''
if TRAIN_E:
    run_campaign("E", device=DEVICE)
display(plan_table().query("family == 'E'"))
smoke_file = ROOT / "results/pretrained_smoke.json"
if smoke_file.exists():
    display(pd.DataFrame(json.loads(smoke_file.read_text(encoding="utf-8"))))
    print("Ces deux pas techniques ne sont pas les entraînements E : aucun checkpoint ni F1 de benchmark n'a été produit.")
print("Aucun fine-tuning n'est déclaré exécuté sur la seule base d'un test de forme.")
''')
lesson("E.3 — Comparaison des résultats disponibles", "Comparer les F1 et la stabilité E.", "La meilleure graine isolée ne résume pas une méthode.", "Moyenne des trois graines sur dev pour la variante ; graine 42 prédéfinie comme représentant final.", "Un gain moyen accompagné d'une forte variance demande de la prudence.", "[NLNDE](https://aclanthology.org/2023.semeval-1.68/) souligne l'effet des sources et des initialisations.", '''
e_results = show_family("E")
if len(e_results):
    display(e_results.groupby("variant").agg(f1=("weighted_f1", "mean"), macro=("macro_f1", "mean"), std=("weighted_f1", "std"), graines=("seed", "nunique")))
curve_e = plot_histories("E")
if curve_e:
    display(Image(filename=str(curve_e)))
print("Aucun apprentissage sur un corpus d'un autre pays n'est réalisé.")
''')
recap("E", "Implémenté le fine-tuning des deux checkpoints sur ary seulement.", "Fine-tuning adapte les représentations au sentiment local.", "Temps CPU, mémoire d'AdamW et équité des budgets.", "Accumulation de gradients, même LR et mêmes graines ; état des runs visible.", "Sans résultats E complets, on ne peut pas conclure qu'il dépasse les autres familles.", "Pourquoi comparer les écritures ? Un modèle adapté au latin peut présenter un compromis sur l'arabe.")

lesson("3 — Problèmes rencontrés et solutions inspirées de la recherche scientifique", "Relier observation, publication, solution et essai réel.", "Une solution proposée doit être distinguée d'une solution évaluée.", "Produire les statuts à partir des métriques dev enregistrées.", "Un delta fastText positif sur dev est décrit, sans promettre un gain général.", "Les sources primaires sont liées dans chaque ligne et détaillées dans research/papers.md.", '''
solutions = solutions_table()
display(solutions)
print("Les mentions 'testée avec amélioration' décrivent seulement le delta moyen dev des runs disponibles, sans test de significativité.")
''')

lesson("4 — Sélection scientifique avant test", "Choisir une variante par famille et un candidat global sur dev.", "Choisir sur test transformerait l'évaluation finale en validation supplémentaire.", "Moyenne Weighted-F1 dev des graines ; graine 42 représentante ; verrouillage des variantes, règles et hashes.", "La famille absente de résultats ne peut pas gagner par réputation.", "[SemEval](https://aclanthology.org/2023.semeval-1.315/) et règles du TP.", '''
if len(results_table()):
    display(select_on_dev())
else:
    print("Aucune sélection possible : aucun run.")
if LOCK_AND_TEST:
    lock = lock_selection()
    final_evaluation()
lock_path = ROOT / "results/selection_lock.json"
if lock_path.exists():
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    display(pd.DataFrame(lock["selected_variants"]))
    print("Candidat fixé sur dev :", lock["candidate"], "— représentant : graine", lock["representative_seed"])
else:
    print("Pas encore verrouillé. Terminer les comparaisons souhaitées avant de passer LOCK_AND_TEST à True.")
''')

md("# Partie F — Robustesse, après verrouillage\n\nNous pouvons maintenant analyser les labels test à titre d'évaluation. **Aucun résultat F ne modifie le modèle, les décalages ou les hyperparamètres.** Les faibles effectifs mixtes et les collisions détectées limitent les conclusions. Une analyse a posteriori décrit ce qui s'est passé ; elle n'autorise pas un nouveau choix utilisant test.")
lesson("F1 — Changement de distribution et décalages", "Comparer les proportions dev/test et l'effet des règles déjà fixées.", "Une règle optimisée sur dev peut se transférer imparfaitement si les proportions changent.", "Lire les distributions produites après verrouillage, puis comparer argmax et décalages dev sur test sans réoptimisation.", "Un décalage neutre adapté à dev peut devenir moins favorable sur test.", "[Revue](https://doi.org/10.1016/j.jksuci.2023.101570), variation entre corpus.", '''
shift = show_csv("f1_label_shift.csv")
final_conflicts = show_csv("audit/conflicts_including_test_final.csv")
if len(shift):
    shift.set_index(shift.columns[0]).plot.bar(color=[COLORS[1], COLORS[2]])
    plt.ylabel("Proportion"); plt.title("F1 — labels après verrouillage")
    plt.tight_layout(); plt.savefig(ROOT / "results/figures/F1_shift.png", dpi=140); plt.show()
recorded = results_table()
test_rows = recorded.loc[recorded.split == "test"] if len(recorded) else recorded
display(test_rows[[c for c in ["family", "variant", "seed", "decision", "weighted_f1", "macro_f1"] if c in test_rows]].head(30))
print("Analyse a posteriori : aucun décalage optimal test n'est recherché ni réinjecté.")
if (ROOT / "results/comparison.csv").exists():
    conditional_labels, corpus_markers, diagnostic_oov = posthoc_diagnostics()
    display(conditional_labels.loc[conditional_labels.script.eq("latin")])
    display(corpus_markers)
    show_figure("F1_conditional_shift.png")
    display(Markdown((ROOT / "results/generalization_analysis.md").read_text(encoding="utf-8")))
''')
lesson("F2 — Apprendre une écriture, tester l'autre", "Mesurer le transfert arabe→latin et latin→arabe au sein de ary.", "Une bonne moyenne globale peut masquer la dépendance à l'écriture vue au train.", "Deux modèles union appris uniquement sur le sous-ensemble train correspondant ; décalages choisis sur dev ; test sur l'écriture opposée.", "L'apprentissage arabe n'inclut aucun texte latin train dans cette expérience.", "[Baert et al.](https://aclanthology.org/2020.coling-main.51/), spécificité de l'Arabizi.", '''
cross_script = show_csv("f2_cross_script.csv")
if len(cross_script):
    print(interpret(cross_script))
print("Les tailles des sous-ensembles et les classes sont enregistrées. Cette expérience ne transfère jamais entre pays.")
''')
lesson("F3 — Scores par écriture et OOV test", "Étudier les performances sur chaque alphabet.", "Le Weighted-F1 global peut être dominé par les groupes fréquents.", "Vocabulaire train inchangé ; calcul OOV et F1 par groupe après verrouillage.", "Un taux OOV latin plus fort signalerait une différence de couverture, sans prouver la cause des erreurs.", "[fastText](https://aclanthology.org/Q17-1010/) et [DarijaBERT](https://link.springer.com/article/10.1007/s41060-023-00498-2).", '''
if (ROOT / "results/selection_lock.json").exists():
    all_oov = oov_table({"train": train, "dev": dev, "test": test_texts}, vocab)
    all_oov.to_csv(ROOT / "results/oov.csv", index=False)
    display(all_oov)
    for file in sorted((ROOT / "results").glob("*_s42_scripts.csv")):
        print(file.stem)
        display(pd.read_csv(file))
else:
    print("Non exécuté : sélection non verrouillée.")
print("Translittération facultative non exécutée : aucun système validé n'est ajouté. Les scores mixtes à faible effectif exigent prudence.")
''')
lesson("F4 — Retirer emojis et ponctuation répétée", "Mesurer la sensibilité du candidat final à des indices de polarité.", "Un modèle peut dépendre fortement des marqueurs émotionnels de surface.", "Uniquement tweets test polarisés initialement bien classés, puis suppression déterministe et nouvelle prédiction avec checkpoint inchangé.", "« زوين 😍!!! » devient « زوين » ; le sentiment contextuel peut rester ou devenir ambigu.", "[Revue SA dialectale](https://doi.org/10.1016/j.jksuci.2023.101570).", '''
perturbations = show_csv("f4_perturbations.csv", 12)
comparison_file = ROOT / "results/comparison.csv"
if comparison_file.exists():
    comparison = pd.read_csv(comparison_file)
    if "robustness_f4" in comparison:
        display(comparison[["family", "robustness_f4"]].dropna())
print("Le taux de changement porte sur le sous-ensemble effectivement modifié et initialement correct. Il ne décrit pas la robustesse de tous les tweets.")
''')
lesson("F5 — Trois graines, bootstrap et McNemar", "Quantifier deux sources d'incertitude.", "Un écart faible peut venir de l'échantillon ou de l'initialisation.", "1 000 bootstrap appariés par lignes pour l'IC du F1 ; écart-type entre graines ; McNemar exact sur deux modèles sélectionnés sur dev.", "McNemar compte les désaccords de justesse, pas les différences directes de F1.", "[NLNDE](https://aclanthology.org/2023.semeval-1.68/) utilise plusieurs graines ; le bootstrap ici est notre analyse.", '''
if comparison_file.exists():
    uncertainty = pd.read_csv(comparison_file)
    columns = [c for c in ["family", "variant", "weighted_f1", "ci_low", "ci_high", "bootstrap", "test_seed_mean", "test_seed_std", "n_seeds"] if c in uncertainty]
    display(uncertainty[columns])
else:
    print("Bootstrap non exécuté.")
mcnemar_file = ROOT / "results/f5_mcnemar.json"
if mcnemar_file.exists():
    display(pd.Series(json.loads(mcnemar_file.read_text(encoding="utf-8")), name="McNemar exact").to_frame())
else:
    print("McNemar non exécuté : au moins deux familles évaluées sont nécessaires.")
print("IC conditionnel au modèle entraîné ; graines et bootstrap ne sont pas interchangeables. Les trois répétitions TF-IDF peuvent être identiques.")
''')
lesson("F6 — Trente erreurs et revue humaine", "Construire un dossier de 30 erreurs diversifiées.", "Le score seul ne révèle ni ironie, ni erreur d'annotation, ni code-switching.", "Échantillonnage en tour de rôle entre catégories heuristiques, jointure exacte avec les votes disponibles, validation humaine laissée explicite.", "Une erreur étiquetée 'ironie possible' reste une hypothèse tant qu'une personne n'a pas lu le contexte.", "[Votes officiels AfriSenti](https://github.com/afrisenti-semeval/afrisent-semeval-2023/tree/main/data_with_annotators_labels).", '''
errors = show_csv("f6_30_erreurs.csv", 30)
if len(errors):
    display(errors.categorie_provisoire.value_counts().to_frame("erreurs"))
    print("Exemples :", len(errors), "— validation humaine :", errors.validation_humaine.value_counts().to_dict())
print("Les mentions anonymisées peuvent empêcher la jointure aux votes. Aucun vote test n'a servi à choisir les modèles. Les catégories humaines restent provisoires.")
review_file = ROOT / "results/error_analysis.md"
if review_file.exists():
    display(Markdown(review_file.read_text(encoding="utf-8")))
''')

md("# Partie G — Comparaison finale et limites\n\nLe tableau comporte toujours les **cinq familles**. Les cellules de performance manquantes correspondent à des expériences non exécutées, jamais à un score nul inventé. Les mesures test principales utilisent la graine 42 prédéfinie ; les colonnes moyenne/écart-type décrivent les graines disponibles. La latence CPU est mesurée après chargement et échauffement, sur 30 prédictions unitaires. La taille des poids additionne la tête et, pour B/D, l'encodeur externe réellement présent dans le cache ; tokenizer et fichiers de configuration ne sont pas comptés comme poids.")
lesson("G.1 — Tableau, matrice et conclusion scientifique", "Comparer qualité, temps, taille, paramètres, écritures et incertitude.", "Un choix pratique peut dépendre autant du coût que du F1.", "Afficher comparison.csv et la matrice 3×3 du candidat fixé sur dev ; expliquer les performances test sans changer le candidat.", "Le plus rapide peut être utile à l'application même si une autre méthode a un F1 légèrement supérieur.", "[AfriSenti](https://arxiv.org/abs/2302.08956) et [SemEval](https://aclanthology.org/2023.semeval-1.315/).", '''
final = show_csv("comparison.csv", 5)
show_figure("modele_final.png")
if len(final):
    measured = final.dropna(subset=["weighted_f1"])
    print(interpret(measured))
    if lock_path.exists():
        print("Choix scientifique effectué sur dev :", json.loads(lock_path.read_text(encoding="utf-8"))["candidate"])
print("Limites : petit dev, rareté du mixte, collisions textuelles, contamination externe non exclue, catégories d'erreurs provisoires, runs manquants explicités.")
print("Les colonnes sensitivity_* décrivent une analyse a posteriori sans collisions train ; elles ne remplacent pas les métriques officielles et n'ont servi à aucun choix.")
''')

lesson("G.2 — Oral, Q1–Q12 provisoires et état de livraison", "Préparer une explication autonome et honnête.", "Le professeur doit pouvoir distinguer code implémenté, vérification technique et résultat expérimental.", "Douze thèmes avec explication, mesures disponibles, artefact, problème, solution, conclusion et source ; énoncés officiels encore nécessaires pour une correspondance exacte.", "Dire 'architecture vérifiée' ne signifie pas 'modèle entraîné et meilleur'.", "Bibliographie vérifiée dans research/papers.md.", '''
display(Markdown(oral_report()))
status = plan_table()
status.to_csv(ROOT / "results/execution_status.csv", index=False)
display(status)
display(pd.DataFrame({"livrable": ["Notebook", "Sources", "Résultats authentiques", "Application", "Q1–Q12 officielles"], "chemin_ou_état": ["notebooks/TP2_Darija.ipynb", "research/papers.md", "results/results.csv", "python -m streamlit run app.py", "Énoncés absents : thèmes provisoires seulement"]}))
print("Pour l'oral : annoncer le périmètre ary, expliquer le protocole, lire les résultats réels, puis présenter les limites.")
''')

notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Darija-Sentiment (Conda)", "language": "python", "name": "darija-sentiment"}, "language_info": {"name": "python", "version": "3.11.5"}})
assert sum(c.cell_type == "code" for c in cells) == 40
path = ROOT / "notebooks/TP2_Darija.ipynb"
path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, path)
print(f"{path} : {len(cells)} cellules dont 40 code.")
