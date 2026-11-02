"""Génère le notebook autonome, sans exécuter de cellule ni entraîner de modèle."""
from pathlib import Path
import nbformat as nbf
import base64, hashlib, io
import zlib
import os
import shutil
from datetime import datetime, timezone
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))
def lesson(title, objective, principle, code):
    md(f"## {title}\n\n**Objectif :** {objective}\n\n**Méthode et interprétation :** {principle}")
    cells.append(nbf.v4.new_code_cell(code.strip()))

md("""
# TP2 · Darija-Sentiment · nouvelle campagne A–G

**Notebook préparé, non exécuté. Aucun nouveau score n'est disponible avant votre exécution.**

Le sentiment marocain `ary` est la tâche principale : négatif, neutre, positif. Le corpus algérien
`arq/train` intervient **uniquement dans E2**, avec une seconde tête sur le même encodeur.
Les partitions officielles ary restent intactes. Les exemples pédagogiques ne sont pas ajoutés au train.

**Démarrage :** importer ce notebook avec Python 3.10–3.13, lire les options au début
de la première cellule, puis exécuter dans l'ordre. Le code et les données ary sont incorporés : même
importé seul, le notebook peut les restaurer dans un dossier inscriptible. Le démarrage installe
les dépendances manquantes/incompatibles à l'étape 1.6 si `INSTALL_DEPENDENCIES=True`. Un redémarrage de session peut
être nécessaire une fois si pip remplace des bibliothèques déjà chargées. La section 2 vérifie les
imports, les données, CUDA, AraVec et l'accès aux configurations des modèles avant les entraînements.
AraVec peut être téléchargé automatiquement depuis sa source officielle et ses trois fichiers sont
vérifiés par SHA-256 ; les poids des autres modèles sont téléchargés lors de leur utilisation.
Une connexion Internet reste nécessaire si ces ressources ne sont pas déjà disponibles.
Python 3.10–3.13 accepte NumPy 1 ou 2 avec Gensim 4.4 : les versions compatibles de NumPy,
SciPy et PyTorch du runtime sont conservées. Installer seulement un module non chargé ne demande
pas de redémarrage. Si une bibliothèque chargée est remplacée, reprendre après un vrai redémarrage ;
réexécuter la cellule seule ne suffit pas.

**Sorties :** nouvelle campagne `campaign_v4`, sur Google Drive par défaut dans Colab.
L'étape 1.4 sauvegarde les anciennes campagnes encore dans `/content` avant pip.
Les sorties observées sont conservées dans `TP2_Darija_observe_avant_v4.ipynb`.
Les notebooks historiques et leur HTML sont conservés sous le suffixe `_historique`.
Un nouveau nom de campagne est requis si le code ou les hyperparamètres changent après le démarrage.
Un run terminé est réutilisé ; un réseau interrompu reprend après sa dernière époque sauvegardée.

**Protocole :** train apprend ; val (= dev) choisit les variantes, les époques et les décalages multiclasse ;
test est ouvert après vérification de toutes les parties et verrouillage. Les réseaux utilisent 42/43/44 ;
les modèles déterministes A, B et D+LR sont exécutés une seule fois.
Le représentant est toujours 42. Weighted-F1 assure la comparaison AfriSenti ; Macro-F1 et scores par écriture
complètent l'analyse. B utilise des modèles fixes : une seule exécution déterministe par variante.

**Limite de cette refonte :** le test ary a déjà été observé dans l'étude historique. Les résultats de cette
nouvelle campagne seront des réévaluations exploratoires, même si aucun réglage automatique n'utilise ses labels.
Une confirmation indépendante exige de nouveaux exemples annotés. Aucun meilleur score n'est garanti.

Sources : [AfriSenti](https://aclanthology.org/2023.emnlp-main.862/),
[SemEval-2023 Task 12](https://aclanthology.org/2023.semeval-1.315/),
[DarijaBERT](https://doi.org/10.1007/s41060-023-00498-2),
[fastText](https://aclanthology.org/Q17-1010/).
""")

# Le contenu embarque code et petits fichiers de donnees, jamais de poids ni resultats.
bundle_files = list((ROOT / "src").glob("*.py"))
bundle_files += list((ROOT / "data/ary").glob("*.tsv"))
bundle_files += [p for p in (ROOT / "data/review").glob("*") if p.is_file()]
bundle_files += [p for p in (ROOT / "data/provenance").glob("*") if p.is_file()]
bundle_files += [ROOT / name for name in ("requirements.txt", "requirements-notebook.txt", "research/questions_officielles.md", "research/papers.md", "scripts/notebook_bootstrap.py", "scripts/notebook_setup.py", "scripts/backup_results.py", "scripts/cache_pretrained.py")]
archive_buffer = io.BytesIO()
with ZipFile(archive_buffer, "w", ZIP_DEFLATED) as bundle:
    for path in sorted(bundle_files):
        entry = ZipInfo(str(path.relative_to(ROOT)).replace("\\", "/"))
        entry.compress_type = ZIP_DEFLATED
        bundle.writestr(entry, path.read_bytes())
embedded = base64.b64encode(archive_buffer.getvalue()).decode("ascii")
digest = hashlib.sha256(archive_buffer.getvalue()).hexdigest()
bootstrap = r'''
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile
import os, sys, base64, hashlib, io

# Options à modifier avant de lancer le notebook.
PROJECT_DIR = os.environ.get("DARIJA_PROJECT_DIR", "")
CAMPAIGN_NAME = "campaign_v4"
OUTPUT_DIR = ""  # Facultatif : dossier persistant, par exemple sur Drive.
USE_GOOGLE_DRIVE = True
BACKUP_PREVIOUS_COLAB_RUNS = True
RUN_SMOKE_TEST = True
INSTALL_DEPENDENCIES = True
AUTO_DOWNLOAD_ARAVEC = True
CHECK_MODEL_ACCESS = True
RUN_TRAINING = True
RUN_FINAL_EVALUATION = True
DEVICE = "auto"
ARAVEC_PATH = ""
CONFIG_OVERRIDES = {}  # {"micro_batch_e": 1} si la mémoire GPU est insuffisante.
print("[1/6] Options chargées — Python", sys.version.split()[0])
'''
lesson("1.1 · Imports et options", "Définir les options sans installation ni téléchargement.",
       "Exécuter 1.1 à 1.6 une par une. La sauvegarde sur Drive précède toute installation. Ne pas poursuivre après une erreur.", bootstrap)

archive_cell = "# Code et données intégrés : conserver ces deux lignes.\n"
archive_cell += "EMBEDDED_PROJECT_SHA256 = " + repr(digest) + "\nEMBEDDED_PROJECT = " + repr(embedded) + "\n"
archive_cell += r'''
# Charger les utilitaires depuis l'archive vérifiée.
payload = base64.b64decode(EMBEDDED_PROJECT, validate=True)
if hashlib.sha256(payload).hexdigest() != EMBEDDED_PROJECT_SHA256:
    raise ValueError("Archive endommagée : réimporter le notebook.")
setup = SimpleNamespace()
with ZipFile(io.BytesIO(payload)) as archive:
    exec(compile(archive.read("scripts/notebook_bootstrap.py"), "notebook_bootstrap.py", "exec"), vars(setup))
print("[2/6] Archive vérifiée et utilitaires chargés")
'''
lesson("1.2 · Vérifier l'archive intégrée", "Contrôler l'intégrité du code et des données.",
       "Cette étape ne lance pas pip. Une erreur ici concerne l'archive, son empreinte ou le chargement des utilitaires.", archive_cell)

lesson("1.3 · Restaurer le projet", "Trouver ou restaurer les fichiers du projet.",
       "Une erreur ici concerne PROJECT_DIR, la restauration des fichiers ou les droits d'accès. Aucune bibliothèque n'est installée.", r'''
PROJECT = setup.bootstrap_project(EMBEDDED_PROJECT, EMBEDDED_PROJECT_SHA256, PROJECT_DIR)
print("[3/6] Projet disponible :", PROJECT)
''')

lesson("1.4 · Drive et sauvegarde de l'exécution précédente", "Sauvegarder les résultats Colab avant pip ou redémarrage.",
       "Dans la même session Colab, autoriser le montage Drive. Les campagnes présentes dans les anciens dossiers runtime sont copiées avec vérification SHA-256. Une session déjà supprimée ne peut pas être récupérée. Les anciennes mesures restent séparées de campaign_v4.", r'''
setup.PROJECT_BACKUP_SCRIPT = PROJECT / "scripts/backup_results.py"
OUTPUT_DIR = setup.prepare_storage(CAMPAIGN_NAME, OUTPUT_DIR, USE_GOOGLE_DRIVE, BACKUP_PREVIOUS_COLAB_RUNS)
print("[4/6] Sauvegarde préalable terminée ; sortie persistante :", OUTPUT_DIR or "projet local")
''')

lesson("1.5 · Préparer les résultats", "Vérifier le nom de campagne et le dossier de sortie.",
       "Une erreur ici concerne CAMPAIGN_NAME ou OUTPUT_DIR. Choisir un dossier inscriptible ; les résultats existants sont conservés.", r'''
if Path(CAMPAIGN_NAME).name != CAMPAIGN_NAME or CAMPAIGN_NAME in {"", ".", ".."} or "\\" in CAMPAIGN_NAME:
    raise ValueError("CAMPAIGN_NAME doit être un simple nom de dossier.")
OUTPUT = (Path(OUTPUT_DIR).expanduser() if OUTPUT_DIR else PROJECT / "experiments" / CAMPAIGN_NAME).resolve()
if not setup.writable_directory(OUTPUT):
    raise PermissionError(f"Dossier non inscriptible : {OUTPUT}. Modifier OUTPUT_DIR.")
print("[5/6] Dossier de résultats accessible :", OUTPUT)
''')

lesson("1.6 · Vérifier et installer les dépendances", "Isoler la vérification des versions et l'installation pip.",
       "Le message de redémarrage reçu précédemment provient de cette étape. Si des bibliothèques chargées sont remplacées, redémarrer la session et reprendre à 1.1. En cas d'échec pip, lire ses lignes ERROR au-dessus du traceback.", r'''
print("[6/6] Vérification des dépendances — installation autorisée :", INSTALL_DEPENDENCIES)
setup.ensure_dependencies(PROJECT, install=INSTALL_DEPENDENCIES)
print("[6/6] Dépendances prêtes. Continuer avec la section 2 : configuration et ressources.")
''')
lesson("2 - Configuration et verification des ressources", "Verifier les ressources avant A.",
       "Adapter les options de la cellule 1. Les fichiers AraVec manquants sont telecharges et verifies avant la campagne, ainsi que l'acces aux configurations des modeles. Les poids complets restent charges au moment de leur utilisation.",
       (ROOT / "scripts/notebook_setup.py").read_text(encoding="utf-8"))
lesson("2.1 · Essai court de toutes les familles", "Vérifier entraînement, sauvegarde et rechargement sur de petits lots.",
       "A, B, C, D, E1, E2 et P utilisent des sorties temporaires. Dans Colab, les véritables encodeurs sont utilisés sur un petit sous-ensemble train/dev. Aucun score de cet essai n'est un résultat scientifique ; la mémoire sur des lots complets reste à vérifier.", '''
if RUN_SMOKE_TEST:
    from src.smoke import smoke_check
    display(pd.Series(smoke_check(CONFIG["device"], real_pretrained=True)))
''')
lesson("3 · Audit des partitions", "Repérer doublons et collisions sans labels test.",
"Les fichiers officiels ne sont pas modifiés. Le test chargé ici contient uniquement ses textes ; les contradictions incluant ses labels sont étudiées après verrouillage.", '''
train, dev, test_texts = [load_split(s) for s in ("train", "dev", "test")]
counts, collisions, conflicts = audit_data()
display(counts); display(collisions); display(conflicts)
from src.validation import audit_near_duplicates
display(audit_near_duplicates())
assert "label" not in test_texts
''')
lesson("4 · Lire les tweets", "Comprendre arabe, latin et mélange d'écritures.",
"Une écriture latine peut contenir du français ou de l'Arabizi ; ce détecteur n'identifie pas la langue. Nettoyage v2 : retirer URLs/mentions, conserver négation, accents, emojis, chiffres et répétitions.", '''
for script, group in train.groupby("script"):
    display(group[["row_id", "tweet", "text", "label"]].head(3).assign(ecriture=script))
''')
lesson("A.1 · Distribution des labels", "Afficher les effectifs et proportions par écriture.",
"Le global et les sous-groupes répondent à deux questions différentes. Les distributions test seront ajoutées en F1 après verrouillage.", '''
distribution = campaign.distributions()
display(distribution)
for script in ["global", "arabic", "latin", "mixed"]:
    part = distribution.loc[distribution.script.eq(script)]
    if not part.empty:
        part.pivot(index="label", columns="split", values="proportion").reindex(LABELS).plot.bar(title=f"Labels · {script}")
        plt.ylabel("Proportion"); plt.tight_layout(); plt.show()
''')
lesson("A.2 · Longueurs et troncature", "Mesurer les informations susceptibles d'être tronquées.",
"Les longueurs regex ne sont pas les longueurs WordPiece. Les limites sont fixées avant test : 128 tokens C/D/E, 512 caractères C4. TF-IDF utilise le texte complet.", '''
lengths = pd.concat([f.assign(split=s, token_count=f.text.map(lambda t: len(tokens(t))), char_count=f.text.str.len()) for s, f in [("train", train), ("dev", dev)]])
display(lengths.groupby(["split", "script"])[["token_count", "char_count"]].describe())
lengths.boxplot(column="token_count", by=["split", "script"], rot=30, showfliers=False)
plt.suptitle(""); plt.title("Longueurs avant troncature"); plt.tight_layout(); plt.show()
''')
lesson("A.3 · Majorité et raccourci d'écriture", "Construire deux points de comparaison triviaux.",
"Les majorités sont apprises sur train seulement, puis appliquées à val. Un score script-only élevé révèle une régularité exploitable du corpus, sans démontrer le mécanisme causal des autres modèles.", '''
display(campaign.shortcuts())
display(pd.crosstab(train.script, train.label, normalize="index").reindex(columns=LABELS))
''')
lesson("A.4 · Entraîner les baselines", "Comparer mots, caractères et combinaison équilibrée.",
"Mots 1–2 et caractères 2–5 ; LR class_weight=balanced. La combinaison compare C=0.25/1/4 sur val. Une seule graine car ces modèles sont déterministes. Les symboles sont conservés dans la référence ; P compare les nettoyages. F2 entraîne aussi union sur une seule écriture.", '''
if RUN_TRAINING:
    campaign.run_part("A", CONFIG)
else:
    print("A non lancé : RUN_TRAINING=False")
''')
lesson("A.5 · Résultats et décision", "Comparer F1 brut et F1 avec règle val.",
"La règle multiclasse est argmax(log(p_c)+b_c), premier décalage fixé à zéro. Les 25 couples de décalages sont choisis sur val ; le gain val n'est pas une évaluation indépendante.", '''
a_summary = dev_summary("A")
display(a_summary)
if not a_summary.empty:
    a_summary.set_index("variant")[["weighted_f1", "macro_f1"]].plot.bar(title="A · validation")
    plt.tight_layout(); plt.show()
rows = results_table()
if not rows.empty:
    display(rows.loc[rows.family.eq("A") & rows.split.eq("dev"), ["variant", "seed", "decision", "weighted_f1", "macro_f1"]])
''')
lesson("B.1 · Transfert sans entraînement local", "Exécuter les trois NLI et CAMeLBERT.",
"Modèle NLI imposé : MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7. B2 : CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment. Leurs poids restent fixes ; tous auront des scores test après verrouillage.", '''
if RUN_TRAINING:
    campaign.run_part("B", CONFIG)
''')
lesson("B.2 · Hypothèses NLI", "Expliquer comment une classe devient une hypothèse.",
"Chaque tweet est une prémisse appariée à trois hypothèses. Le modèle utilise les logits d'entailment ; l'auto-attention traite conjointement les deux textes. Trois formulations sont testées, une par langue ; ce n'est pas une étude de paraphrases à langue constante.", '''
from src.models import HYPOTHESES
display(pd.DataFrame(HYPOTHESES, index=LABELS))
display(dev_summary("B"))
''')
lesson("B.3 · Sensibilité et écritures", "Comparer les décisions et les sous-groupes sur val.",
"Les écarts après réglage intègrent formulation et décalages. CAMeLBERT n'est pas un modèle Arabizi : mesurer plutôt que supposer sa couverture. Une absence de contamination externe ne peut pas être certifiée.", '''
rows = results_table()
if not rows.empty:
    b = rows.loc[rows.family.eq("B") & rows.split.eq("dev")]
    display(b[["variant", "decision", "weighted_f1", "macro_f1"]])
    b.pivot(index="variant", columns="decision", values="weighted_f1").plot.bar(title="B · sensibilité sur val")
    plt.tight_layout(); plt.show()
    for path in sorted((ROOT / "results").glob("B_*_dev_tuned.csv")):
        p = pd.read_csv(path)
        display(pd.DataFrame([{"script": s, **campaign.metric_scores(g.label, g.prediction)} for s, g in p.groupby("script")]).assign(run=path.stem))
''')
lesson("C.1 · Vocabulaire et OOV", "Construire le vocabulaire uniquement sur train.",
"Fréquence minimale 2 ; PAD=0 et UNK=1. Pour fastText v2, les UNK disposent de vecteurs composés par sous-mots, sans apprendre de vocabulaire ou de paramètres sur val/test. Cela ne garantit pas qu'ils représentent bien le sentiment.", '''
vocab = build_vocab(train.text, min_freq=2)
display(pd.DataFrame(encode_texts(train.text.head().tolist(), vocab, CONFIG["max_length"])[:, :15]))
from src.preprocess import oov_table
display(oov_table({"train": train, "dev": dev}, vocab))
print("Taille du vocabulaire :", len(vocab))
''')
lesson("C.2 · C1, C2, C3 et C4", "Entraîner les réseaux à budget commun.",
"BiLSTM, BiLSTM-attention maison, Transformer 2 couches / d_model=128 / 4 têtes avec et sans positions ; embeddings aléatoires, AraVec et fastText. Char-CNN utilise son vocabulaire de caractères. Maximum 15 époques et early stopping sur val ; trois graines.", '''
if RUN_TRAINING:
    campaign.run_part("C", CONFIG)
''')
lesson("C.3 · Architecture, embeddings et positions", "Comparer les variantes et leur variabilité.",
"Lire moyenne et écart-type sur les trois graines. L'ablation positionnelle compare la même architecture, les mêmes embeddings et le même budget ; le meilleur score isolé ne suffit pas.", '''
c_summary = dev_summary("C")
display(c_summary)
if not c_summary.empty:
    c_summary.set_index("variant").weighted_f1.plot.bar(yerr=c_summary.set_index("variant")["std"], title="C · moyenne val ± écart-type", figsize=(13, 4))
    plt.tight_layout(); plt.show()
''')
lesson("C.4 · Déséquilibre sur le meilleur C", "Choisir d'abord le meilleur modèle C sur val.",
"La moyenne des trois graines décide de l'architecture ET des embeddings. On conserve cette référence, puis compare perte pondérée et suréchantillonnage train uniquement. La sélection n'utilise jamais les résultats test.", '''
if RUN_TRAINING:
    display(campaign.choose_c_for_balance())
    campaign.run_part("C_balance", CONFIG)
''')
lesson("C.5 · Tableau du déséquilibre", "Séparer effet de l'entraînement et effet de la décision.",
"Le tableau croise sans rééquilibrage, perte pondérée et suréchantillonnage avec argmax brut et règle val. Suréchantillonner avant séparation ferait circuler des copies entre apprentissage et validation.", '''
selection_file = ROOT / "results/c_imbalance_selection.json"
if selection_file.exists():
    base = json.loads(selection_file.read_text(encoding="utf-8"))["variant"]
    rows = results_table()
    balance = rows.loc[rows.family.eq("C") & rows.split.eq("dev") & rows.variant.isin([base, base+"_weighted", base+"_oversampling"])]
    display(balance.groupby(["variant", "decision"]).agg(f1=("weighted_f1", "mean"), macro=("macro_f1", "mean"), std=("weighted_f1", "std")).reset_index())
''')
lesson("C.6 · OOV AraVec et fastText par écriture", "Distinguer absence du lexique et vecteur disponible.",
"Un mot absent du lexique fastText peut recevoir un vecteur composé par sous-mots ; AraVec ne le compose pas. Les vecteurs AraVec manquants sont initialisés aléatoirement dans le réseau. Ne pas confondre OOV du vocabulaire C et couverture des embeddings.", '''
if (ROOT / "results/c_imbalance_selection.json").exists():
    coverage = campaign.embedding_oov(CONFIG)
    display(coverage)
    coverage.loc[coverage.split.eq("dev")].pivot(index="script", columns="embedding", values="lexical_oov_rate").plot.bar(title="OOV lexical des embeddings · val")
    plt.tight_layout(); plt.show()
''')
lesson("C.7 · Attention sur cinq exemples", "Observer de vrais poids d'attention.",
"Cinq textes val prédéfinis ; meilleure variante d'attention sur val, checkpoint graine42. Chercher emojis, opinion, français, sans transformer ces poids descriptifs en explication causale.", '''
attention_c = attention_examples("C", n=5)
display(attention_c); show_attention(attention_c)
''')
lesson("D.1 · DarijaBERT gelé", "Comparer CLS+LR, moyenne+LR et états+BiLSTM-attention.",
"L'encodeur ne reçoit aucun gradient. La moyenne masque PAD et inclut les tokens spéciaux. L'extraction est mise en cache par données, nettoyage et longueur ; son coût est enregistré séparément.", '''
if RUN_TRAINING:
    campaign.run_part("D", CONFIG)
''')
lesson("D.2 · Qualité et coût d'apprentissage", "Comparer les trois têtes sur val.",
"Peu de paramètres entraînables ne signifie pas une inférence peu coûteuse : DarijaBERT reste nécessaire. La latence CPU complète sera mesurée en G, sans cache de représentations.", '''
display(dev_summary("D"))
rows = results_table()
if not rows.empty:
    display(rows.loc[rows.family.eq("D") & rows.split.eq("dev") & rows.decision.eq("dev_offsets"), ["variant", "seed", "train_seconds", "trainable_parameters", "weighted_f1"]])
''')
lesson("D.3 · Attention D2 sur cinq exemples", "Visualiser les sous-tokens du checkpoint D2.",
"Les unités sont celles du tokenizer BERT. Les poids sont normalisés et PAD est masqué ; une forte attention sur un emoji ne prouve pas qu'il cause la décision.", '''
attention_d = attention_examples("D", n=5)
display(attention_d); show_attention(attention_d)
''')
lesson("E.1 · Protocole de fine-tuning", "Préparer une comparaison équitable.",
"DarijaBERT et DarijaBERT-Arabizi : AdamW 2e-5, perte pondérée à partir de train, batch effectif 32, maximum 5 époques, minimum 3 avant arrêt anticipé, warmup 10 %, trois graines. Micro-batch 2 et accumulation ; précision mixte sur CUDA et checkpointing de gradients. Les décalages sont choisis sur val après sélection de l'époque.", '''
display(pd.Series({"lr": 2e-5, "batch_effectif": 32, "micro_batch": CONFIG["micro_batch_e"], "epochs": CONFIG["e_epochs"], "warmup": 0.1, "weighted_loss": True, "longueur": CONFIG["max_length"], "seeds": CONFIG["seeds"]}).to_frame("E1/E2"))
print("AraBizi spécialisé peut mieux couvrir le latin ; ce n'est pas un résultat avant mesure.")
''')
lesson("E.2 · E1 : deux modèles fine-tunés", "Adapter les poids des encodeurs à ary.",
"Six runs complets, sans score inventé. Le fine-tuning CPU reste possible mais coûteux. Réduire le micro-batch conserve le batch effectif ; changer un réglage impose un nouveau nom de campagne.", '''
if RUN_TRAINING:
    campaign.run_part("E1", CONFIG)
display(dev_summary("E"))
''')
reduced_payload = base64.b64encode(zlib.compress((ROOT / "scripts/reduce_remaining.py").read_bytes())).decode("ascii")
reduced_loader = '# Plan court E2/P, applicable au kernel actif sans redemarrage.\nimport base64 as _plan_b64, zlib as _plan_zlib\nexec(_plan_zlib.decompress(_plan_b64.b64decode(' + repr(reduced_payload) + ')).decode("utf-8"), globals())\nactivate_reduced_plan(CONFIG, globals())\n\n'
lesson("E.3 · E2 court : encodeur partagé, têtes ary/arq", "Comparer E1 à une seule tâche auxiliaire à budget réduit.",
"Plan décidé avant E2/P : λ=0.3, graine42, maximum 3 époques, un seul run. L'encodeur est choisi sur E1/val. Chaque pas optimise L_ary + λ L_arq. Les résultats E1 terminés sont conservés ; la comparaison est exploratoire car budget et nombre de graines diffèrent. Le plan additionnel est enregistré sur Drive et verrouillé avant test. Ne pas réexécuter la configuration pour appliquer ce changement au kernel actif.", reduced_loader + '''
if CONFIG["include_e2"] and RUN_TRAINING:
    display(campaign.choose_e2_encoder())
    campaign.run_part("E2", CONFIG)
    display(pd.Series(json.loads((ROOT / "results/arq_audit.json").read_text(encoding="utf-8"))).to_frame("Audit arq"))
''')
lesson("E.4 · Référence E1 et λ=0.3", "Comparer l'auxiliaire à la référence mono-tâche.",
"Une tâche proche peut apporter un signal utile ; elle peut aussi introduire un vocabulaire, une distribution ou des gradients défavorables. Le coût E2 inclut les calculs auxiliaires. L'effet observé ne sera pas supposé positif.", '''
e_summary = dev_summary("E")
display(e_summary)
if not e_summary.empty:
    e_summary.set_index("variant")[["weighted_f1", "macro_f1"]].plot.bar(title="E1/E2 · validation", figsize=(11,4))
    plt.tight_layout(); plt.show()
''')
lesson("E.5 · Courbes d'apprentissage", "Repérer surapprentissage et arrêt anticipé.",
"Les courbes proviennent des historiques réels. L'époque retenue utilise le F1 val brut ; le réglage des décalages est une étape distincte. Une absence de courbe signifie absence de run.", '''
for family in "CDE":
    figure = plot_histories(family)
    if figure:
        display(Image(filename=str(figure)))
''')
lesson("P court · Sans emojis et sans ponctuation", "Tester uniquement la suppression conjointe.",
"Le plan court activé en E.3 conserve un TF-IDF union et un DarijaBERT, graine42, maximum 3 époques pour BERT. Les références conservatrices viennent d'A_union_s42 et E_darijabert_s42. Aucun autre nettoyage ni baseline de raccourcis n'est lancé. Résultats exploratoires : une seule graine et budget réduit.", reduced_loader + '''
if CONFIG["include_cleaning_study"] and RUN_TRAINING:
    campaign.run_part("P", CONFIG)
display(dev_summary("P"))
study = results_table()
if not study.empty:
    study = study.loc[study.family.eq("P") | (study.family.eq("E") & study.variant.eq("darijabert")) | (study.family.eq("A") & study.variant.eq("union"))]
    study.to_csv(ROOT / "results/cleaning_study.csv", index=False)
    display(study.loc[study.split.eq("dev")])
''')
lesson("Avant F · Contrôle de complétude", "Vérifier qu'aucune expérience obligatoire n'a été omise.",
"La campagne refuse l'évaluation finale si une partie manque. La reprise d'un notebook déjà verrouillé n'autorise aucun nouvel apprentissage.", '''
status = campaign.status(CONFIG)
display(status)
status.to_csv(ROOT / "results/execution_status.csv", index=False)
if not results_table().empty:
    display(select_on_dev())
''')
lesson("Frontière val/test · Verrouillage", "Figer les choix avant le test.",
"Toutes les variantes demandées sont déclarées avant test, avec leur règle val. Le tableau G conserve un représentant par famille choisi sur val. Les comparaisons complètes sont sauvegardées dans results.csv ; le meilleur score test ne remplace jamais le candidat verrouillé.", '''
if RUN_FINAL_EVALUATION:
    final = campaign.finalize(CONFIG)
    display(final)
else:
    print("F/G non exécutés : RUN_FINAL_EVALUATION=False")
''')
lesson("F1.1 · Distributions val/test et raccourcis", "Mesurer les changements globaux et conditionnels.",
"Une variation de proportions n'établit pas à elle seule un pur label shift. étudier aussi les écritures et les scores par classe. Les labels test ne sont lus qu'après verrouillage.", '''
after_test()
display(campaign.distributions(include_test=True))
display(campaign.shortcuts(include_test=True))
shift = show_file("f1_label_shift.csv")
shift.set_index(shift.columns[0]).plot.bar(title="F1 · proportions val/test")
plt.tight_layout(); plt.show()
''')
lesson("F1.2 · F1 en fonction des décalages", "Visualiser la sensibilité a posteriori.",
"Grille finie de deux décalages relatifs. Le maximum test est un oracle optimiste de diagnostic, jamais une nouvelle décision déployée. Les fichiers meta et le candidat restent intacts.", '''
after_test()
grid = campaign.threshold_diagnostic()
fig, axes = plt.subplots(1, 2, figsize=(12,4))
for ax, split in zip(axes, ["dev", "test"]):
    matrix = grid.pivot(index="neutral_offset", columns="positive_offset", values=f"{split}_weighted_f1")
    im = ax.imshow(matrix.to_numpy(), origin="lower", aspect="auto", extent=[-1.5,1.5,-1.5,1.5])
    ax.set(title=f"F1 pondéré · {split} · diagnostic", xlabel="Décalage positif", ylabel="Décalage neutre")
    fig.colorbar(im, ax=ax)
plt.tight_layout(); plt.savefig(ROOT / "results/figures/F1_threshold_grid.png", dpi=140); plt.show()
diagnostic = json.loads((ROOT / "results/f1_posthoc_oracle.json").read_text(encoding="utf-8"))
display(pd.Series(diagnostic).to_frame())
curve = grid.loc[np.isclose(grid.positive_offset, diagnostic["frozen_offsets"][2])]
curve.plot(x="neutral_offset", y=["dev_weighted_f1", "test_weighted_f1"], title="F1 = f(décalage neutre), décalage positif figé")
plt.axvline(diagnostic["frozen_offsets"][1], color="black", linestyle="--", label="Décision choisie sur val")
plt.ylabel("Weighted-F1"); plt.legend(); plt.tight_layout()
plt.savefig(ROOT / "results/figures/F1_threshold_curve.png", dpi=140); plt.show()
''')
lesson("F2 · écriture non vue", "Mesurer arabe→latin et latin→arabe.",
"TF-IDF union est entraîné sur un seul sous-ensemble train, une seule fois car déterministe ; les décalages utilisent val. Comparer à l'union train complet sur la même écriture test. Taille du train et couverture d'écriture changent ensemble : l'écart n'isole pas causalement l'alphabet.", '''
after_test()
cross = show_file("f2_cross_script.csv")
if not cross.empty:
    display(cross.groupby(["train_script", "test_script"]).weighted_f1.agg(["mean", "std"]))
    full = pd.read_csv(ROOT / "results/A_union_s42_scripts.csv")
    display(full.assign(reference="union train complet"))
''')
lesson("F3 · Qualité et couverture par écriture", "Comparer arabe, latin et mixte.",
"Le faible effectif mixte limite les conclusions. La translittération est optionnelle et n'est pas ajoutée sans système validé. Les embeddings ne sont pas ajustés sur test pour produire ce rapport.", '''
after_test()
display(campaign.embedding_oov(CONFIG, include_test=True))
lock = json.loads((ROOT / "results/selection_lock.json").read_text(encoding="utf-8"))
for winner in lock["selected_variants"]:
    show_file(f"{winner['family']}_{winner['variant']}_s42_scripts.csv")
''')
lesson("F4 · Emojis et ponctuation séparés", "Mesurer les bascules sur une population commune.",
"Chaque modèle est évalué sur les mêmes tweets, incluant neutres et erreurs initiales. Trois perturbations : emojis/émoticônes seuls, ponctuation seule, les deux. Les différences de F1 avec labels originaux sont diagnostiques : une transformation peut changer le sentiment. Les véritables scores des textes transformés demandent une annotation humaine.", '''
after_test()
comparison = pd.read_csv(ROOT / "results/comparison.csv")
robustness = pd.DataFrame([{**json.loads(r.robustness_f4), "family": r.family} for r in comparison.dropna(subset=["robustness_f4"]).itertuples()])
display(robustness)
if not robustness.empty:
    robustness.set_index("family").flip_rate_changed.plot.bar(title="F4 · bascules de prédiction, textes modifiés")
    plt.tight_layout(); plt.show()
show_file("f4_perturbations.csv", 8)
lock = json.loads((ROOT / "results/selection_lock.json").read_text(encoding="utf-8"))
candidate_run = f"{lock['candidate']['family']}_{lock['candidate']['variant']}_s42"
show_file(f"f4_{candidate_run}_summary.csv")
''')
lesson("F4 humain · Contrôler le maintien du sentiment", "Relire les transformations avant de conclure à une erreur.",
"Compléter f4_human_review.csv : human_label, sentiment_preserved, reviewer et validation_humaine='validé'. Les colonnes humaines sont préservées. Aucun score humain n'est annoncé tant que ces champs ne sont pas remplis.", '''
after_test()
from src.validation import robustness_review, evaluate_human_sets
lock = json.loads((ROOT / "results/selection_lock.json").read_text(encoding="utf-8"))
candidate_run = f"{lock['candidate']['family']}_{lock['candidate']['variant']}_s42"
candidate_folder = ROOT / "models" / candidate_run
candidate_meta = json.loads((candidate_folder / "meta.json").read_text(encoding="utf-8"))
candidate_predictor = load_predictor(candidate_folder, CONFIG["device"])
display(robustness_review(load_split("test", labels=True), candidate_predictor, candidate_meta["offsets"]))
print("À annoter :", ROOT / "results/f4_human_review.csv")
''')
lesson("F7 · Tests linguistiques et nouvel ensemble indépendant", "Évaluer des exemples réellement validés par des humains.",
"Les 12 exemples construits dans human_review/behavioral_cases.csv sont des propositions à annoter, pas des données de test validées. independent.csv est vide jusqu'à votre collecte de nouveaux textes et leur annotation avec source/date/relecteur. Les suggestions ne sont jamais des labels. Les fichiers restent sur Drive ; les quasi-doublons du benchmark bloquent le score indépendant. Ne pas régler le modèle sur ces évaluations.", '''
after_test()
display(evaluate_human_sets(candidate_predictor, candidate_meta["offsets"]))
show_file(f"{candidate_run}_subgroups.csv", 30)
print("Fichiers humains persistants :", ROOT / "human_review")
del candidate_predictor
''')
lesson("F5 · Graines, IC et McNemar", "Décrire la variabilité d'entraînement et d'échantillonnage.",
"IC percentile 95 % par 1 000 bootstrap de tweets, conditionnel au checkpoint 42. L'écart-type sur trois graines mesure autre chose. McNemar exact compare les erreurs appariées des deux meilleures variantes choisies sur val ; ce n'est pas un test direct de différence de F1.", '''
after_test()
comparison = pd.read_csv(ROOT / "results/comparison.csv")
display(campaign.uncertainty_table())
display(comparison[["family", "variant", "weighted_f1", "ci_low", "ci_high", "test_seed_mean", "test_seed_std", "n_seeds", "bootstrap"]])
display(pd.Series(json.loads((ROOT / "results/f5_mcnemar.json").read_text(encoding="utf-8"))).to_frame("McNemar"))
plt.errorbar(comparison.family, comparison.weighted_f1, yerr=[(comparison.weighted_f1-comparison.ci_low).clip(lower=0), (comparison.ci_high-comparison.weighted_f1).clip(lower=0)], fmt="o", capsize=4)
plt.ylabel("Weighted-F1 test et IC bootstrap 95 %"); plt.tight_layout(); plt.show()
''')
lesson("F6.1 · Dossier de trente erreurs", "Lire et classer les erreurs du candidat fixé sur val.",
"Catégories suggérées : ironie, graphies/tokenisation Arabizi, mélange français, neutre/polarisé, annotation. Les heuristiques sont des hypothèses ; chaque ligne nécessite une lecture humaine. Le désaccord de votes n'est pas une preuve de bruit.", '''
after_test()
errors = show_file("f6_30_erreurs.csv", 30)
display(campaign.error_review_summary())
print("Fichier à compléter :", ROOT / "results/f6_30_erreurs.csv")
''')
lesson("F6.2 · Valider la revue", "Séparer propositions automatiques et conclusions humaines.",
"Compléter categorie_humaine, validation_humaine='validé' et commentaire_humain dans le CSV. Relancer cette cellule pour lire votre revue. L'échantillonnage diversifié ne mesure pas la fréquence des catégories dans toutes les erreurs ; ne pas extrapoler sa part de bruit.", '''
after_test()
review = pd.read_csv(ROOT / "results/f6_30_erreurs.csv", keep_default_na=False)
display(campaign.error_review_summary())
if "categorie_humaine" in review:
    validated = review.loc[review.validation_humaine.eq("validé")]
    display(validated.categorie_humaine.value_counts().to_frame("erreurs validées de cet échantillon"))
''')
lesson("G.1 · Tableau final et matrice de confusion", "Choisir en tenant compte de qualité, coût et limites.",
"Le candidat a déjà été fixé sur val. Les colonnes montrent temps d'apprentissage, extraction partagée éventuelle, latence CPU après échauffement, taille des poids, paramètres, robustesse et écritures. Aucun changement de candidat après observation du test.", '''
after_test()
final = show_file("comparison.csv")
all_variants = show_file("comparison_all_variants.csv", 100)
display(Image(filename=str(ROOT / "results/figures/modele_final.png")))
print("Candidat fixé sur val :", lock["candidate"])
''')
lesson("G.2 · Compromis qualité/coût", "Présenter les résultats de manière visuelle.",
"Les temps et latences dépendent du matériel. La mesure CPU utilise 30 inférences unitaires, après chargement et échauffement. La taille inclut les ressources nécessaires à l'inférence ; elle n'est pas une mesure de RAM de pic.", '''
after_test()
fig, axes = plt.subplots(1, 3, figsize=(15,4))
for ax, x, title in zip(axes, ["cpu_ms_per_tweet", "weight_mb", "train_seconds"], ["Latence CPU (ms/tweet)", "Poids (Mio)", "Apprentissage (s)"]):
    positive = final.loc[final[x].gt(0)]
    ax.scatter(positive[x], positive.weighted_f1)
    for r in positive.itertuples():
        ax.annotate(r.family, (getattr(r,x), r.weighted_f1))
    ax.set(xlabel=title, ylabel="Weighted-F1 test", xscale="log")
plt.tight_layout(); plt.savefig(ROOT / "results/figures/G_quality_cost.png", dpi=140); plt.show()
''')
lesson("G.3 · Préparer l'oral Q1–Q12", "Relier chaque réponse à une mesure réelle.",
"Présenter d'abord protocole et corpus, puis baselines, représentations, adaptation, robustesse et choix pratique. Pour chaque question : observation → mesure → interprétation → limite. Les sources et les questions officielles sont disponibles dans research/.", '''
questions = PROJECT / "research/questions_officielles.md"
if questions.exists():
    display(Markdown(questions.read_text(encoding="utf-8")))
checklist = {"Q1–Q2": "A + distributions + shortcut_baselines.csv", "Q3": "nettoyage conservateur + F4 ; aucune ablation individuelle du français/Arabizi n'est revendiquée", "Q4–Q5": "B, résultats val/test et scripts", "Q6–Q7": "C, déséquilibre sur meilleur C et ablation des positions", "Q8": "D vs E1, temps/paramètres et écritures", "Q9": "E2, lambda=0 (E1) vs 0.1/0.3/0.7", "Q10": "F1, décision val et oracle grille test séparé", "Q11": "F2/F3 et compromis des deux écritures", "Q12": "F6 validé humainement et tableau G"}
display(pd.Series(checklist).to_frame("Preuves à commenter"))
print("Un score ne devient un résultat qu'après exécution. Ne pas présenter de gain non mesuré.")
''')

md("""
## Conclusion à rédiger après exécution

1. Quel modèle gagne sur val, puis que montre son test ? Indiquer moyenne, graines et IC.
2. Quel compromis observe-t-on entre arabe, latin et mixte ?
3. E1 dépasse-t-il D ? E2 améliore-t-il E1 et pour quel λ choisi sur val ?
4. Quelle part des 30 erreurs a été validée humainement ? Quelles limites persistent ?
5. Quel modèle convient au domaine de déploiement envisagé, avec quels besoins de suivi ?

**Extensions facultatives non réalisées ici :** MARBERTv2, translittération, cibles souples à partir des votes.
La suppression isolée du français, des chiffres Arabizi et des répétitions n'est pas évaluée par F4 ;
la réponse complète à Q3 exigerait ces ablations distinctes. Ne pas inventer leurs effets.

**Provenance des anciens scores :** notebook `_historique` et anciens `results/` ; aucune sortie historique
n'est une mesure de cette campagne. L'application et le PPTX historiques ne sont pas mis à jour automatiquement.
""")
notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Darija-Sentiment (Conda)", "language": "python", "name": "darija-sentiment"}, "language_info": {"name": "python", "version": "3.11"}})
path = Path(os.environ.get("DARIJA_NOTEBOOK_OUTPUT", str(ROOT / "notebooks/TP2_Darija.ipynb")))
if path.exists():
    previous = nbf.read(path, as_version=4)
    if any(c.get("outputs") for c in previous.cells):
        archive_dir = path.parent / "archive"
        archive_dir.mkdir(exist_ok=True)
        shutil.copy2(path, archive_dir / f"TP2_Darija_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.ipynb")
nbf.write(notebook, path)
print(f"Notebook préparé sans exécution : {path} ({len(cells)} cellules, {sum(c.cell_type == 'code' for c in cells)} code)")
