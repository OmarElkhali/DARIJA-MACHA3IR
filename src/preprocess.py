"""Données officielles, nettoyage conservateur et audit sans labels test."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import re
import unicodedata
import urllib.request

import numpy as np
import pandas as pd
from .resources import file_sha256, download_atomic

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("DARIJA_OUTPUT_ROOT", str(PROJECT_ROOT))).resolve()
CLEANING = os.environ.get("DARIJA_CLEANING", "historical")
LABELS = ["negative", "neutral", "positive"]
COMMIT = "5aec3cfcf87b59f885e98641f72e52182776429d"
BASE_URL = f"https://raw.githubusercontent.com/afrisenti-semeval/afrisent-semeval-2023/{COMMIT}/"
# Références SHA-256 calculées lors du téléchargement de cette révision officielle.
REFERENCE_SHA256 = {
    "data/ary/train.tsv": "9cf4e642bd8425e506192c161e22498e4867c7cc9814ea5bae7e2182b87c9c2b",
    "data/ary/dev.tsv": "03c4b5612080aeb7011b2d893504ccd6cd0d0aca329dd3ba34ba7defc170cd60",
    "data/ary/test.tsv": "1bef08f24090d001c7443c9a90781609901608361dd9a3f8a93dbf3fa68a2417",
    "data_with_annotators_labels/README.txt": "79c844c62b80848426fccbb820e29672bfd4319fbff5526ba1dcfd3759217aed",
    "data_with_annotators_labels/morrocan_arabic_individual_labels.csv": "600fab26b17d259bfc5ccff2c3abdafb26329529df5d3771968652f0c44ad381",
    "README.md": "26a5240c3fed0fe95a579764ffde01dd131df45350d0f006443a6634c0951732",
}
ARABIC = re.compile(r"[\u0620-\u064a\u066e-\u06d3\u06fa-\u06fc]")
LATIN = re.compile(r"[A-Za-zÀ-ÿ]")


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    return file_sha256(path)


def download_data():
    """Fige la révision et conserve les fichiers bruts à l'identique."""
    paths = [f"data/ary/{split}.tsv" for split in ("train", "dev", "test")]
    paths += ["data_with_annotators_labels/README.txt", "data_with_annotators_labels/morrocan_arabic_individual_labels.csv", "README.md"]
    records = []
    for remote in paths:
        local = PROJECT_ROOT / (remote if remote.startswith("data/") else "data/provenance/" + remote.replace("/", "_"))
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists():
            download_atomic(BASE_URL + remote, local, REFERENCE_SHA256[remote])
        if sha256(local) != REFERENCE_SHA256[remote]:
            raise ValueError(f"Intégrité incorrecte : {local}. Le fichier diffère de la révision officielle figée.")
        records.append({"path": str(local.relative_to(PROJECT_ROOT)), "url": BASE_URL + remote, "sha256": sha256(local)})
    save_json(PROJECT_ROOT / "data/manifest.json", {"commit": COMMIT, "files": records})
    return pd.DataFrame(records)


TEXT_MODES = ("conservative_v2", "without_emojis", "without_punctuation", "without_both")
EMOJI_SEQUENCE = re.compile(
    r"[#*0-9]\ufe0f?\u20e3|[\U0001F1E6-\U0001F1FF]{2}|"
    r"[\U0001F000-\U0001FAFF\u2600-\u27BF\u00a9\u00ae\u203c\u2049\u2122\u2139\u3030\u303d\u3297\u3299]"
    r"[\ufe0e\ufe0f]?[\U0001F3FB-\U0001F3FF]?(?:\u200d[\U0001F000-\U0001FAFF\u2600-\u27BF][\ufe0e\ufe0f]?[\U0001F3FB-\U0001F3FF]?)*"
    r"[\U000E0020-\U000E007F]*")
EMOTICON = re.compile(r"(?<!\w)(?:[:;=8xX][-^']?[)(/\\DPpOo]|<3)(?!\w)")


def emoji_parts(text):
    return EMOJI_SEQUENCE.findall(str(text)) + EMOTICON.findall(str(text))


def remove_emojis(text):
    return EMOTICON.sub(" ", EMOJI_SEQUENCE.sub(" ", str(text)))


def remove_punctuation(text):
    strip = lambda value: "".join(" " if unicodedata.category(c).startswith("P") else c for c in value)
    text = str(text)
    parts, start = [], 0
    for match in sorted([*EMOJI_SEQUENCE.finditer(text), *EMOTICON.finditer(text)], key=lambda m: m.start()):
        if match.start() >= start:
            parts += [strip(text[start:match.start()]), match.group()]
            start = match.end()
    return "".join([*parts, strip(text[start:])])


def clean_text(text, mode=None):
    """Conserve négation, chiffres Arabizi, accents, emojis et répétitions."""
    mode = CLEANING if mode is None else mode
    if mode not in {*TEXT_MODES, "historical", "emoji_only", "punctuation_only"}:
        raise ValueError(f"Nettoyage inconnu : {mode}")
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " URL " if mode == "historical" else " ", text)
    text = re.sub(r"@\w+", " USER " if mode == "historical" else " ", text)
    if mode == "emoji_only":
        text = " ".join(emoji_parts(text))
    elif mode == "punctuation_only":
        text = " ".join(c for c in remove_emojis(text) if unicodedata.category(c).startswith("P"))
    else:
        if mode in {"without_emojis", "without_both"}:
            text = remove_emojis(text)
        if mode in {"without_punctuation", "without_both"}:
            text = remove_punctuation(text)
    return re.sub(r"\s+", " ", text).strip()


def transform_frame(frame, mode):
    frame = frame.copy()
    frame["text"] = frame.get("tweet", frame.text).map(lambda text: clean_text(text, mode) or "<EMPTY>")
    return frame


def script_type(text):
    # ponytail: heuristique d'alphabet ; annoter les langues si l'on veut mesurer le français.
    # Une écriture latine ne prouve pas que le texte est en français ou en Arabizi.
    # Les mentions anonymisées @user et les URLs ne doivent pas créer du latin artificiel.
    text = re.sub(r"https?://\S+|www\.\S+|@\w+|\[\[PHOTO\]\]", "", str(text))
    arabic, latin = bool(ARABIC.search(text)), bool(LATIN.search(text))
    return "mixed" if arabic and latin else "arabic" if arabic else "latin" if latin else "other"


def load_split(split, lang="ary", labels=False):
    if split not in {"train", "dev", "test"} or lang != "ary":
        raise ValueError("Partition ou langue inconnue")
    if split == "test" and labels:
        lock = ROOT / "results/selection_lock.json"
        if not lock.exists():
            raise RuntimeError("Labels test réservés à F/G : verrouiller la sélection sur dev d'abord.")
    columns = ["tweet", "label"] if labels or split != "test" else ["tweet"]
    path = PROJECT_ROOT / f"data/{lang}/{split}.tsv"
    if sha256(path) != REFERENCE_SHA256[f"data/ary/{split}.tsv"]:
        raise ValueError(f"La partition officielle a été modifiée : {path}")
    frame = pd.read_csv(path, sep="\t", usecols=columns, keep_default_na=False)
    if "label" in frame and not frame.label.isin(LABELS).all():
        raise ValueError("Labels non conformes dans le fichier officiel")
    if frame.tweet.str.strip().eq("").any():
        raise ValueError("Tweet vide détecté")
    frame["row_id"] = [f"{lang}:{split}:{i}" for i in range(len(frame))]
    frame["script"] = frame.tweet.map(script_type)
    frame["text"] = frame.tweet.map(clean_text).replace("", "<EMPTY>")
    return frame


def audit_data():
    frames = {s: load_split(s) for s in ("train", "dev", "test")}
    counts, conflicts, collisions = [], [], []
    for split, frame in frames.items():
        counts.append({"split": split, "n": len(frame), "expected": {"train": 5583, "dev": 494, "test": 2961}[split],
                       "raw_duplicates": int(frame.tweet.duplicated().sum()), "clean_duplicates": int(frame.text.duplicated().sum())})
        if split != "test":
            bad = frame.groupby("text").label.nunique()
            conflicts += [{"split": split, "text": t, "labels": ",".join(sorted(frame.loc[frame.text == t, "label"].unique()))} for t in bad[bad > 1].index]
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        for text in sorted(set(frames[a].text) & set(frames[b].text)):
            collisions.append({"split_a": a, "split_b": b, "text": text})
    # Contradictions inter-partitions train/dev seulement, jamais de labels test ici.
    joint = pd.concat([frames["train"].assign(split="train"), frames["dev"].assign(split="dev")])
    bad = joint.groupby("text").label.nunique()
    conflicts += [{"split": "train+dev", "text": t, "labels": ",".join(sorted(joint.loc[joint.text == t, "label"].unique()))} for t in bad[bad > 1].index]
    out = ROOT / "results/audit"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(counts).to_csv(out / "counts.csv", index=False)
    pd.DataFrame(collisions, columns=["split_a", "split_b", "text"]).to_csv(out / "collisions.csv", index=False)
    pd.DataFrame(conflicts, columns=["split", "text", "labels"]).to_csv(out / "conflicts.csv", index=False)
    return pd.DataFrame(counts), pd.DataFrame(collisions), pd.DataFrame(conflicts)


def tokens(text, character=False):
    return list(clean_text(text)) if character else re.findall(r"\w+|[^\w\s]", clean_text(text).lower(), flags=re.UNICODE)


def build_vocab(texts, character=False, min_freq=2):
    counts = Counter(t for text in texts for t in tokens(text, character))
    return {"<PAD>": 0, "<UNK>": 1, **{t: i + 2 for i, t in enumerate(sorted(t for t, n in counts.items() if n >= min_freq))}}


def encode_texts(texts, vocab, max_length=64, character=False):
    # ponytail: vocabulaire fermé ; une composition fastText des OOV demanderait une autre expérience.
    values = np.zeros((len(texts), max_length), dtype=np.int64)
    for i, text in enumerate(texts):
        ids = [vocab.get(t, 1) for t in tokens(text, character)][:max_length] or [1]
        values[i, :len(ids)] = ids
    return values


def oov_table(frames, vocab):
    rows = []
    for split, frame in frames.items():
        for script, group in frame.groupby("script"):
            words = [w for t in group.text for w in tokens(t)]
            rows.append({"split": split, "script": script, "tokens": len(words), "oov_rate": sum(w not in vocab for w in words) / max(1, len(words))})
    return pd.DataFrame(rows)


def annotation_audit(frame):
    """Jointure conservative exacte ; les mentions masquées peuvent empêcher le match."""
    votes = pd.read_csv(PROJECT_ROOT / "data/provenance/data_with_annotators_labels_morrocan_arabic_individual_labels.csv")
    votes["text"] = votes["text"].map(clean_text)
    label_cols = ["label_1", "label_2", "label_3"]
    votes["vote_disagreement"] = votes[label_cols].nunique(axis=1) > 1
    votes = votes.drop_duplicates("text")
    return frame.merge(votes[["text", "text_id", *label_cols, "vote_disagreement"]], on="text", how="left", validate="many_to_one")
