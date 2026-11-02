"""Architectures A–E. Sorties dans l'ordre negative, neutral, positive."""
import math
import hashlib
import json
import time
import tempfile
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from .preprocess import LABELS, ROOT, tokens

MODEL_REVISIONS = {
    "SI2M-Lab/DarijaBERT": "4025e5465fc6b3c9aa516839dd0bd039fa2e6d99",
    "SI2M-Lab/DarijaBERT-arabizi": "399963e234e368ed8428f32bf040e4a6e03c819c",
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7": "b5113eb38ab63efdd7f280f8c144ea8b13f978ce",
    "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment": "b0df630d23e5aa5f8a326017605337f6d0863ecd",
}
NLI_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
CAMEL_ID = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
BERT_ID = "SI2M-Lab/DarijaBERT"
HYPOTHESES = {
    "ar": ["هذا النص يعبر عن شعور سلبي.", "هذا النص محايد عاطفيا.", "هذا النص يعبر عن شعور إيجابي."],
    "fr": ["Ce texte exprime un sentiment négatif.", "Ce texte exprime un sentiment neutre.", "Ce texte exprime un sentiment positif."],
    "en": ["This text expresses a negative sentiment.", "This text expresses a neutral sentiment.", "This text expresses a positive sentiment."],
}


def classical_model(kind, seed=42, C=1.0, preserve_symbols=False):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000,
                          tokenizer=tokens if preserve_symbols else None,
                          token_pattern=None if preserve_symbols else r"(?u)\b\w\w+\b")
    char = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=100000)
    if kind == "majority":
        return Pipeline([("features", word), ("classifier", DummyClassifier(strategy="most_frequent"))])
    features = {"word": word, "char": char, "union": FeatureUnion([("word", word), ("char", char)])}[kind]
    return Pipeline([("features", features), ("classifier", LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed))])


class Attention(nn.Module):
    """Score appris par token ; padding masqué avant le softmax."""
    def __init__(self, dimension):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dimension, 64), nn.Tanh(), nn.Linear(64, 1, bias=False))

    def forward(self, states, mask):
        scores = self.score(states).squeeze(-1).masked_fill(~mask, -1e4)
        weights = scores.softmax(dim=1)
        return (states * weights.unsqueeze(-1)).sum(dim=1), weights


class TextNetwork(nn.Module):
    def __init__(self, vocab_size, kind="bilstm", embeddings=None, positional=True):
        super().__init__()
        self.kind, self.positional = kind, positional
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        if embeddings is not None:
            self.embedding.weight.data.copy_(torch.tensor(embeddings))
        if kind in {"bilstm", "attention"}:
            self.encoder = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
            if kind == "attention":
                self.attention = Attention(128)
        elif kind == "transformer":
            layer = nn.TransformerEncoderLayer(128, 4, dim_feedforward=256, dropout=0.1, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            positions = torch.arange(512).unsqueeze(1)
            frequencies = torch.exp(torch.arange(0, 128, 2) * (-math.log(10000.0) / 128))
            pe = torch.zeros(512, 128)
            pe[:, 0::2], pe[:, 1::2] = torch.sin(positions * frequencies), torch.cos(positions * frequencies)
            self.register_buffer("positions", pe)
        elif kind == "charcnn":
            self.convolutions = nn.ModuleList([nn.Conv1d(128, 64, k, padding=k // 2) for k in (3, 5, 7)])
        else:
            raise ValueError(kind)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(192 if kind == "charcnn" else 128, 3))

    def forward(self, ids, oov_vectors=None, return_attention=False):
        mask = ids.ne(0)
        embedded = self.embedding(ids)
        if oov_vectors is not None:
            embedded = torch.where(ids.eq(1).unsqueeze(-1), oov_vectors, embedded)
        attention = None
        if self.kind in {"bilstm", "attention"}:
            # Le packing évite que la direction arrière lise le padding.
            packed = nn.utils.rnn.pack_padded_sequence(embedded, mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
            states, _ = self.encoder(packed)
            states, _ = nn.utils.rnn.pad_packed_sequence(states, batch_first=True, total_length=ids.shape[1])
            if self.kind == "attention":
                pooled, attention = self.attention(states, mask)
            else:
                pooled = (states * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        elif self.kind == "transformer":
            if self.positional:
                embedded = embedded + self.positions[:ids.shape[1]]
            states = self.encoder(embedded, src_key_padding_mask=~mask)
            pooled = (states * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        else:
            states = embedded.transpose(1, 2)
            pooled = torch.cat([conv(states).relu().masked_fill(~mask.unsqueeze(1), -1e4).amax(2) for conv in self.convolutions], dim=1)
        logits = self.classifier(pooled)
        return (logits, attention) if return_attention else logits


def embedding_matrix(vocab, texts, method="random", seed=42, aravec_path=None, cache_dir=None):
    """AraVec est projeté à 128 dimensions avec une matrice aléatoire fixée (sans dev/test)."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0, 0.05, (len(vocab), 128)).astype("float32")
    covered = 0
    fasttext_path = None
    embedding_cached, embedding_seconds = False, 0.0
    if method != "random":
        from gensim.models import FastText, Word2Vec, KeyedVectors
        if method == "fasttext":
            # workers=1 pour la reproductibilité ; corpus strictement train.
            corpus = [tokens(t) for t in texts]
            digest = hashlib.sha256(repr(corpus).encode()).hexdigest()[:16]
            if cache_dir is not None:
                fasttext_path = Path(cache_dir) / f"fasttext_s{seed}_{digest}.model"
                fasttext_path.parent.mkdir(parents=True, exist_ok=True)
            if fasttext_path is not None and fasttext_path.exists():
                embedding_cached = True
                model = FastText.load(str(fasttext_path))
                timing = Path(str(fasttext_path) + ".timing.json")
                if timing.exists():
                    embedding_seconds = json.loads(timing.read_text(encoding="utf-8"))["seconds"]
            else:
                started = time.perf_counter()
                model = FastText(sentences=corpus, vector_size=128, window=5, min_count=2, sg=1, epochs=10, seed=seed, workers=1, bucket=20000)
                embedding_seconds = time.perf_counter() - started
                if fasttext_path is not None:
                    from .preprocess import save_json
                    with tempfile.TemporaryDirectory(prefix="fasttext_build_", dir=fasttext_path.parent) as temporary:
                        staged = Path(temporary) / fasttext_path.name
                        model.save(str(staged))
                        save_json(Path(str(fasttext_path) + ".timing.json"), {"seconds": embedding_seconds})
                        for companion in Path(temporary).iterdir():
                            if companion != staged:
                                companion.replace(fasttext_path.parent / companion.name)
                        staged.replace(fasttext_path)  # Publier le modèle seulement une fois ses tableaux prêts.
            vectors = model.wv
        elif method == "aravec":
            if not aravec_path or not Path(aravec_path).exists():
                raise FileNotFoundError("AraVec absent : renseigner le chemin officiel .mdl dans la configuration.")
            try:
                vectors = Word2Vec.load(str(aravec_path), mmap="r").wv
            except (AttributeError, TypeError):
                vectors = KeyedVectors.load(str(aravec_path))
        else:
            raise ValueError(method)
        projection = rng.normal(0, 1 / math.sqrt(vectors.vector_size), (vectors.vector_size, 128)).astype("float32")
        for word, idx in vocab.items():
            if idx < 2:
                continue
            if method == "fasttext" or word in vectors:
                vector = vectors[word]
                matrix[idx] = vector if len(vector) == 128 else vector @ projection
                covered += 1
    matrix[0] = 0
    return matrix, {"method": method, "covered": covered, "vocabulary": len(vocab) - 2, "dimension": 128, "aravec_path": str(aravec_path), "fasttext_path": str(fasttext_path) if fasttext_path else None,
                    "embedding_cache_used": embedding_cached, "embedding_training_seconds": embedding_seconds}


@lru_cache(maxsize=1)
def fasttext_vectors(path):
    from gensim.models import FastText
    return FastText.load(str(path)).wv


def word_inputs(texts, vocab, max_length, fasttext_path=None):
    """UNK peut recevoir un vecteur composé par fastText, sans élargir le vocabulaire appris."""
    from .preprocess import encode_texts
    texts = list(texts)
    ids = torch.tensor(encode_texts(texts, vocab, max_length))
    if fasttext_path is None:
        return (ids,)
    vectors = fasttext_vectors(str(fasttext_path))
    fallback = np.zeros((len(texts), max_length, 128), dtype="float32")
    for row, text in enumerate(texts):
        for col, word in enumerate(tokens(text)[:max_length]):
            if word not in vocab:
                fallback[row, col] = vectors[word]
    return ids, torch.from_numpy(fallback)


class FrozenSequence(nn.Module):
    def __init__(self, dimension=768):
        super().__init__()
        self.encoder = nn.LSTM(dimension, 64, batch_first=True, bidirectional=True)
        self.attention = Attention(128)
        self.classifier = nn.Linear(128, 3)

    def forward(self, states, mask, return_attention=False):
        states = states.to(self.encoder.weight_ih_l0.dtype)
        packed = nn.utils.rnn.pack_padded_sequence(states, mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
        sequence, _ = self.encoder(packed)
        sequence, _ = nn.utils.rnn.pad_packed_sequence(sequence, batch_first=True, total_length=mask.shape[1])
        pooled, weights = self.attention(sequence, mask.bool())
        logits = self.classifier(pooled)
        return (logits, weights) if return_attention else logits


class FineTunedSentiment(nn.Module):
    """Encodeur ajusté sur ary et une seule tête sentiment marocain."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.config.hidden_size, 3)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, **unused):
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.classifier(self.dropout(states[:, 0]))


class MultiTaskSentiment(FineTunedSentiment):
    """Même encodeur, deux têtes ; l'inférence principale appelle la tête ary."""
    def __init__(self, encoder):
        super().__init__(encoder)
        self.auxiliary_classifier = nn.Linear(encoder.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask, task="ary"):
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        head = self.classifier if task == "ary" else self.auxiliary_classifier
        return head(self.dropout(states[:, 0]))


class FrozenEncoder:
    def __init__(self, model_id=BERT_ID, device="cpu", local_files_only=False, revision=None):
        from transformers import AutoModel, AutoTokenizer
        self.model_id, self.device = model_id, device
        revision = revision or MODEL_REVISIONS[model_id]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_id, revision=revision, local_files_only=local_files_only, add_pooling_layer=False).to(device).eval()
        self.model.requires_grad_(False)

    def extract(self, texts, pooling="mean", batch_size=8, max_length=64, output_path=None):
        outputs, masks = [], []
        mapped = None
        if output_path is not None:
            if pooling != "states":
                raise ValueError("Le cache mémoire mappée concerne les états de tokens")
            mapped = np.lib.format.open_memmap(output_path, mode="w+", dtype="float16", shape=(len(texts), max_length, self.model.config.hidden_size))
        try:
            for start in range(0, len(texts), batch_size):
                inputs = self.tokenizer(list(texts[start:start + batch_size]), padding="max_length", truncation=True, max_length=max_length, return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    states = self.model(**inputs).last_hidden_state
                    mask = inputs.attention_mask
                    if pooling == "cls":
                        output = states[:, 0]
                    elif pooling == "mean":
                        # Moyenne incluant les tokens spéciaux, excluant PAD ; convention documentée.
                        output = (states * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
                    elif pooling == "states":
                        output = states
                    elif pooling == "all":
                        output = {"cls": states[:, 0].cpu().numpy().copy(), "mean": ((states * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)).cpu().numpy(), "states": states.cpu().numpy().astype("float16")}
                    else:
                        raise ValueError(pooling)
                if mapped is not None:
                    mapped[start:start + len(inputs.input_ids)] = output.cpu().numpy().astype("float16")
                else:
                    outputs.append(output if pooling == "all" else output.cpu().numpy().astype("float16" if pooling == "states" else "float32"))
                masks.append(mask.cpu().numpy())
        except BaseException:
            if mapped is not None:
                mapped._mmap.close()
            raise
        if mapped is not None:
            mapped.flush()
            return mapped, np.concatenate(masks)
        if pooling == "all":
            return {key: np.concatenate([o[key] for o in outputs]) for key in outputs[0]}, np.concatenate(masks)
        return np.concatenate(outputs), np.concatenate(masks)


class PretrainedSentiment:
    def __init__(self, kind="nli", language="fr", device="cpu", local_files_only=False, revision=None):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.kind, self.language, self.device = kind, language, device
        self.model_id = NLI_ID if kind == "nli" else CAMEL_ID
        revision = revision or MODEL_REVISIONS[self.model_id]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=revision, local_files_only=local_files_only)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, revision=revision, local_files_only=local_files_only).to(device).eval()

    def predict_proba(self, texts, batch_size=8):
        results = []
        mapping = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        if self.kind == "nli":
            entailment = next(i for i, v in mapping.items() if "entail" in v)
        else:
            if set(mapping.values()) != set(LABELS):
                raise ValueError(f"Mapping CAMeLBERT à vérifier : {mapping}")
            order = [next(i for i, v in mapping.items() if v == label) for label in LABELS]
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start:start + batch_size])
            if self.kind == "nli":
                premises = [t for t in batch for _ in LABELS]
                hypotheses = HYPOTHESES[self.language] * len(batch)
                inputs = self.tokenizer(premises, hypotheses, padding=True, truncation="only_first", max_length=128, return_tensors="pt")
            else:
                inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            with torch.inference_mode():
                logits = self.model(**inputs.to(self.device)).logits
                scores = logits[:, entailment].reshape(-1, 3).softmax(-1) if self.kind == "nli" else logits.softmax(-1)[:, order]
            results.append(scores.cpu().numpy())
        return np.concatenate(results)
