"""Présentation interactive du TP : graphiques réels et inférence locale séquentielle."""
from collections import Counter
from pathlib import Path
import gc
import html
import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix

from src.preprocess import ROOT, LABELS, clean_text, load_split, script_type, tokens
from src.research import PAPERS
from src.ui_content import FAMILIES, GLOSSARY, variant_name

st.set_page_config(page_title="Darija / Sentiment", page_icon=":material/translate:", layout="wide", initial_sidebar_state="collapsed")
FOREST, SAGE, RED, GREY = "#173D36", "#91A58C", "#B66D60", "#A5AA9B"
PALETTE = {"positive": "#386A55", "neutral": GREY, "negative": RED}
SENTIMENT = {"negative": "Négatif", "neutral": "Neutre", "positive": "Positif"}
SCRIPTS = {"arabic": "Arabe", "latin": "Latine", "mixed": "Mixte", "other": "Autre"}
PAGES = ["Vue d’ensemble", "Laboratoire", "Corpus", "Modèles", "Robustesse", "Recherche"]
METRICS = {"weighted_f1": "Weighted-F1", "macro_f1": "Macro-F1", "precision": "Précision pondérée", "recall": "Rappel pondéré"}


def esc(value):
    return html.escape(str(value))


def number(value, digits=0):
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


@st.cache_data(show_spinner=False)
def cached_csv(path, modified):
    return pd.read_csv(path)


def table(name):
    path = ROOT / "results" / name
    return cached_csv(str(path), path.stat().st_mtime_ns) if path.exists() else pd.DataFrame()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}


@st.cache_data(show_spinner=False)
def cached_split(split, modified):
    return load_split(split)  # Le corpus test reste sans labels dans cette page.


def corpus(split):
    path = ROOT / f"data/ary/{split}.tsv"
    return cached_split(split, path.stat().st_mtime_ns)


def navigate(page):
    st.session_state["navigation"] = page


def open_phrase(text):
    st.session_state["pending_phrase"] = text
    navigate("Laboratoire")


def reset_corpus_filters():
    st.session_state.update(corpus_script="all", corpus_search="", corpus_sentiment="all")


def intro(title, subtitle, eyebrow=None):
    st.markdown(f'<div class="page-intro">{f"<div class=eyebrow>{esc(eyebrow)}</div>" if eyebrow else ""}<h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>', unsafe_allow_html=True)


def insight(title, text):
    st.markdown(f'<div class="insight"><strong>{esc(title)}</strong><br>{esc(text)}</div>', unsafe_allow_html=True)


def empty(title, text):
    st.markdown(f'<div class="empty-state"><strong>{esc(title)}</strong>{esc(text)}</div>', unsafe_allow_html=True)


def chart(figure, key, height=330):
    figure.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font=dict(family="Segoe UI, Arial, sans-serif", size=13, color=FOREST),
                         margin=dict(l=10, r=25, t=15, b=35), legend=dict(orientation="h", y=1.15, x=0),
                         colorway=[FOREST, SAGE, RED, GREY], hoverlabel=dict(bgcolor="#FCFBF7", font_size=13),
                         modebar=dict(bgcolor="rgba(0,0,0,0)", color="#68736C", activecolor=FOREST))
    figure.update_xaxes(gridcolor="#E2E2D8", zeroline=False, automargin=True)
    figure.update_yaxes(gridcolor="#E2E2D8", zeroline=False, automargin=True)
    st.plotly_chart(figure, theme=None, use_container_width=True, key=key,
                    config={"displaylogo": False, "scrollZoom": False, "responsive": True, "toImageButtonOptions": {"format": "png", "filename": f"darija_{key}", "scale": 2}})


def bar(values, names, key, colors=None, percent=True, height=330, error=None, full_scale=True):
    values = np.asarray(values, dtype=float)
    options = dict(type="data", array=error, visible=True, color="#68736C") if error is not None else None
    figure = go.Figure(go.Bar(x=values, y=list(names), orientation="h", marker_color=colors or FOREST,
                              text=[f"{v:.1%}" if percent else number(v, 1) for v in values], textposition="outside", cliponaxis=False,
                              error_x=options, hovertemplate="%{y}<br>%{x:.4f}<extra></extra>" if percent else "%{y}<br>%{x:.2f}<extra></extra>"))
    figure.update_yaxes(autorange="reversed", showgrid=False)
    upper = 1.04 if percent and full_scale else max(values.max(initial=0) * 1.22, .01 if percent else 1)
    figure.update_xaxes(range=[0, upper], tickformat=".0%" if percent else None)
    chart(figure, key, height)


def download(frame, label, filename, key):
    st.download_button(label, frame.to_csv(index=False).encode("utf-8-sig"), filename, "text/csv", key=key)


def model_catalog():
    from huggingface_hub import try_to_load_from_cache
    records = []
    external = {}
    for file in sorted((ROOT / "models").glob("*/meta.json")):
        try:
            meta = read_json(file)
            kind, config = meta["kind"], meta["config"]
            required = {
                "classical": ["classifier.joblib"],
                "neural": ["weights.pt", "vocab.json"],
                "pretrained": ["pretrained.json"],
                "frozen": ["classifier.joblib" if config.get("pooling") != "states" else "weights.pt"],
                "finetuned": ["weights.pt", "encoder_config/config.json", "tokenizer/tokenizer_config.json"],
            }[kind]
            missing = [name for name in required if not (file.parent / name).is_file()]
            if kind in {"pretrained", "frozen"}:
                identity = (config["model_id"], config["revision"])
                if identity not in external:
                    def cached(name):
                        path = try_to_load_from_cache(identity[0], name, revision=identity[1])
                        return isinstance(path, str) and Path(path).is_file()
                    external[identity] = (cached("config.json") and cached("tokenizer_config.json")
                                          and any(cached(name) for name in ("tokenizer.json", "vocab.txt", "spm.model"))
                                          and any(cached(name) for name in ("model.safetensors", "pytorch_model.bin")))
                if not external[identity]:
                    missing.append("encodeur ou tokenizer absent du cache local")
            records.append({**meta, "available": not missing, "missing": missing,
                            "display": f"{meta['family']} · {variant_name(meta['variant'])} · graine {meta['seed']}"})
        except (OSError, ValueError, KeyError, TypeError) as error:
            st.warning(f"Checkpoint ignoré ({file.parent.name}) : métadonnées illisibles ou incomplètes — {error}")
    return records


def selected_predictions(family="A"):
    row = COMPARISON.loc[COMPARISON.family.eq(family) & COMPARISON.run_id.notna()]
    return table(f"{row.iloc[0].run_id}_test_tuned.csv") if len(row) else pd.DataFrame()


def show_confusion(family="A", normalize=False, key="confusion"):
    predictions = selected_predictions(family)
    if predictions.empty:
        empty("Pas de matrice disponible", "Cette famille n'a pas encore de prédictions test enregistrées.")
        return
    matrix = confusion_matrix(predictions.label, predictions.prediction, labels=LABELS)
    values = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1) if normalize else matrix
    figure = go.Figure(go.Heatmap(z=values, x=[SENTIMENT[x] for x in LABELS], y=[SENTIMENT[x] for x in LABELS],
                                colorscale=[[0, "#F0F2E9"], [1, FOREST]], showscale=False,
                                text=[[f"{x:.1%}" if normalize else str(x) for x in line] for line in values],
                                texttemplate="%{text}", hovertemplate="Réel : %{y}<br>Prédit : %{x}<br>Valeur : %{z}<extra></extra>"))
    figure.update_xaxes(title="Sentiment prédit", side="bottom")
    figure.update_yaxes(title="Label officiel", autorange="reversed")
    chart(figure, key, 310)


def overview():
    left, right = st.columns([3, 1], gap="large")
    with left:
        st.markdown('<div class="eyebrow">Analyse de sentiment · Darija marocaine</div><div class="hero-title">La darija,<br>au-delà des mots.</div><p class="hero-subtitle">Explorer les sentiments. Comprendre les modèles.<br>Lire les résultats, avec leur contexte.</p>', unsafe_allow_html=True)
        a, b, _ = st.columns([1.2, 1.2, 1])
        a.button("Tester une phrase →", type="primary", on_click=navigate, args=("Laboratoire",), use_container_width=True)
        b.button("Découvrir le projet", on_click=navigate, args=("Modèles",), use_container_width=True)
    with right:
        st.markdown('<div class="word-art" aria-label="Machaaïr, sentiments en arabe"><span lang="ar" dir="rtl">مشاعر</span></div>', unsafe_allow_html=True)
    n = sum(len(corpus(split)) for split in ("train", "dev", "test"))
    evaluated = COMPARISON.loc[COMPARISON.weighted_f1.notna(), "family"].nunique()
    stats = [(number(n), "tweets officiels"), (number(RUNS.run_id.nunique()), "runs enregistrés"), (str(len(FAMILIES)), f"familles · {evaluated} évaluées"), ("3", "écritures principales")]
    st.markdown('<div class="stat-ledger">' + "".join(f'<div class="stat"><div class="stat-value">{value}</div><div class="stat-label">{label}</div></div>' for value, label in stats) + '</div>', unsafe_allow_html=True)
    left, right = st.columns([1.75, 1], gap="large")
    with left:
        st.subheader("L’épreuve du test")
        a, b = st.columns(2)
        split = a.radio("Partition du comparatif", ["test", "dev"], horizontal=True, format_func=lambda x: "Test officiel" if x == "test" else "Validation dev", key="overview_split")
        metric = b.selectbox("Mesure", list(METRICS), format_func=METRICS.get, key="overview_metric")
        ids = COMPARISON.run_id.dropna().tolist()
        frame = RUNS.loc[RUNS.run_id.isin(ids) & RUNS.split.eq(split) & RUNS.decision.eq("dev_offsets")].sort_values("family")
        if len(frame):
            names = [f"{row.family} · {variant_name(row.variant)}" for row in frame.itertuples()]
            bar(frame[metric], names, "overview_scores", [FOREST if x == "A" else SAGE for x in frame.family], height=300)
        st.caption("Représentants graine 42. Le choix des variantes utilise la moyenne dev des graines. E : campagne complète non exécutée.")
    with right:
        st.subheader("Une démarche, étape par étape.")
        steps = [("Données", "Des partitions officielles, une provenance et un audit."), ("Apprentissage", "Les paramètres apprennent uniquement sur train."), ("Validation", "Dev choisit les variantes, époques et décisions."), ("Test", "Les choix sont verrouillés avant de mesurer.")]
        st.markdown('<div class="timeline">' + "".join(f'<div class="timeline-step"><span class="step-number">{i}</span><div><strong>{title}</strong><p>{text}</p></div></div>' for i, (title, text) in enumerate(steps, 1)) + '</div>', unsafe_allow_html=True)
        best = COMPARISON.loc[COMPARISON.family.eq("A")].iloc[0]
        insight("Le score a besoin de contexte.", f"Le candidat retenu passe de {best.dev_weighted_f1_mean:.3f} sur dev à {best.weighted_f1:.3f} sur test. Le transfert vers les textes latins est une limite importante.")
        st.button("Explorer la robustesse →", on_click=navigate, args=("Robustesse",))
    st.divider()
    detail, cost, export = st.tabs(["Lire les erreurs", "Qualité et coût", "Les résultats en détail"])
    with detail:
        a, b = st.columns([1.15, 1], gap="large")
        with a:
            st.subheader("La matrice, en un regard")
            family = st.selectbox("Famille à examiner", list("ABCD"), format_func=lambda x: f"{x} · {FAMILIES[x]['name']}", key="overview_confusion_family")
            normalized = st.checkbox("Afficher les proportions par classe", key="overview_normalized")
            show_confusion(family, normalized, "overview_confusion")
        with b:
            st.subheader("Comment la lire ?")
            st.write("La diagonale contient les bonnes prédictions. Une case hors diagonale indique une confusion entre le label officiel et le sentiment prédit.")
            st.write("Le neutre regroupe beaucoup de prédictions du candidat, notamment sur le latin. Une moyenne globale masque cette différence entre écritures.")
            st.button("Voir les 30 erreurs commentées", on_click=navigate, args=("Robustesse",))
    with cost:
        measured = COMPARISON.dropna(subset=["weighted_f1", "cpu_ms_per_tweet"])
        figure = go.Figure(go.Scatter(x=measured.cpu_ms_per_tweet, y=measured.weighted_f1, mode="markers+text", text=measured.family,
                          textposition="top center", marker=dict(size=18, color=[FOREST, "#B5BAA9", SAGE, "#6D8C78"], line=dict(width=1, color=FOREST)),
                          customdata=measured[["weight_mb", "trainable_parameters"]], hovertemplate="Famille %{text}<br>Latence : %{x:.2f} ms/tweet<br>Weighted-F1 : %{y:.4f}<br>Poids : %{customdata[0]:.1f} Mo<extra></extra>"))
        figure.update_xaxes(title="Latence CPU par tweet (échelle logarithmique)", type="log")
        figure.update_yaxes(title="Weighted-F1 test", tickformat=".0%", range=[0, 1])
        chart(figure, "quality_cost", 360)
        st.caption("Chargement exclu ; 30 prédictions unitaires après échauffement. La taille des fichiers de poids et les coûts partagés d'extraction sont documentés dans le comparatif.")
    with export:
        fields = ["family", "variant", "weighted_f1", "macro_f1", "precision", "recall", "train_seconds", "cpu_ms_per_tweet", "weight_mb", "trainable_parameters", "n_seeds"]
        st.dataframe(COMPARISON[fields], hide_index=True, use_container_width=True)
        download(COMPARISON, "Télécharger le comparatif CSV", "darija_comparatif.csv", "comparison_download")


def prediction_panel(row):
    sentiments = ["positive", "neutral", "negative"]
    rail = "".join(f'<span class="score-segment" style="width:{row[x]*100:.5f}%;background:{PALETTE[x]}"></span>' for x in sentiments)
    legend = "".join(f'<div>{SENTIMENT[x]}<strong>{number(row[x]*100, 1)} %</strong></div>' for x in sentiments)
    st.markdown(f'<div class="result-panel"><div class="result-model">{esc(row["name"])}</div><div class="result-word">{SENTIMENT[row["prediction"]]}</div><div>Scores bruts · non calibrés</div><div class="score-rail" aria-label="Scores des trois sentiments">{rail}</div><div class="score-legend">{legend}</div></div>', unsafe_allow_html=True)


def laboratory():
    intro("Une phrase. Plusieurs lectures.", "Écrivez en arabe, en Arabizi ou dans les deux. Comparez les modèles réellement disponibles.", "Le laboratoire")
    if "pending_phrase" in st.session_state:
        st.session_state["lab_text"] = st.session_state.pop("pending_phrase")
    st.session_state.setdefault("lab_text", "هاد الخدمة زوينة بزاف")
    available = {m["run_id"]: m for m in CATALOG if m["available"]}
    examples = [("Arabe", "هاد الخدمة زوينة بزاف"), ("Arabizi", "hadchi zwin bzaf"), ("Mixte", "هاد الفيلم زوين mais la fin ma3jbatnich")]
    left, right = st.columns([1.08, 1], gap="large")
    with left:
        buttons = st.columns(3)
        for column, (label, phrase) in zip(buttons, examples):
            column.button(label, on_click=lambda value=phrase: st.session_state.update(lab_text=value), use_container_width=True, key=f"example_{label}")
        advanced = st.toggle("Toutes les variantes et graines", key="advanced_models", help=f"Le mode simple présente les représentants des familles mesurées. Le mode avancé donne accès aux {len(available)} checkpoints complets locaux.")
        recommended = set(COMPARISON.run_id.dropna())
        options = list(available) if advanced else [key for key in available if key in recommended]
        defaults = ["A_char_s42"] if "A_char_s42" in options else options[:1]
        selector_key = "chosen_advanced" if advanced else "chosen_simple"
        if selector_key in st.session_state:
            st.session_state[selector_key] = [key for key in st.session_state[selector_key] if key in options]
        with st.container(border=True, key="prediction_form"):
            phrase = st.text_area("Votre phrase", key="lab_text", max_chars=10000, height=150, placeholder="Écrivez une phrase en darija…")
            selected = st.multiselect("Modèles à comparer", options, default=defaults, format_func=lambda key: available[key]["display"], key=selector_key)
            submit = st.button("Analyser la phrase", type="primary", use_container_width=True)
        st.caption("Analyse au clic, sans téléchargement. Chaque modèle est chargé puis libéré avant le suivant. Les exemples ne sont pas ajoutés au benchmark.")
        if selected:
            with st.expander("Limites de lecture des modèles sélectionnés"):
                limits = []
                for run_id in selected:
                    meta = available[run_id]
                    config = meta["config"]
                    unit = "caractères" if config.get("kind") == "charcnn" else "tokens regex" if meta["kind"] == "neural" else "sous-tokens, tokens spéciaux inclus"
                    limits.append({"Modèle": meta["display"], "Lecture maximale": f"{config['max_length']} {unit}" if "max_length" in config else "Texte complet"})
                st.dataframe(pd.DataFrame(limits), hide_index=True, use_container_width=True)
                st.caption("Au-delà de cette limite, la fin du texte est tronquée. Pour NLI, la limite inclut aussi l’hypothèse de sentiment.")
        unavailable = [meta for meta in CATALOG if not meta["available"]]
        if unavailable:
            with st.expander(f"{len(unavailable)} checkpoint(s) incomplet(s)"):
                for meta in unavailable:
                    st.write(f"{meta['display']} : {', '.join(meta['missing'])}")
        if not options:
            st.info("Aucun checkpoint complet n'est disponible localement. Consultez Modèles pour connaître les expériences exécutées.")
        if submit:
            if not phrase.strip():
                st.warning("Saisissez une phrase non vide.")
            elif len(phrase) > 10000:
                st.warning("La limite est de 10 000 caractères.")
            elif not selected:
                st.warning("Sélectionnez au moins un modèle.")
            else:
                from src.train import load_predictor
                from src.evaluate import decide
                output, failures = [], []
                progress = st.progress(0, text="Préparation de l’analyse…")
                for index, run_id in enumerate(selected):
                    meta = available[run_id]
                    progress.progress(index / len(selected), text=f"{index+1}/{len(selected)} · {variant_name(meta['variant'])}")
                    predictor = None
                    try:
                        started = time.perf_counter()
                        predictor = load_predictor(ROOT / meta["checkpoint"])
                        scores = predictor([clean_text(phrase)])
                        row = {"run_id": run_id, "name": variant_name(meta["variant"]), "family": meta["family"], "seed": meta["seed"],
                               "prediction": decide(scores, meta["offsets"])[0], "offsets": meta["offsets"], "seconds_with_load": time.perf_counter()-started}
                        row.update({label: float(scores[0, i]) for i, label in enumerate(LABELS)})
                        output.append(row)
                    except (OSError, RuntimeError, ValueError, KeyError) as error:
                        failures.append(f"{meta['display']} : {error}")
                    finally:
                        del predictor
                        gc.collect()
                progress.empty()
                result = {"text": phrase, "clean_text": clean_text(phrase), "requested_models": selected, "predictions": output, "failures": failures}
                st.session_state["last_analysis"] = result
                if output:
                    history = st.session_state.setdefault("prediction_history", [])
                    history.insert(0, {"phrase": phrase, "modèles": len(output), "lectures": " · ".join(f"{x['name']} : {SENTIMENT[x['prediction']]}" for x in output)})
                    del history[8:]
    result = st.session_state.get("last_analysis")
    with right:
        if result and (phrase != result["text"] or set(selected) != set(result.get("requested_models", []))):
            st.info("La phrase ou les modèles ont changé. Relancez l’analyse pour actualiser ces résultats.")
        if result and result["predictions"]:
            by_id = {row["run_id"]: row for row in result["predictions"]}
            active = st.selectbox("Résultat à examiner", list(by_id), format_func=lambda run_id: f"{by_id[run_id]['family']} · {by_id[run_id]['name']} · graine {by_id[run_id]['seed']}", key="result_model")
            row = by_id[active]
            prediction_panel({**row, "name": f"{row['name']} · graine {row['seed']}"})
            st.caption(f"Résultat pour : {result['text']}")
            with st.expander("Comprendre la décision"):
                st.write("Le modèle produit trois scores normalisés. La règle choisie sur dev ajoute un décalage à leur logarithme, puis sélectionne le plus grand résultat.")
                st.latex(r"\hat{y}=\operatorname{argmax}_{c}\,[\log(p_c)+b_c]")
                st.dataframe(pd.DataFrame({"Sentiment": [SENTIMENT[x] for x in LABELS], "Score brut": [row[x] for x in LABELS], "Décalage dev": row["offsets"], "Score de décision": np.log(np.clip([row[x] for x in LABELS], 1e-12, 1)) + row["offsets"]}), hide_index=True, use_container_width=True)
                st.caption("Ces scores ne sont pas des probabilités calibrées de vérité. Une phrase isolée ne mesure pas le F1 du modèle.")
        else:
            empty("Votre prochaine lecture commence ici.", "Choisissez un modèle puis analysez une phrase. Le sentiment et les trois scores apparaîtront dans cet espace.")
        if result:
            for failure in result["failures"]:
                st.error(failure)
    st.divider()
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.subheader("Ce que lit le prétraitement")
        text = phrase
        words = tokens(text)
        st.caption(f"Écriture {SCRIPTS[script_type(text)].lower()} · {len(words)} tokens regex · {len(text)} caractères")
        st.markdown('<div class="tokens" dir="auto">' + "".join(f'<span class="token">{esc(word)}</span>' for word in words[:60]) + '</div>', unsafe_allow_html=True)
        if len(words) > 60:
            st.caption("Les 60 premiers tokens sont affichés.")
        st.caption("Ce découpage explique le prétraitement commun ; les Transformers utilisent ensuite leurs propres sous-tokens.")
        with st.expander("Voir le nettoyage conservateur"):
            st.write(clean_text(text))
            st.write("URLs et mentions sont remplacées. Les chiffres Arabizi, accents, négations et emojis sont conservés.")
    with right:
        st.subheader("Comparer les lectures")
        if result and result["predictions"]:
            predictions = pd.DataFrame(result["predictions"])
            labels = [f"{row.family} · {row['name']}<br>Graine {row.seed}" for _, row in predictions.iterrows()]
            figure = go.Figure()
            for sentiment in ["positive", "neutral", "negative"]:
                figure.add_bar(name=SENTIMENT[sentiment], x=labels, y=predictions[sentiment], marker_color=PALETTE[sentiment], hovertemplate="%{x}<br>Score : %{y:.3f}<extra></extra>")
            figure.update_layout(barmode="group")
            figure.update_yaxes(tickformat=".0%", range=[0, 1], title="Score brut")
            chart(figure, "live_scores", 300)
            display = predictions[["run_id", "prediction", "negative", "neutral", "positive"]].rename(columns={"run_id": "Modèle", "prediction": "Sentiment"})
            display["Sentiment"] = display.Sentiment.map(SENTIMENT)
            display.insert(0, "Phrase", result["text"])
            st.dataframe(display, hide_index=True, use_container_width=True)
            download(display, "Exporter cette analyse", "darija_prediction.csv", "prediction_download")
            if len(predictions) > 1:
                votes = predictions.prediction.value_counts()
                leaders = votes.loc[votes.eq(votes.max())]
                summary = "Égalité : " + ", ".join(f"{SENTIMENT[label]} ({int(count)})" for label, count in leaders.items()) if len(leaders) > 1 else f"{int(votes.iloc[0])} modèle(s) sur {len(predictions)} prédisent « {SENTIMENT[votes.index[0]]} »"
                insight("L’accord entre modèles", summary + ". Un accord entre modèles ne prouve pas que le sentiment est correct.")
        else:
            empty("Une comparaison, pas une intuition.", "Ajoutez plusieurs modèles pour voir comment leurs lectures d’une même phrase se rapprochent ou divergent.")
    if st.session_state.get("prediction_history"):
        with st.expander("Mes dernières analyses dans cette session"):
            history = st.session_state["prediction_history"]
            st.dataframe(pd.DataFrame(history), hide_index=True, use_container_width=True)
            previous = st.selectbox("Phrase à réutiliser", range(len(history)), format_func=lambda index: history[index]["phrase"][:90], key="history_phrase")
            st.button("Reprendre cette phrase", on_click=open_phrase, args=(history[previous]["phrase"],), key="replay_phrase")
            st.caption("Historique limité à huit analyses, conservé dans la session du navigateur et non enregistré dans le benchmark.")


def corpus_page():
    intro("Le corpus, avant le modèle.", "Explorer les tweets officiels, leurs écritures et leurs distributions. Le benchmark reste intact.", "AfriSenti · ary uniquement")
    a, b, c = st.columns([1, 1, 2])
    split = a.selectbox("Partition", ["train", "dev", "test"], format_func=lambda x: {"train": "Train · apprentissage", "dev": "Dev · validation", "test": "Test · textes seulement"}[x], key="corpus_split")
    frame = corpus(split).copy()
    writing = b.selectbox("Écriture", ["all", *sorted(frame.script.unique())], format_func=lambda x: "Toutes les écritures" if x == "all" else SCRIPTS[x], key="corpus_script")
    search = c.text_input("Rechercher dans les tweets", placeholder="Un mot, une expression, un hashtag…", key="corpus_search")
    filtered = frame.loc[frame.script.eq(writing)] if writing != "all" else frame
    if search:
        filtered = filtered.loc[filtered.tweet.str.contains(search, case=False, regex=False)]
    if "label" in frame:
        sentiment = st.radio("Filtrer le sentiment", ["all", *LABELS], horizontal=True,
                             format_func=lambda value: "Tous les sentiments" if value == "all" else SENTIMENT[value], key="corpus_sentiment")
        if sentiment != "all":
            filtered = filtered.loc[filtered.label.eq(sentiment)]
    x, y, z = st.columns(3)
    x.metric("Tweets dans cette sélection", number(len(filtered)))
    y.metric("Longueur médiane", f"{int(filtered.tweet.str.len().median()) if len(filtered) else 0} caractères")
    z.metric("Textes distincts après nettoyage", number(filtered.text.nunique()))
    st.button("Réinitialiser les filtres", on_click=reset_corpus_filters, key="reset_corpus_filters")
    if filtered.empty:
        empty("Aucun tweet ne correspond.", "Essayez une autre expression ou une autre écriture.")
        return
    explore, examples, audit = st.tabs(["Explorer les distributions", "Lire les tweets", "Qualité des données"])
    with explore:
        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("Les écritures")
            counts = filtered.script.value_counts()
            fig = go.Figure(go.Pie(labels=[SCRIPTS[x] for x in counts.index], values=counts.values, hole=.7, marker_colors=[FOREST, SAGE, "#C2AF97", GREY], textinfo="label+percent", hovertemplate="%{label}<br>%{value} tweets<extra></extra>"))
            fig.update_layout(showlegend=False)
            chart(fig, "corpus_scripts", 300)
        with right:
            st.subheader("Les sentiments")
            if "label" in filtered:
                counts = filtered.label.value_counts().reindex(LABELS, fill_value=0)
                bar(counts, [SENTIMENT[x] for x in LABELS], "corpus_labels", [PALETTE[x] for x in LABELS], percent=False, height=300)
            else:
                empty("Les labels test restent réservés.", "Le corpus n’affiche que les textes test. Les mesures finales et l’analyse a posteriori sont accessibles dans Robustesse.")
        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("La longueur des tweets")
            fig = go.Figure(go.Histogram(x=filtered.tweet.str.len(), nbinsx=35, marker_color=FOREST, hovertemplate="%{x} caractères<br>%{y} tweets<extra></extra>"))
            fig.update_xaxes(title="Caractères"); fig.update_yaxes(title="Tweets")
            chart(fig, "corpus_lengths")
        with right:
            st.subheader("Les fragments fréquents")
            n = st.slider("Nombre de tokens à afficher", 8, 25, 12, key="frequent_tokens")
            counts = Counter(token for text in filtered.text for token in tokens(text) if len(token) > 1 and token not in {"user", "url"})
            common = counts.most_common(n)
            if common:
                bar([x[1] for x in common], [x[0] for x in common], "frequent_words", percent=False, height=max(330, n*22))
            st.caption("Comptage descriptif des tokens regex, hors marqueurs user/url ; leur fréquence ne mesure pas une importance de modèle.")
        if "label" in filtered:
            st.subheader("La relation entre écriture et sentiment")
            counts = pd.crosstab(filtered.script, filtered.label).reindex(columns=LABELS, fill_value=0)
            fig = go.Figure()
            for label in LABELS:
                fig.add_bar(name=SENTIMENT[label], x=[SCRIPTS[x] for x in counts.index], y=counts[label], marker_color=PALETTE[label])
            fig.update_layout(barmode="stack"); fig.update_yaxes(title="Tweets")
            chart(fig, "corpus_cross")
    with examples:
        st.write("Sélectionnez une ligne pour lire le tweet et l’envoyer au laboratoire.")
        limit = st.slider("Nombre de lignes affichées", 10, 100, 25, step=5, key="corpus_limit")
        visible = filtered[["row_id", "tweet", "script", *(["label"] if "label" in filtered else [])]].head(limit)
        selection = st.dataframe(visible, hide_index=True, use_container_width=True, key="corpus_table", on_select="rerun", selection_mode="single-row", height=400)
        selected_rows = selection.selection.rows
        row = visible.iloc[selected_rows[0]] if selected_rows and selected_rows[0] < len(visible) else visible.iloc[0]
        st.markdown(f'<div class="quote" dir="auto">{esc(row.tweet)}</div>', unsafe_allow_html=True)
        st.button("Tester ce tweet dans le laboratoire →", on_click=open_phrase, args=(row.tweet,), key="try_corpus_phrase")
        download(visible, "Exporter les lignes affichées", f"darija_{split}_selection.csv", "corpus_download")
    with audit:
        quality = table("audit/counts.csv")
        st.dataframe(quality, hide_index=True, use_container_width=True)
        insight("Des fichiers officiels conservés à l’identique.", "20 doublons internes train et 2 collisions train/dev sont documentés. Aucun chevauchement train/test n’a été trouvé. Les SHA-256 et la révision Git sont enregistrés.")
        for filename, title in [("audit/collisions.csv", "Les deux collisions textuelles"), ("audit/conflicts.csv", "Contradictions train/dev")]:
            with st.expander(title):
                data = table(filename)
                if len(data):
                    st.dataframe(data, hide_index=True, use_container_width=True)
                else:
                    st.write("Aucune contradiction détectée dans cet audit.")
        st.caption("Arabe, latin et mixte décrivent des alphabets. Ils ne constituent pas une annotation fiable de la langue ou du français.")


def history_charts(rows, key):
    histories = []
    for row in rows.itertuples():
        path = ROOT / row.checkpoint / "history.csv"
        if path.exists():
            frame = pd.read_csv(path)
            if {"epoch", "loss", "dev_weighted_f1"} <= set(frame):
                histories.append((row, frame))
    if not histories:
        st.caption("Aucune courbe par époque pour cette sélection : les solveurs classiques et les modèles fixes ne produisent pas ce type d’historique.")
        return
    loss, f1 = go.Figure(), go.Figure()
    for row, frame in histories:
        name = f"{variant_name(row.variant)} · {row.seed}"
        loss.add_scatter(x=frame.epoch, y=frame.loss, mode="lines+markers", name=name)
        f1.add_scatter(x=frame.epoch, y=frame.dev_weighted_f1, mode="lines+markers", name=name)
    for fig in (loss, f1):
        fig.update_xaxes(title="Époque", dtick=1)
    loss.update_yaxes(title="Loss train")
    f1.update_yaxes(title="Weighted-F1 dev", tickformat=".0%", range=[0, 1])
    a, b = st.columns(2)
    with a:
        chart(loss, f"{key}_loss", 360)
    with b:
        chart(f1, f"{key}_f1", 360)


def models_page():
    intro("Cinq façons de lire un sentiment.", "De quelques fragments de texte à un encodeur pré-entraîné. Les méthodes, leurs résultats et leurs limites.", "Les familles A à E")
    family = st.radio("Choisir une famille", list(FAMILIES), format_func=lambda x: f"{x} · {FAMILIES[x]['name']}", horizontal=True, key="model_family")
    content = FAMILIES[family]
    rows = RUNS.loc[RUNS.family.eq(family) & RUNS.split.eq("dev") & RUNS.decision.eq("dev_offsets") & RUNS.purpose.eq("benchmark")]
    left, right = st.columns([1.18, 1], gap="large")
    with left:
        st.subheader(content["title"])
        st.write(content["intro"])
        flow = '<div class="flow">' + '<span class="flow-arrow">→</span>'.join(f'<div class="flow-step">{esc(step)}</div>' for step in content["flow"]) + '</div>'
        st.markdown(flow, unsafe_allow_html=True)
        st.markdown('<div class="lesson-grid">' + "".join(f'<div><strong>{title}</strong><p>{esc(content[key])}</p></div>' for title, key in [("Objectif", "objective"), ("Principe", "principle"), ("Exemple", "example"), ("À retenir", "limit")]) + '</div>', unsafe_allow_html=True)
    with right:
        st.subheader("Les variantes sur dev")
        if len(rows):
            metric = st.selectbox("Mesure des variantes", ["weighted_f1", "macro_f1"], format_func=METRICS.get, key="model_metric")
            summary = rows.groupby("variant")[metric].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
            bar(summary["mean"], [variant_name(x) for x in summary.index], "model_variants", height=max(310, len(summary)*29), error=summary["std"].fillna(0).to_numpy())
            st.caption("Moyenne sur les graines disponibles ; les barres indiquent l’écart-type entre graines, pas un intervalle de confiance. B est fixe.")
        else:
            empty("Une méthode prête, sans score complet.", "Le fine-tuning des deux checkpoints est implémenté et contrôlé sur un pas AdamW réel. La campagne complète n’a pas été exécutée : aucun F1 n’est inventé.")
            smoke = read_json(ROOT / "results/pretrained_smoke.json")
            if smoke:
                st.dataframe(pd.DataFrame(smoke)[["model_id", "n_train_examples", "test", "benchmark"]], hide_index=True, use_container_width=True)
    performance, explanation, attention = st.tabs(["Lire un apprentissage", "Approfondir la méthode", "Observer l’attention"])
    with performance:
        if len(rows):
            a, b = st.columns([2, 1])
            variants = a.multiselect("Variantes à tracer", sorted(rows.variant.unique()), default=[rows.groupby("variant").weighted_f1.mean().idxmax()], format_func=variant_name, key=f"curves_{family}")
            seed = b.selectbox("Graine des courbes", sorted(rows.seed.unique()), key=f"seed_{family}")
            history_charts(rows.loc[rows.variant.isin(variants) & rows.seed.eq(seed)], f"history_{family}")
            with st.expander("Métriques, temps et paramètres de tous les runs de cette famille"):
                st.dataframe(rows[["run_id", "seed", "weighted_f1", "macro_f1", "train_seconds", "trainable_parameters", "decision"]], hide_index=True, use_container_width=True)
                download(rows, "Exporter ces runs", f"darija_famille_{family}.csv", f"runs_{family}")
        else:
            st.write("Le protocole E prévoit AdamW, LR 2e-5, batch effectif 32, warmup 10 %, trois à cinq époques et trois graines. Le micro-batch est adapté explicitement aux ressources.")
    with explanation:
        st.subheader("Une question à se poser")
        st.write(f"**{content['question']}**")
        st.write(content["answer"])
        paper = next(p for p in PAPERS if p["id"] == content["paper"])
        st.markdown(f"**Recherche associée :** [{paper['title']}]({paper['source']})")
        st.write(paper["application"])
        if family == "C":
            st.write("Le Mini-Transformer utilise deux couches, 128 dimensions et quatre têtes. Le vocabulaire est construit avec une fréquence minimale de 2. Le Character-CNN utilise des caractères, pas des embeddings de mots AraVec.")
        if family == "D":
            st.dataframe(pd.DataFrame(read_json(ROOT / "results/frozen_extraction_time.json")), hide_index=True, use_container_width=True)
            st.caption("Ce coût d’extraction train/dev est partagé. Le comparatif distingue ce coût du temps du run.")
    with attention:
        if family not in {"C", "D"}:
            st.write("Les visualisations enregistrées concernent BiLSTM-attention (C) et la tête attentive sur DarijaBERT gelé (D). Choisissez une de ces familles.")
        else:
            weights = table(f"attention_{family}_5_examples.csv")
            if len(weights):
                example = st.selectbox("Un des cinq exemples dev", weights.example.unique(), key=f"attention_example_{family}")
                selected = weights.loc[weights.example.eq(example)]
                maximum = max(float(selected.attention.max()), 1e-12)
                st.markdown('<div class="tokens" dir="auto">' + "".join(f'<span class="token" title="Poids : {row.attention:.4f}" style="background:rgba(56,106,85,{.07+.6*row.attention/maximum:.3f})">{esc(row.token)}</span>' for row in selected.itertuples()) + '</div>', unsafe_allow_html=True)
                top = selected.nlargest(15, "attention")
                bar(top.attention, top.token, f"attention_plot_{family}", height=380, full_scale=False)
                st.caption("Poids normalisés sur les tokens de cet exemple. Cette vue décrit l’attention ; elle n’établit pas une cause de la prédiction.")
            else:
                st.info("Aucun poids d’attention enregistré pour cette famille.")


def robustness_page():
    intro("Un score ne raconte pas toute l’histoire.", "Changements de distribution, écritures, perturbations et incertitude : les expériences F1 à F6, après verrouillage.", "Robustesse & esprit critique")
    shift, writing, stress, uncertainty, errors = st.tabs(["F1 · Généralisation", "F2–F3 · Écritures", "F4 · Perturbations", "F5 · Incertitude", "F6 · Erreurs"])
    with shift:
        a, b = st.columns(2, gap="large")
        distributions = table("f1_label_shift.csv")
        with a:
            st.subheader("Les proportions globales")
            fig = go.Figure()
            if len(distributions):
                labels = distributions.iloc[:, 0].map(SENTIMENT)
                for name, color in [("dev", SAGE), ("test", FOREST)]:
                    fig.add_bar(name=name, x=labels, y=distributions[name], marker_color=color)
                fig.update_yaxes(tickformat=".0%", title="Proportion")
                chart(fig, "global_shift")
        with b:
            st.subheader("À l’intérieur de chaque écriture")
            conditional = table("f1_conditional_labels.csv")
            if len(conditional):
                label = st.selectbox("Sentiment à suivre", LABELS, index=1, format_func=SENTIMENT.get, key="shift_label")
                fig = go.Figure()
                for split, color in [("train", "#B7C4AE"), ("dev", SAGE), ("test", FOREST)]:
                    part = conditional.loc[conditional.split.eq(split) & conditional.label.eq(label) & conditional.script.ne("other")]
                    fig.add_bar(name=split, x=part.script.map(SCRIPTS), y=part.proportion, marker_color=color)
                fig.update_yaxes(tickformat=".0%", title="Proportion dans l’écriture")
                chart(fig, "conditional_shift")
        insight("Le raccourci change de sens.", "Les textes latins sont neutres à 52,6 % sur dev, contre 29,9 % sur test. Le candidat en classe pourtant 86,8 % comme neutres. C’est un constat compatible avec un raccourci ; son mécanisme causal reste à démontrer.")
        chosen = set(COMPARISON.run_id.dropna())
        rules = RUNS.loc[RUNS.run_id.isin(chosen) & RUNS.split.eq("test")]
        fig = go.Figure()
        for rule, color, name in [("argmax", SAGE, "Argmax brut"), ("dev_offsets", FOREST, "Règle choisie sur dev")]:
            part = rules.loc[rules.decision.eq(rule)].sort_values("family")
            fig.add_bar(name=name, x=part.family, y=part.weighted_f1, marker_color=color)
        fig.update_yaxes(tickformat=".0%", title="Weighted-F1 test", range=[0, 1])
        chart(fig, "decision_rules")
        st.caption("Analyse a posteriori : même si une autre règle obtient un meilleur score test, elle ne remplace pas le choix verrouillé sur dev.")
        with st.expander("Lire le diagnostic complet et les marqueurs du corpus"):
            st.markdown((ROOT / "results/generalization_analysis.md").read_text(encoding="utf-8"))
            st.dataframe(table("corpus_markers.csv"), hide_index=True, use_container_width=True)
    with writing:
        a, b = st.columns(2, gap="large")
        with a:
            st.subheader("F2 · Transférer entre écritures")
            cross = table("f2_cross_script.csv")
            if len(cross):
                group = cross.groupby(["train_script", "test_script"]).weighted_f1.mean()
                bar(group.values, [f"{SCRIPTS[a]} → {SCRIPTS[b]}" for a, b in group.index], "cross_script", height=290)
                st.caption("TF-IDF union, sous-ensemble train d’une seule écriture, trois graines. Aucun transfert entre pays.")
        with b:
            st.subheader("F3 · La couverture du vocabulaire")
            oov = table("oov.csv")
            fig = go.Figure()
            if len(oov):
                for script, color in [("arabic", FOREST), ("latin", RED), ("mixed", SAGE)]:
                    part = oov.loc[oov.script.eq(script)].set_index("split").reindex(["train", "dev", "test"])
                    fig.add_scatter(x=part.index, y=part.oov_rate, name=SCRIPTS[script], mode="lines+markers", line=dict(color=color))
                fig.update_yaxes(tickformat=".0%", title="Taux OOV")
                chart(fig, "oov_rates", 290)
            st.caption("Vocabulaire de mots min_freq=2 ; ces OOV ne mesurent pas directement les n-grammes du TF-IDF caractères.")
        st.subheader("Les performances test par écriture")
        records = []
        for row in COMPARISON.dropna(subset=["scripts"]).itertuples():
            records.extend({"family": row.family, **item} for item in json.loads(row.scripts))
        if records:
            frame = pd.DataFrame(records)
            heat = frame.pivot(index="family", columns="script", values="weighted_f1").reindex(columns=["arabic", "latin", "mixed"])
            fig = go.Figure(go.Heatmap(z=heat.values, x=[SCRIPTS[x] for x in heat.columns], y=heat.index, colorscale=[[0, "#F0F2E9"], [1, FOREST]], zmin=0, zmax=1, texttemplate="%{z:.1%}", hovertemplate="Famille %{y}<br>%{x} : %{z:.4f}<extra></extra>"))
            fig.update_yaxes(autorange="reversed")
            chart(fig, "scripts_heatmap", 340)
        st.caption("Les effectifs mixtes sont faibles. La translittération facultative n’a pas été exécutée et ne reçoit aucun score fictif.")
    with stress:
        st.subheader("Ce qui change sans emojis ni ponctuation répétée")
        rows = []
        for row in COMPARISON.dropna(subset=["robustness_f4"]).itertuples():
            rows.append({"family": row.family, **json.loads(row.robustness_f4)})
        if rows:
            frame = pd.DataFrame(rows)
            bar(frame.flip_rate_changed, frame.family, "perturbations", height=300)
            st.dataframe(frame[["family", "eligible", "changed", "flip_rate_changed"]], hide_index=True, use_container_width=True)
        st.caption("Tweets polarisés initialement corrects, puis réellement modifiés. Chaque famille a son propre sous-ensemble : ce n’est pas un classement causal sur une population commune.")
        sample = table("f4_perturbations.csv")
        if len(sample):
            st.dataframe(sample[["text", "changed_text", "label", "altered_prediction"]].head(12), hide_index=True, use_container_width=True)
    with uncertainty:
        st.subheader("L’incertitude se mesure aussi")
        measured = COMPARISON.dropna(subset=["weighted_f1", "ci_low", "ci_high"])
        fig = go.Figure(go.Scatter(x=measured.weighted_f1, y=measured.family, mode="markers", marker=dict(color=FOREST, size=12), error_x=dict(type="data", symmetric=False, array=measured.ci_high-measured.weighted_f1, arrayminus=measured.weighted_f1-measured.ci_low, color=FOREST), hovertemplate="Famille %{y}<br>Weighted-F1 %{x:.4f}<extra></extra>"))
        fig.update_xaxes(title="Weighted-F1 test · IC bootstrap 95 %", tickformat=".0%", range=[0, 1])
        fig.update_yaxes(autorange="reversed")
        chart(fig, "confidence_intervals", 300)
        st.caption("1 000 rééchantillonnages de tweets, conditionnellement au checkpoint graine 42. La variabilité entre graines est une autre source d’incertitude.")
        st.dataframe(measured[["family", "test_seed_mean", "test_seed_std", "n_seeds", "ci_low", "ci_high"]], hide_index=True, use_container_width=True)
        test = read_json(ROOT / "results/f5_mcnemar.json")
        if test:
            insight("McNemar : comparer les erreurs appariées.", f"{test['first']} vs {test['second']} : p = {test['p_exact']:.3f}. Ce test porte sur les taux d’erreur, pas directement sur le F1 ; il ne démontre pas un écart à 5 % lorsque p > 0,05.")
    with errors:
        st.subheader("Trente erreurs, trente pistes de lecture")
        data = table("f6_30_erreurs.csv")
        if len(data):
            counts = data.categorie_provisoire.value_counts()
            bar(counts, [x.replace("_", " ") for x in counts.index], "error_categories", percent=False, height=300)
            category = st.selectbox("Catégorie automatique", ["all", *data.categorie_provisoire.unique()], format_func=lambda x: "Toutes les catégories" if x == "all" else x.replace("_", " "), key="error_category")
            subset = data if category == "all" else data.loc[data.categorie_provisoire.eq(category)]
            st.dataframe(subset[["row_id", "tweet", "label", "prediction", "categorie_provisoire", "validation_humaine"]], hide_index=True, use_container_width=True, height=350)
            download(data, "Exporter les 30 erreurs et leurs votes", "darija_30_erreurs.csv", "errors_download")
        st.caption("Catégories provisoires, pas des causes démontrées. Un désaccord entre annotateurs ne prouve pas que le label officiel est faux.")
        with st.expander("Lire l’analyse assistée des 30 exemples", expanded=st.session_state.get("presentation_mode", False)):
            st.markdown((ROOT / "results/error_analysis.md").read_text(encoding="utf-8"))


def research_page():
    intro("Comprendre, relier, expliquer.", "Les publications qui motivent les expériences, les solutions réellement essayées et des repères pour la présentation orale.", "Le carnet de recherche")
    papers, solutions, oral, glossary = st.tabs(["Les publications", "Problèmes & solutions", "Préparer l’oral", "Le petit lexique"])
    with papers:
        query = st.text_input("Rechercher une publication", placeholder="Arabizi, DarijaBERT, fastText…", key="paper_search")
        options = [p for p in PAPERS if query.casefold() in " ".join(str(v) for v in p.values()).casefold()]
        if options:
            selected = st.selectbox("Publication", options, format_func=lambda p: f"{p['title']} · {p['year']}", key="paper_selected")
            st.markdown(f'<div class="paper-title">{esc(selected["title"])}</div><div class="source-line">{esc(selected["authors"])} · {esc(selected["year"])}</div>', unsafe_allow_html=True)
            a, b = st.columns(2, gap="large")
            with a:
                st.markdown("**Le problème**"); st.write(selected["problem"])
                st.markdown("**La solution des auteurs**"); st.write(selected["solution"])
                st.markdown("**Ce qu’ils ont réellement démontré**"); st.write(selected["result"])
            with b:
                st.markdown("**Dans ce projet**"); st.write(selected["application"])
                st.markdown("**L’expérience proposée**"); st.write(selected["experiment"])
                st.link_button("Lire la source primaire ↗", selected["source"])
        else:
            empty("Aucune publication correspondante.", "Essayez un nom d’auteur ou une méthode.")
        st.caption("11 fiches vérifiées. Les résultats publiés ne sont jamais présentés comme nos scores. La recherche est ciblée, non exhaustive ; l’accès automatisé à Google Scholar n’a pas abouti.")
    with solutions:
        frame = table("research_solutions.csv")
        for row in frame.to_dict("records"):
            with st.expander(f"{row['Problème']} — {row['Statut']}"):
                st.write(f"**Observation :** {row['Observation mesurée']}")
                st.write(f"**Solution étudiée :** {row['Solution proposée']}")
                st.write(f"**Résultat réel :** {row['Résultat après essai']}")
                st.link_button("Publication associée ↗", row["Publication"])
        st.caption("Les gains ou baisses indiqués concernent dev. Aucune idée issue d’un article n’est déclarée automatiquement efficace sur test.")
    with oral:
        st.subheader("Une présentation en cinq temps")
        steps = [("Le problème", "Reconnaître trois sentiments dans une darija aux graphies variables."), ("Les données", "Expliquer ary, les partitions officielles et les limites de l’audit."), ("Les méthodes", "Présenter A à E, du TF-IDF au fine-tuning."), ("Les résultats", "Lire les chiffres réels et la baisse sur test, sans masquer les runs absents."), ("Les limites", "Distinguer observation, hypothèse, vérification technique et expérience complète.")]
        st.markdown('<div class="timeline">' + "".join(f'<div class="timeline-step"><span class="step-number">{i}</span><div><strong>{esc(title)}</strong><p>{esc(text)}</p></div></div>' for i, (title, text) in enumerate(steps, 1)) + '</div>', unsafe_allow_html=True)
        st.info("Les énoncés officiels Q1–Q12 sont encore attendus. Les douze thèmes du document d’oral sont provisoires et ne prétendent pas les remplacer.")
        with st.expander("Ouvrir les douze repères et questions probables"):
            st.markdown((ROOT / "research/oral.md").read_text(encoding="utf-8"))
        a, b = st.columns(2)
        for column, name, label, mime in [(a, "TP2_Darija.html", "Télécharger le notebook lisible", "text/html"), (b, "TP2_Darija.ipynb", "Télécharger le notebook exécutable", "application/x-ipynb+json")]:
            path = ROOT / "notebooks" / name
            if path.exists():
                column.download_button(label, path.read_bytes(), name, mime, use_container_width=True)
    with glossary:
        term = st.selectbox("Une notion à éclaircir", list(GLOSSARY), key="glossary_term")
        insight(term, GLOSSARY[term])
        st.write("Le F1 combine précision et rappel. Un score de 0,54 n’est pas une probabilité qu’une nouvelle phrase soit correcte : c’est une mesure calculée sur un ensemble annoté.")
        st.latex(r"F1_c=\frac{2\,P_c\,R_c}{P_c+R_c}")


st.markdown(f'<style>{(ROOT / "assets/style.css").read_text(encoding="utf-8")}</style>', unsafe_allow_html=True)
# Streamlit efface les widgets masqués : conserver le brouillon et les choix du laboratoire.
for key in ("lab_text", "advanced_models", "chosen_simple", "chosen_advanced"):
    if key in st.session_state:
        st.session_state[key] = st.session_state[key]
RUNS = table("results.csv")
COMPARISON = table("comparison.csv")
CATALOG = model_catalog()
if RUNS.empty or COMPARISON.empty:
    st.title("Darija / Sentiment")
    st.info("Les résultats enregistrés ne sont pas encore disponibles. Exécutez le notebook pour préparer la présentation.")
    st.stop()
brand, navigation, mode = st.columns([2.45, 7.6, 1.6], gap="medium")
with brand:
    st.markdown('<div class="brand"><span class="brand-mark">D</span>Darija / Sentiment</div>', unsafe_allow_html=True)
with navigation:
    with st.container(key="top_navigation"):
        page = st.radio("Navigation", PAGES, horizontal=True, key="navigation", label_visibility="collapsed")
with mode:
    with st.container(key="presentation_control"):
        st.toggle("Mode soutenance", key="presentation_mode", help="Ajoute des repères pour présenter les résultats à l’oral.")
st.markdown('<div class="top-rule"></div>', unsafe_allow_html=True)
if st.session_state.presentation_mode:
    st.caption(f"Groupe 12 · Master Big Data · {PAGES.index(page)+1}/{len(PAGES)} · {page}")
{"Vue d’ensemble": overview, "Laboratoire": laboratory, "Corpus": corpus_page, "Modèles": models_page, "Robustesse": robustness_page, "Recherche": research_page}[page]()
if st.session_state.presentation_mode:
    st.divider()
    previous, center, following = st.columns([1, 2, 1])
    index = PAGES.index(page)
    if index:
        previous.button("← Étape précédente", on_click=navigate, args=(PAGES[index-1],), use_container_width=True)
    center.caption("À l’oral : expliquer le constat, citer sa preuve, puis énoncer sa limite.")
    if index < len(PAGES)-1:
        following.button("Étape suivante →", on_click=navigate, args=(PAGES[index+1],), use_container_width=True)
st.markdown('<div class="footer"><span>Darija / Sentiment — TP2 · Groupe 12</span><span>AfriSenti · ary uniquement · Résultats réels, choix verrouillés sur dev.</span></div>', unsafe_allow_html=True)
