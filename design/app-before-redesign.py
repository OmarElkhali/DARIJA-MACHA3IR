"""Application locale ; charge séquentiellement les seuls checkpoints disponibles."""
from pathlib import Path
import gc
import json

import pandas as pd
import streamlit as st

from src.preprocess import ROOT, LABELS, load_split, clean_text
from src.evaluate import results_table, decide
from src.train import load_predictor, plan_table

st.set_page_config(page_title="TP2 · Sentiment en Darija", page_icon="🇲🇦", layout="wide")
st.title("TP2 · Analyse de sentiment en Darija")
st.caption("Groupe 12 · Corpus marocain ary · Mesures issues des fichiers enregistrés")
page = st.sidebar.radio("Navigation", ["Dashboard", "Dataset", "Tester une phrase"])


def read_csv(path):
    return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()


def show_image(path):
    if Path(path).exists():
        st.image(str(path))


if page == "Dashboard":
    comparison = read_csv(ROOT / "results/comparison.csv")
    if comparison.empty:
        st.info("Évaluation finale non exécutée. Les résultats dev ci-dessous servent à la sélection.")
    else:
        st.subheader("Comparaison finale — choix verrouillés sur dev")
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        measured = comparison.dropna(subset=["weighted_f1", "macro_f1"])
        if len(measured):
            st.bar_chart(measured.set_index("family")[["weighted_f1", "macro_f1"]])
    st.subheader("Journal des expériences réelles")
    table = results_table()
    if len(table):
        split = st.selectbox("Partition", sorted(table.split.unique()))
        st.dataframe(table.loc[table.split == split], use_container_width=True, hide_index=True)
    else:
        st.info("Aucun run enregistré.")
    st.subheader("Couverture des variantes")
    st.dataframe(plan_table(), hide_index=True, use_container_width=True)
    show_image(ROOT / "results/figures/modele_final.png")
    for family in "CDE":
        show_image(ROOT / f"results/figures/curves_{family}.png")
elif page == "Dataset":
    split = st.selectbox("Partition à explorer", ["train", "dev", "test (textes seulement)"])
    try:
        frame = load_split(split.split()[0])
    except FileNotFoundError:
        st.error("Télécharger les données avec le notebook ou la commande indiquée dans README.md.")
        st.stop()
    st.metric("Tweets", len(frame))
    st.bar_chart(frame.script.value_counts())
    if "label" in frame:
        st.bar_chart(frame.label.value_counts())
    st.dataframe(frame.head(30), use_container_width=True, hide_index=True)
    st.caption("Latin ne signifie pas automatiquement Arabizi ; mixed ne prouve pas la présence de français.")
    for name in ("counts", "collisions", "conflicts"):
        path = ROOT / f"results/audit/{name}.csv"
        if path.exists():
            st.subheader(name)
            st.dataframe(read_csv(path), use_container_width=True, hide_index=True)
else:
    st.write("Saisissez une phrase en darija arabe ou Arabizi. Chaque modèle sélectionné est chargé puis libéré avant le suivant.")
    folders = []
    for file in sorted((ROOT / "models").glob("*/meta.json")):
        meta = json.loads(file.read_text(encoding="utf-8"))
        kind = meta["kind"]
        required = "classifier.joblib" if kind == "classical" or (kind == "frozen" and meta["config"]["pooling"] != "states") else "pretrained.json" if kind == "pretrained" else "weights.pt"
        if (file.parent / required).exists():
            folders.append(file.parent.name)
    with st.form("prediction"):
        text = st.text_area("Votre phrase", "هاد الخدمة زوينة بزاف")
        defaults = [name for name in folders if name == "A_char_s42"] or folders[:1]
        chosen = st.multiselect("Checkpoints disponibles", folders, default=defaults)
        submit = st.form_submit_button("Analyser")
    if submit:
        if not text.strip():
            st.warning("Saisissez une phrase non vide.")
        elif not chosen:
            st.warning("Sélectionnez au moins un checkpoint disponible.")
        elif len(text) > 10000:
            st.warning("Limite : 10 000 caractères par phrase.")
        else:
            predictions = []
            for name in chosen:
                folder = ROOT / "models" / name
                meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
                with st.spinner(f"Prédiction : {name}"):
                    try:
                        predictor = load_predictor(folder)
                        probabilities = predictor([clean_text(text)])
                        row = {"Modèle": name, "Sentiment": decide(probabilities, meta["offsets"])[0], "Règle": f"argmax(log(p) + {meta['offsets']})"}
                        row.update({label: float(probabilities[0, i]) for i, label in enumerate(LABELS)})
                        predictions.append(row)
                        del predictor
                    except (OSError, RuntimeError, ValueError) as error:
                        st.error(f"{name} indisponible : {error}")
                    gc.collect()
            if predictions:
                st.dataframe(pd.DataFrame(predictions), hide_index=True, use_container_width=True)
                st.caption("Scores bruts normalisés, non calibrés. La règle dev peut déplacer la classe gagnante. Une phrase isolée ne mesure pas la qualité générale du modèle.")
