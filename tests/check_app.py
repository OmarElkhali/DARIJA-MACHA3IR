"""Navigation, graphiques, état de session et inférence réelle de la présentation."""
from pathlib import Path
import sys
import json
from datetime import datetime, timezone
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest
from src.preprocess import ROOT, save_json, sha256
from src.evaluate import assert_locked


def widget(elements, key):
    return next(element for element in elements if element.key == key)


def submit(app):
    next(button for button in app.button if button.label == "Analyser la phrase").click().run()
    assert not app.exception, app.exception


def goto(app, page):
    widget(app.radio, "navigation").set_value(page).run()
    assert not app.exception, app.exception


def main():
    path = Path(__file__).resolve().parents[1] / "app.py"
    assert_locked()
    app = AppTest.from_file(str(path), default_timeout=120).run()
    assert not app.exception, app.exception
    pages = {}
    for page in widget(app.radio, "navigation").options:
        goto(app, page)
        pages[page] = {"charts": len(app.get("plotly_chart")), "exceptions": len(app.exception)}
    goto(app, "Corpus")
    widget(app.radio, "corpus_sentiment").set_value("negative").run()
    negative_tweets = next(frame.value for frame in app.dataframe if "row_id" in frame.value.columns)
    assert negative_tweets.label.eq("negative").all()
    widget(app.selectbox, "corpus_split").set_value("test").run()
    tweets = next(frame.value for frame in app.dataframe if "row_id" in frame.value.columns)
    assert "label" not in tweets.columns and len(tweets)
    widget(app.text_input, "corpus_search").set_value("phrase_qui_n_existe_pas_58309").run()
    assert not app.exception and not app.dataframe
    next(button for button in app.button if button.key == "reset_corpus_filters").click().run()
    assert widget(app.text_input, "corpus_search").value == ""
    assert widget(app.selectbox, "corpus_script").value == "all"
    next(button for button in app.button if button.key == "try_corpus_phrase").click().run()
    assert widget(app.radio, "navigation").value == "Laboratoire"
    assert widget(app.text_area, "lab_text").value == tweets.tweet.iloc[0]
    goto(app, "Modèles")
    for family in "ABCDE":
        widget(app.radio, "model_family").set_value(family).run()
        assert not app.exception, app.exception
    widget(app.radio, "model_family").set_value("C").run()
    attention_selector = widget(app.selectbox, "attention_example_C")
    attention_selector.set_value(attention_selector.options[-1]).run()
    assert not app.exception
    widget(app.radio, "model_family").set_value("B").run()
    assert widget(app.selectbox, "seed_B").options == ["42"]
    goto(app, "Laboratoire")
    widget(app.text_area, "lab_text").set_value("brouillon personnel à conserver").run()
    widget(app.toggle, "advanced_models").set_value(True).run()
    widget(app.multiselect, "chosen_advanced").set_value(["A_word_s43"]).run()
    goto(app, "Corpus")
    goto(app, "Laboratoire")
    assert widget(app.text_area, "lab_text").value == "brouillon personnel à conserver"
    assert widget(app.toggle, "advanced_models").value
    assert widget(app.multiselect, "chosen_advanced").value == ["A_word_s43"]
    original_is_file = Path.is_file
    with patch.object(Path, "is_file", lambda path: False if path == ROOT / "models/C_attention_fasttext_s42/vocab.json" else original_is_file(path)):
        app.run()
        assert not app.exception
        assert "C_attention_fasttext_s42" not in widget(app.multiselect, "chosen_advanced").options
    original_read_text = Path.read_text
    with patch.object(Path, "read_text", lambda path, *a, **kw: "{" if path == ROOT / "models/A_char_s44/meta.json" else original_read_text(path, *a, **kw)):
        app.run()
        assert not app.exception and app.warning
        assert "A_char_s44" not in widget(app.multiselect, "chosen_advanced").options
    widget(app.toggle, "advanced_models").set_value(False).run()
    widget(app.text_area, "lab_text").set_value("hadchi zwin bzaf")
    ids = ["A_char_s42", "B_camelbert_s42", "C_attention_fasttext_s42", "D_states_s42"]
    widget(app.multiselect, "chosen_simple").set_value(ids)
    with patch("requests.sessions.Session.request", side_effect=AssertionError("L’inférence ne doit pas contacter Internet")) as network:
        submit(app)
        assert network.call_count == 0
    predictions = app.session_state["last_analysis"]["predictions"]
    assert len(predictions) == 4 and not app.session_state["last_analysis"]["failures"]
    assert any("Égalité :" in item.value for item in app.markdown), "La comparaison doit rendre visible le partage des votes"
    for prediction in predictions:
        assert prediction["prediction"] in {"negative", "neutral", "positive"}
        assert abs(sum(prediction[x] for x in ("negative", "neutral", "positive")) - 1) < 1e-6
    widget(app.selectbox, "result_model").set_value("D_states_s42").run()
    assert any('class="result-model">États + BiLSTM-attention · graine 42' in item.value for item in app.markdown)
    widget(app.multiselect, "chosen_simple").set_value(["A_char_s42"]).run()
    assert any("modèles ont changé" in item.value for item in app.info)
    widget(app.text_area, "lab_text").set_value("")
    submit(app)
    assert len(app.warning) >= 1
    widget(app.text_area, "lab_text").set_value("une nouvelle phrase")
    widget(app.multiselect, "chosen_simple").set_value([])
    submit(app)
    assert len(app.warning) >= 1
    widget(app.multiselect, "chosen_simple").set_value(["A_char_s42"])
    with patch("src.train.load_predictor", side_effect=OSError("Checkpoint absent — test contrôlé")):
        submit(app)
        assert len(app.error) == 1 and not app.session_state["last_analysis"]["predictions"]
    with patch("src.train.load_predictor", return_value=lambda texts: [[float("nan"), 0, 1]]):
        submit(app)
        assert len(app.error) == 1 and not app.session_state["last_analysis"]["predictions"]
    widget(app.text_area, "lab_text").set_value("هاد الخدمة زوينة بزاف")
    submit(app)
    assert len(app.session_state["last_analysis"]["predictions"]) == 1
    widget(app.toggle, "advanced_models").set_value(True).run()
    assert len(widget(app.multiselect, "chosen_advanced").options) == 76
    widget(app.multiselect, "chosen_advanced").set_value(["A_char_s42", "A_char_s43"])
    submit(app)
    comparison_plot = next(element for element in app.get("plotly_chart") if element.proto.id.endswith("-live_scores"))
    plot_spec = json.loads(comparison_plot.proto.spec)
    assert len(set(plot_spec["data"][0]["x"])) == 2, "Les graines distinctes doivent rester identifiables dans le graphique"
    widget(app.selectbox, "history_phrase").set_value(0).run()
    next(button for button in app.button if button.key == "replay_phrase").click().run()
    assert widget(app.text_area, "lab_text").value == app.session_state["prediction_history"][0]["phrase"]
    widget(app.toggle, "presentation_mode").set_value(True).run()
    next(button for button in app.button if button.label == "Étape suivante →").click().run()
    assert widget(app.radio, "navigation").value == "Corpus"
    assert app.session_state["prediction_history"]
    assert_locked()
    report = {"verified_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": {name: sha256(ROOT / name) for name in ("app.py", "assets/style.css", "src/models.py", "src/train.py", "src/evaluate.py", "tests/check_app.py")},
              "pages": pages, "families_documented": list("ABCDE"), "real_prediction_families": [row["family"] for row in predictions],
              "real_predictions": predictions, "available_checkpoints": 76, "corpus_test_labels_hidden": True,
              "empty_input": True, "empty_model_selection": True, "missing_checkpoint_handled": True,
              "corpus_to_laboratory": True, "session_history": True, "presentation_navigation": True,
              "distinct_seed_labels": True, "history_replay": True,
              "draft_and_model_selection_survive_navigation": True, "stale_model_selection_warning": True,
              "incomplete_and_malformed_checkpoints_handled": True, "invalid_probabilities_rejected": True,
              "local_inference_without_http_requests": True, "all_prediction_panels_selectable": True,
              "corpus_filters_reset": True,
              "benchmark_checkpoints_unchanged": True, "browser_visual_verification": "non réalisée : contrôle navigateur arrêté par l’outil"}
    save_json(ROOT / "results/ui_verification.json", report)
    print(json.dumps({"pages": pages, "real_models_tested": ids, "checks": "passed"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
