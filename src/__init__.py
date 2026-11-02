"""TP2 Darija : petits modules lisibles, sans framework expérimental."""
import os

# Le TP utilise PyTorch ; évite de charger TensorFlow en plus dans Transformers.
os.environ.setdefault("USE_TF", "0")
# Le téléchargement HTTPS standard a été vérifié sur cette machine ; Xet y stagnait.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
