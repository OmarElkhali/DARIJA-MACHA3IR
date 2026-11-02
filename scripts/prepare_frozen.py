"""Une seule extraction train/dev pour les trois variantes D ; pas de labels test."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.preprocess import ROOT, load_split, save_json
from src.models import FrozenEncoder
from src.train import feature_cache_path, seed_everything

seed_everything(42)
encoder = FrozenEncoder()
records = []
for split in ("train", "dev"):
    started = time.perf_counter()
    if all(feature_cache_path(split, p).exists() for p in ("cls", "mean", "states")):
        print("Déjà extrait :", split, flush=True)
        continue
    print("Extraction BERT gelé :", split, flush=True)
    features, mask = encoder.extract(load_split(split).text.tolist(), "all", batch_size=4)
    for pooling, array in features.items():
        np.savez_compressed(feature_cache_path(split, pooling), features=array, mask=mask)
        print(split, pooling, array.shape, array.dtype, flush=True)
    records.append({"split": split, "seconds": time.perf_counter() - started, "n": len(mask)})
    del features, mask
    save_json(ROOT / "results/frozen_extraction_time.json", records)
