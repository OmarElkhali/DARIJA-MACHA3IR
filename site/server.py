"""Serveur local du site HTML/CSS/JS et des six véritables modèles de campaign_v4."""
from functools import lru_cache, partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import threading
import time

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / 'experiments/campaign_v4'
DIST = Path(__file__).resolve().parent / 'dist'
os.environ.update(DARIJA_OUTPUT_ROOT=str(ROOT), DARIJA_CLEANING='conservative_v2', DARIJA_MAX_LENGTH='128')
sys.path.insert(0, str(PROJECT))
INFERENCE_LOCK = threading.Lock()
HOSTED_ORIGIN = 'https://darija-sentiment-laboratoire.speedypaste.chatgpt.site'


def representatives():
    path = ROOT / 'results/selection_lock.json'
    if path.exists():
        lock = json.loads(path.read_text(encoding='utf-8'))
        selected = {w['family']: f"{w['family']}_{w['variant']}_s42" for w in lock['selected_variants']}
    else:
        measured = json.loads((DIST / 'data.json').read_text(encoding='utf-8'))
        selected = {w['family']: w['run_id'] for w in measured['comparison']}
    if set(selected) != set('ABCDEP'):
        raise ValueError('Six représentants A/B/C/D/E/P sont requis dans le verrou.')
    return selected


@lru_cache(maxsize=6)
def predictor(run):
    from src.train import load_predictor
    folder = ROOT / 'models' / run
    manifest = json.loads((folder / 'artifact_sha256.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file():
            raise ValueError(f'Artefact absent : {run}')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != expected:
            raise ValueError(f'Intégrité incorrecte : {run}')
    meta = json.loads((folder / 'meta.json').read_text(encoding='utf-8'))
    return load_predictor(folder, device='cpu'), meta


def predict_all(text):
    import numpy as np
    from src.preprocess import LABELS, clean_text, script_type
    from src.evaluate import decide
    results = []
    with INFERENCE_LOCK:
        for family, run in sorted(representatives().items()):
            start = time.perf_counter()
            try:
                model, meta = predictor(run)
                prepared = clean_text(text, meta['config'].get('cleaning', 'conservative_v2')) or '<EMPTY>'
                probabilities = np.asarray(model([prepared]), dtype=float)
                label = str(decide(probabilities, meta['offsets'])[0])
                scores = np.log(np.clip(probabilities[0], 1e-12, 1)) + np.asarray(meta['offsets'])
                scores = np.exp(scores - scores.max())
                scores /= scores.sum()
                results.append({'family': family, 'run_id': run, 'sentiment': label,
                    'probabilities': dict(zip(LABELS, probabilities[0].tolist())),
                    'decision_scores': dict(zip(LABELS, scores.tolist())),
                    'offsets': meta['offsets'], 'milliseconds': round((time.perf_counter() - start) * 1000, 1),
                    'status': 'ok'})
            except Exception as error:
                print(f'Inférence {run}: {type(error).__name__}: {error}', flush=True)
                results.append({'family': family, 'run_id': run, 'status': 'error',
                    'error': 'Modèle indisponible. Vérifier les poids et le cache des encodeurs dans le serveur local.'})
    return {'text': text, 'cleaned_text': clean_text(text), 'script': script_type(text),
            'results': results, 'score_note': 'Scores de décision normalisés, non calibrés. Décalages fixés sur validation.'}


class Handler(SimpleHTTPRequestHandler):
    def origin_allowed(self):
        origin = self.headers.get('Origin')
        return origin in {None, f'http://127.0.0.1:{self.server.server_port}',
                          f'http://localhost:{self.server.server_port}', HOSTED_ORIGIN,
                          'https://omarelkhali.github.io'}

    def send_json(self, value, status=200):
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        if self.origin_allowed() and self.headers.get('Origin'):
            self.send_header('Access-Control-Allow-Origin', self.headers['Origin'])
            self.send_header('Vary', 'Origin')
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        if not self.origin_allowed():
            self.send_json({'error': 'Origine non autorisée.'}, 403)
            return
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', self.headers.get('Origin', ''))
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/api/status':
            try:
                selected = representatives()
                available = []
                for family, run in selected.items():
                    folder = ROOT / 'models' / run
                    meta = json.loads((folder / 'meta.json').read_text(encoding='utf-8'))
                    required = 'classifier.joblib' if meta['kind'] == 'classical' else 'pretrained.json' if meta['kind'] == 'pretrained' else 'weights.pt'
                    if (folder / required).is_file():
                        available.append(family)
                self.send_json({'status': 'ready' if len(available) == 6 else 'partial', 'engine': 'PyTorch CPU',
                                'families': selected, 'available_families': available,
                                'trained_models': len(available), 'stores_phrases': False})
            except (OSError, ValueError, KeyError) as error:
                self.send_json({'status': 'unavailable', 'error': str(error)}, 503)
            return
        super().do_GET()

    def do_POST(self):
        if self.path != '/api/predict':
            self.send_json({'error': 'Route inconnue.'}, 404)
            return
        if not self.origin_allowed():
            self.send_json({'error': 'Origine non autorisée.'}, 403)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 1 <= length <= 16384 or self.headers.get_content_type() != 'application/json':
                raise ValueError('Une requête JSON de moins de 16 Kio est attendue.')
            body = json.loads(self.rfile.read(length))
            text = body.get('text') if isinstance(body, dict) else None
            if not isinstance(text, str) or not text.strip() or len(text) > 2000:
                raise ValueError('Écrire une phrase de 1 à 2 000 caractères.')
        except (ValueError, UnicodeError) as error:
            self.send_json({'error': str(error)}, 400)
            return
        self.send_json(predict_all(text.strip()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, directory=str(DIST)))
    print(f'Darija / Sentiment : http://127.0.0.1:{args.port}', flush=True)
    server.serve_forever()
