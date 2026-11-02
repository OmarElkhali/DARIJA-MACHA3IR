"""Vérifie l'amorçage autonome sans pip, réseau, modèles ou entraînement."""
import ast
import builtins
import base64
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import shutil
import sys
from collections import namedtuple
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from zipfile import ZipFile

PROJECT = Path(__file__).resolve().parents[1]


def main():
    helpers = runpy.run_path(str(PROJECT / "scripts/notebook_bootstrap.py"))
    resources = runpy.run_path(str(PROJECT / "src/resources.py"))
    nb = json.loads((PROJECT / "notebooks/TP2_Darija.ipynb").read_text(encoding="utf-8"))
    startup = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"][:6]
    syntax = ast.parse("\n".join(startup))
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in syntax.body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in {"EMBEDDED_PROJECT", "EMBEDDED_PROJECT_SHA256"}}
    payload, digest = constants["EMBEDDED_PROJECT"], constants["EMBEDDED_PROJECT_SHA256"]
    assert hashlib.sha256(base64.b64decode(payload)).hexdigest() == digest
    with TemporaryDirectory(prefix="darija_bootstrap_check_") as temporary:
        root = Path(temporary)
        # La configuration doit remplacer toute la grappe src d'un ancien runtime.
        prefix = (PROJECT / "scripts/notebook_setup.py").read_text(encoding="utf-8").split("try:\n    import numpy", 1)[0]
        old_project, new_project = root / "old_imports", root / "new_imports"
        for project in (old_project, new_project):
            (project / "src").mkdir(parents=True)
            (project / "src/__init__.py").write_text("", encoding="utf-8")
            (project / "src/preprocess.py").write_text(
                "import os\nfrom pathlib import Path\nROOT=Path(os.environ['DARIJA_OUTPUT_ROOT'])\nCLEANING=os.environ['DARIJA_CLEANING']\n", encoding="utf-8")
            (project / "src/helper.py").write_text("from .preprocess import ROOT\n", encoding="utf-8")
        namespace = {"PROJECT": new_project.resolve(), "OUTPUT": root / "new_output", "CONFIG_OVERRIDES": {},
                     "os": os, "sys": sys, "Path": Path, "hashlib": hashlib}
        with patch.dict(sys.modules), patch.object(sys, "path", [str(old_project), *sys.path, str(new_project)]), patch.dict(os.environ, {"DARIJA_OUTPUT_ROOT": str(root / "old_output"), "DARIJA_CLEANING": "historical"}):
            for name in list(sys.modules):
                if name == "src" or name.startswith("src."):
                    sys.modules.pop(name)
            old_helper = importlib.import_module("src.helper")
            exec(compile(prefix, "setup_imports", "exec"), namespace)
            new_helper = importlib.import_module("src.helper")
            assert new_helper is not old_helper and new_helper.ROOT == namespace["OUTPUT"]
            assert Path(sys.modules["src.preprocess"].__file__).resolve().is_relative_to(new_project.resolve())
            namespace["_DARIJA_IMPORTED_SOURCE_HASHES"] = namespace["current_sources"]
            exec(compile(prefix, "setup_same_imports", "exec"), namespace)
            assert importlib.import_module("src.helper") is new_helper
            namespace["OUTPUT"] = root / "other_output"
            exec(compile(prefix, "setup_other_output", "exec"), namespace)
            assert importlib.import_module("src.helper").ROOT == namespace["OUTPUT"]
            namespace["_DARIJA_IMPORTED_SOURCE_HASHES"] = namespace["current_sources"]
            (new_project / "src/added.py").write_text("VERSION=2\n", encoding="utf-8")
            previous_helper = sys.modules["src.helper"]
            exec(compile(prefix, "setup_changed_source", "exec"), namespace)
            assert importlib.import_module("src.helper") is not previous_helper
        bootstrap = helpers["bootstrap_project"]
        with patch.dict(bootstrap.__globals__, {"find_project": Mock(side_effect=FileNotFoundError()), "workspace": lambda: root}):
            restored = bootstrap(payload, digest)
        assert (restored / "src/resources.py").is_file()
        assert (restored / "requirements-notebook.txt").is_file()
        assert (restored / "data/ary/train.tsv").read_bytes() == (PROJECT / "data/ary/train.tsv").read_bytes()
        assert not (restored / "models").exists() and not (restored / "results").exists()
        # Deux exécutions restaurent les mêmes fichiers, sans écrasement.
        helpers["extract_bundle"](payload, digest, restored)
        altered = restored / "requirements.txt"
        altered.write_text("custom", encoding="utf-8")
        try:
            helpers["extract_bundle"](payload, digest, restored)
        except FileExistsError:
            assert altered.read_text(encoding="utf-8") == "custom"
        else:
            raise AssertionError("Un fichier utilisateur a été écrasé")
        # Refuser un ZIP dont un membre sort du dossier.
        buffer = io.BytesIO()
        with ZipFile(buffer, "w") as archive:
            archive.writestr("../outside.py", "x")
        bad = buffer.getvalue()
        try:
            helpers["extract_bundle"](base64.b64encode(bad).decode(), hashlib.sha256(bad).hexdigest(), root / "bad")
        except ValueError:
            pass
        else:
            raise AssertionError("Traversal d'archive accepté")
        # La copie d'un montage en lecture seule ne reprend pas ses modèles/résultats.
        (restored / "models").mkdir()
        (restored / "models/unused.bin").write_bytes(b"do not copy")
        with patch.dict(bootstrap.__globals__, {"find_project": lambda _: restored, "workspace": lambda: root / "working", "writable_directory": lambda _: False}):
            staged = bootstrap(payload, digest)
        assert staged != restored and (staged / "src/resources.py").is_file()
        assert not (staged / "models").exists()
        # Un runtime contenant l'ancien profil ne doit pas être réutilisé silencieusement.
        stale = root / "stale_project"
        shutil.copytree(restored, stale)
        (stale / "requirements-notebook.txt").write_text("numpy<2\n", encoding="utf-8")
        with patch.dict(bootstrap.__globals__, {"find_project": lambda _: stale, "workspace": lambda: root / "new_workspace"}):
            updated = bootstrap(payload, digest)
        assert updated != stale
        assert (updated / "requirements-notebook.txt").read_bytes() == (PROJECT / "requirements-notebook.txt").read_bytes()
        # Dépendance incompatible : message clair en mode manuel, pip seulement simulé.
        fake = root / "dependency_project"
        fake.mkdir()
        (fake / "requirements-notebook.txt").write_text("numpy==0.0.1\n", encoding="utf-8")
        dependencies = helpers["ensure_dependencies"]
        original_import = builtins.__import__
        def without_packaging(name, *args, **kwargs):
            if name == "packaging.requirements":
                raise ImportError("packaging absent")
            return original_import(name, *args, **kwargs)
        Version = namedtuple("Version", "major minor micro releaselevel serial")
        runtime = SimpleNamespace(version_info=Version(3, 12, 7, "final", 0), executable=sys.executable, modules={})
        with patch.dict(dependencies.__globals__, {"sys": runtime}), patch("builtins.__import__", side_effect=without_packaging), patch("subprocess.check_call", side_effect=subprocess.CalledProcessError(1, "pip")) as installer:
            try:
                dependencies(fake)
            except RuntimeError as error:
                assert "Installation de packaging impossible" in str(error)
            else:
                raise AssertionError("Échec d'installation de packaging masqué")
            installer.assert_called_once()
            # Ne pas installer de dépendance si le projet est incomplet.
            try:
                dependencies(root / "absent")
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("Projet incomplet accepté")
            installer.assert_called_once()
        versions = {"numpy": "2.2.6", "scipy": "1.15.3", "pandas": "2.2.3", "scikit-learn": "1.6.1",
                    "torch": "2.8.0", "transformers": "4.48.0", "gensim": "4.4.0", "sentencepiece": "0.2.1",
                    "protobuf": "5.29.5", "matplotlib": "3.10.3", "nbformat": "5.9.2", "psutil": "7.0.0",
                    "ipython": "8.30.0", "packaging": "25.0"}
        for minor, numpy in ((11, "1.26.4"), (12, "2.2.6"), (13, "2.2.6")):
            profile = dict(versions)
            profile["numpy"] = numpy
            if minor == 11:
                profile.update(scipy="1.11.1", sentencepiece="0.2.0", matplotlib="3.7.2", psutil="5.9.0", ipython="8.15.0")
            runtime = SimpleNamespace(version_info=Version(3, minor, 7, "final", 0), executable=sys.executable, modules={})
            with patch.dict(dependencies.__globals__, {"sys": runtime}), patch("importlib.metadata.version", side_effect=lambda name: profile[name]), patch("subprocess.check_call") as installer:
                dependencies(PROJECT, install=False)
                installer.assert_not_called()
        # Conserver les bibliothèques compatibles du runtime Colab :
        # ni rétrogradation des versions récentes, ni mise à jour d'IPython 7.
        for minor, ipython in ((11, "7.34.0"), (13, "9.6.0"), (13, "10.0.0")):
            profile = dict(versions, ipython=ipython, matplotlib="3.10.7", psutil="8.0.0", protobuf="6.33.0", pandas="2.3.3", sentencepiece="0.2.0")
            runtime = SimpleNamespace(version_info=Version(3, minor, 7, "final", 0), executable=sys.executable, modules={"IPython": object(), "matplotlib": object(), "psutil": object()})
            with patch.dict(dependencies.__globals__, {"sys": runtime}), patch("importlib.metadata.version", side_effect=lambda name: profile[name]), patch("subprocess.check_call") as installer:
                dependencies(PROJECT)
                installer.assert_not_called()
        # Exécuter les cinq cellules de démarrage dans l'ordre, sans pip ni entraînement.
        standalone = root / "standalone"
        standalone.mkdir()
        namespace = {}
        with patch.object(Path, "cwd", return_value=standalone), patch.dict(os.environ, {"DARIJA_PROJECT_DIR": ""}), patch("importlib.metadata.version", side_effect=lambda name: versions[name]), patch("subprocess.check_call") as installer:
            for i, source in enumerate(startup, 1):
                exec(compile(source, f"startup_{i}", "exec"), namespace)
                if i < 6:
                    installer.assert_not_called()
            first_project = namespace["PROJECT"]
            for i, source in enumerate(startup, 1):
                exec(compile(source, f"startup_{i}", "exec"), namespace)
            assert namespace["PROJECT"] == first_project
            assert namespace["OUTPUT"].is_dir()
            assert (first_project / "src/campaign.py").is_file()
            installer.assert_not_called()
        def installed_report(command, changed):
            Path(command[command.index("--report") + 1]).write_text(
                json.dumps({"install": [{"metadata": {"name": name}} for name in changed]}), encoding="utf-8")

        runtime = SimpleNamespace(version_info=Version(3, 12, 7, "final", 0), executable=sys.executable, modules={"numpy": object()})
        with patch.dict(dependencies.__globals__, {"sys": runtime}), patch("importlib.metadata.packages_distributions", return_value={"numpy": ["numpy"]}), patch("subprocess.check_call", side_effect=lambda command: installed_report(command, ["numpy"])) as installer:
            try:
                dependencies(fake, install=False)
            except RuntimeError as error:
                assert "Dépendances" in str(error)
            else:
                raise AssertionError("Dépendance incompatible acceptée")
            installer.assert_not_called()
            try:
                dependencies(fake, install=True)
            except RuntimeError as error:
                assert "Redémarrer" in str(error)
            else:
                raise AssertionError("Remplacement d'un module chargé accepté")
            installer.assert_called_once()
            # Une deuxième exécution ne doit pas masquer le besoin de redémarrage.
            with patch("importlib.metadata.version", return_value="0.0.1"):
                try:
                    dependencies(fake)
                except RuntimeError as error:
                    assert "Redémarrer" in str(error)
                else:
                    raise AssertionError("Kernel périmé réutilisé")
            installer.assert_called_once()
        # NumPy chargé mais inchangé : installer Gensim seul ne requiert pas de redémarrage.
        (fake / "requirements-notebook.txt").write_text("gensim==4.4.0\n", encoding="utf-8")
        runtime = SimpleNamespace(version_info=Version(3, 12, 7, "final", 0), executable=sys.executable, modules={"numpy": object()})
        with patch.dict(dependencies.__globals__, {"sys": runtime}), patch("importlib.metadata.version", return_value="4.3.0"), patch("importlib.metadata.packages_distributions", return_value={"numpy": ["numpy"]}), patch("subprocess.check_call", side_effect=lambda command: installed_report(command, ["gensim"])) as installer:
            dependencies(fake)
            assert not getattr(runtime, "_darija_restart_required", False)
            installer.assert_called_once()
        # Inclure les remplacements transitifs et les noms d'import distincts.
        runtime.modules = {"sklearn.utils": object()}
        with patch.dict(dependencies.__globals__, {"sys": runtime}), patch("importlib.metadata.version", return_value="4.3.0"), patch("importlib.metadata.packages_distributions", return_value={"sklearn": ["scikit-learn"]}), patch("subprocess.check_call", side_effect=lambda command: installed_report(command, ["gensim", "scikit_learn"])):
            try:
                dependencies(fake)
            except RuntimeError as error:
                assert "scikit-learn" in str(error)
            else:
                raise AssertionError("Remplacement transitif chargé accepté")
        runtime = SimpleNamespace(version_info=Version(3, 12, 7, "final", 0), executable=sys.executable, modules={})
        with patch.dict(dependencies.__globals__, {"sys": runtime}), patch("subprocess.check_call", side_effect=subprocess.CalledProcessError(1, "pip")):
            try:
                dependencies(fake, install=True)
            except RuntimeError as error:
                assert "Installation incomplète" in str(error)
            else:
                raise AssertionError("Échec pip masqué")
        # Téléchargements artificiels : hash, publication atomique, aucun fichier réseau.
        class Response(io.BytesIO):
            def __init__(self, data):
                super().__init__(data)
                self.headers = {"Content-Length": str(len(data))}
        data = b"verified contents"
        target = root / "download.bin"
        with patch("urllib.request.urlopen", return_value=Response(data)):
            resources["download_atomic"]("https://example.invalid/file", target, hashlib.sha256(data).hexdigest())
        assert target.read_bytes() == data
        with patch("urllib.request.urlopen", return_value=Response(b"corrupt")):
            try:
                resources["download_atomic"]("https://example.invalid/file", target, hashlib.sha256(data).hexdigest())
            except ValueError:
                pass
            else:
                raise AssertionError("Empreinte invalide acceptée")
        assert target.read_bytes() == data and not target.with_suffix(".bin.part").exists()
        with patch("urllib.request.urlopen", side_effect=[TimeoutError(), Response(data)]), patch("time.sleep"):
            resources["download_atomic"]("https://example.invalid/file", target)
        # AraVec : petits fichiers factices, validation et extraction seulement ; jamais chargés comme modèles.
        name = "full_uni_cbow_100_twitter.mdl"
        contents = {name: b"fake model", name + ".wv.vectors.npy": b"fake vectors", name + ".trainables.syn1neg.npy": b"fake trainables"}
        hashes = {n: hashlib.sha256(v).hexdigest() for n, v in contents.items()}
        aravec = root / "aravec" / name
        ensure = resources["ensure_aravec"]
        def fake_download(url, path):
            with ZipFile(path, "w") as archive:
                for n, content in contents.items():
                    archive.writestr("subdirectory/" + n, content)
        with patch.dict(ensure.__globals__, {"ARAVEC_FILES": hashes, "download_atomic": fake_download}):
            try:
                ensure(aravec, allow_download=False)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("AraVec incomplet accepté")
            ensure(aravec, allow_download=True)
            assert all((aravec.parent / n).read_bytes() == content for n, content in contents.items())
    print("OK : notebook seul, copie inscriptible, ZIP sûr, dépendances et ressources ; réseau/pip entièrement simulés.")


if __name__ == "__main__":
    main()
