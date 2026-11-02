"""Amorçage stdlib du notebook, également incorporé dans son premier bloc."""
from pathlib import Path
import base64
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZipFile


def find_project(project_dir="", search_roots=None):
    required = ("src/preprocess.py", "src/train.py", "src/campaign.py", "src/resources.py", "requirements.txt", "requirements-notebook.txt")
    def valid(path):
        return all((path / name).is_file() for name in required)
    if project_dir:
        path = Path(project_dir).expanduser().resolve()
        if not valid(path):
            raise FileNotFoundError(f"PROJECT_DIR={path} ne contient pas le projet complet.")
        return path
    for base in [Path.cwd(), *Path.cwd().parents]:
        for path in (base, base / "Darija-Sentiment"):
            if valid(path):
                return path.resolve()
    roots = search_roots if search_roots is not None else [Path.cwd(), Path("/content"), Path("/content/drive/MyDrive"), Path("/kaggle/working"), Path("/kaggle/input")]
    matches = set()
    for root in map(Path, roots):
        if not root.is_dir():
            continue
        if valid(root):
            matches.add(root.resolve())
        for pattern in ("*/src/preprocess.py", "*/*/src/preprocess.py", "*/*/*/src/preprocess.py"):
            try:
                for marker in root.glob(pattern):
                    if valid(marker.parent.parent):
                        matches.add(marker.parent.parent.resolve())
            except PermissionError:
                continue
    if len(matches) == 1:
        return matches.pop()
    if matches:
        raise RuntimeError("Plusieurs projets trouvés ; renseigner PROJECT_DIR : " + ", ".join(map(str, sorted(matches))))
    raise FileNotFoundError("Projet introuvable : ce notebook seul doit restaurer son archive intégrée.")


def writable_directory(path):
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=path):
            pass
        return True
    except OSError:
        return False


def prepare_storage(campaign_name, output_dir="", use_drive=True, backup_previous=True):
    """Copier les campagnes Colab existantes AVANT pip ou redémarrage."""
    if Path(campaign_name).name != campaign_name or campaign_name in {"", ".", ".."} or "\\" in campaign_name:
        raise ValueError("Nom de campagne invalide.")
    if not (Path("/content").is_dir() and (os.environ.get("COLAB_RELEASE_TAG") or "google.colab" in sys.modules)):
        return output_dir
    if use_drive:
        from google.colab import drive
        if not Path("/content/drive/MyDrive").is_dir():
            drive.mount("/content/drive")  # L'utilisateur valide l'accès dans Colab.
        base = Path("/content/drive/MyDrive/Darija-Sentiment")
    elif output_dir:
        base = Path(output_dir).expanduser().resolve().parent.parent
    else:
        raise RuntimeError("Choisir USE_GOOGLE_DRIVE=True ou un OUTPUT_DIR persistant avant de poursuivre.")
    base.mkdir(parents=True, exist_ok=True)
    output = Path(output_dir).expanduser().resolve() if output_dir else base / "experiments" / campaign_name
    if backup_previous:
        import runpy
        from datetime import datetime, timezone
        archived = base / "archives"
        for project in Path("/content").glob("Darija-Sentiment-runtime-*"):
            if not any((project / "experiments").glob("*/campaign.json")):
                continue
            identity = hashlib.sha256(str(project).encode()).hexdigest()[:12]
            marker = archived / f"{identity}.json"
            # Une relance reprend la sauvegarde terminée uniquement si la source est inchangée.
            files = [p for p in (project / "experiments").rglob("*") if p.is_file()]
            fingerprint = {str(p.relative_to(project)): [p.stat().st_size, p.stat().st_mtime_ns] for p in files}
            if marker.exists() and json.loads(marker.read_text(encoding="utf-8"))["source_fingerprint"] == fingerprint:
                previous = json.loads(marker.read_text(encoding="utf-8"))["destination"]
                if not (Path(previous) / "snapshot_manifest.json").is_file():
                    raise FileNotFoundError(f"Sauvegarde précédente absente : {previous}")
                print("Sauvegarde déjà réalisée pour cette source inchangée :", marker)
                continue
            archived.mkdir(parents=True, exist_ok=True)
            destination = archived / f"{project.name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
            backup = runpy.run_path(str(PROJECT_BACKUP_SCRIPT))
            backup["snapshot"](project, destination, directories=("experiments", "src", "scripts"))
            marker.write_text(json.dumps({"source_fingerprint": fingerprint, "destination": str(destination)}), encoding="utf-8")
            print("Campagne Colab sauvegardée avec vérification SHA-256 :", destination)
    return str(output)


def workspace():
    for path in (Path("/kaggle/working"), Path("/content"), Path.cwd(), Path.home()):
        if path.is_dir() and writable_directory(path):
            return path
    raise PermissionError("Aucun dossier de travail inscriptible ; renseigner un espace de travail accessible.")


def extract_bundle(payload, digest, destination):
    """Archive incorporée vérifiée, sans écrasement de fichiers différents ni traversal."""
    content = base64.b64decode(payload, validate=True)
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("Archive incorporée endommagée : réimporter le notebook corrigé.")
    destination = Path(destination).resolve()
    with ZipFile(io.BytesIO(content)) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination) or "\\" in member.filename or ":" in member.filename or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Chemin interdit dans l'archive du notebook")
            if member.is_dir():
                continue
            data = archive.read(member)
            if target.exists():
                if target.read_bytes() != data:
                    raise FileExistsError(f"Fichier différent conservé : {target}. Choisir un autre dossier.")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.write_bytes(data)
            temporary.replace(target)
    return destination


def bootstrap_project(payload, digest, project_dir=""):
    preferred = workspace() / f"Darija-Sentiment-runtime-{digest[:10]}"
    try:
        project = find_project(str(preferred)) if not project_dir and preferred.is_dir() else find_project(project_dir)
    except (FileNotFoundError, RuntimeError):
        if project_dir:
            raise
        project = extract_bundle(payload, digest, preferred)
        print("Code et données restaurés depuis le notebook :", project)
    content = base64.b64decode(payload, validate=True)
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("Archive du notebook endommagée")
    with ZipFile(io.BytesIO(content)) as archive:
        same_version = all((project / name).is_file() and (project / name).read_bytes() == archive.read(name)
                           for name in archive.namelist() if name.startswith("src/") or name == "requirements-notebook.txt")
    if not same_version:
        if project_dir:
            raise RuntimeError("PROJECT_DIR contient une version différente du notebook. Laisser PROJECT_DIR vide pour restaurer la version intégrée, ou mettre à jour le projet complet.")
        project = extract_bundle(payload, digest, preferred)
        print("Ancien projet détecté ; nouvelle version restaurée séparément :", project)
    if writable_directory(project):
        return project
    # Kaggle/input et dossiers montés en lecture seule : ne copier ni poids ni résultats.
    files = [p for directory in ("src", "data") for p in (project / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files += [p for p in (project / "requirements.txt", project / "requirements-notebook.txt") if p.exists()]
    identity = hashlib.sha256(b"".join(hashlib.sha256(p.read_bytes()).digest() for p in sorted(files))).hexdigest()[:10]
    target = workspace() / f"Darija-Sentiment-runtime-{identity}"
    for source in files:
        destination = target / source.relative_to(project)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        elif destination.read_bytes() != source.read_bytes():
            raise FileExistsError(f"Copie de travail différente : {destination}")
    print("Projet en lecture seule copié vers :", target)
    return target


def ensure_dependencies(project, install=True):
    restart_message = "Dépendances installées ; un redémarrage est nécessaire pour les charger. Sur Colab : Exécution > Redémarrer la session, puis relancer la cellule 1. Réexécuter la cellule sans redémarrer ne suffit pas."
    if getattr(sys, "_darija_restart_required", False):
        raise RuntimeError(restart_message)
    if not (3, 10) <= sys.version_info[:2] <= (3, 13):
        raise RuntimeError(f"Python {sys.version_info.major}.{sys.version_info.minor} détecté. Utiliser Python 3.10–3.13 pour les profils de dépendances disponibles.")
    requirements = project / "requirements-notebook.txt"
    if not requirements.is_file():
        raise FileNotFoundError("requirements-notebook.txt absent : utiliser l'archive ou le notebook corrigé complet.")
    try:
        from packaging.requirements import Requirement
    except ImportError:
        if not install:
            raise RuntimeError("Installer packaging dans ce kernel avant de continuer.")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "packaging>=23"])
        except subprocess.CalledProcessError as error:
            raise RuntimeError("Installation de packaging impossible. Vérifier Internet et pip, puis relancer la première cellule.") from error
        importlib.invalidate_caches()
        from packaging.requirements import Requirement
    missing = []
    for line in requirements.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        requirement = Requirement(line)
        marker_environment = {"python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                              "python_full_version": ".".join(map(str, sys.version_info[:3]))}
        if requirement.marker is not None and not requirement.marker.evaluate(marker_environment):
            continue
        try:
            version = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        if version is None or not requirement.specifier.contains(version, prereleases=False):
            missing.append(requirement.name)
    if not missing:
        return
    command = [sys.executable, "-m", "pip", "install", "--prefer-binary", "-r", str(requirements)]
    if not install:
        raise RuntimeError("Dépendances absentes/incompatibles : " + ", ".join(missing) + ". Activer INSTALL_DEPENDENCIES ou exécuter : " + subprocess.list2cmdline(command))
    # Inclure les dépendances transitives et les noms d'import différents (sklearn, PIL...).
    loaded_modules = {name.partition(".")[0] for name in sys.modules}
    loaded_distributions = {
        name.lower().replace("_", "-")
        for module, names in importlib.metadata.packages_distributions().items()
        if module in loaded_modules
        for name in names
    }
    print("Préparation des dépendances dans", sys.executable, ":", ", ".join(missing))
    with tempfile.TemporaryDirectory(prefix="darija_pip_") as temporary:
        report = Path(temporary) / "report.json"
        try:
            subprocess.check_call([*command, "--report", str(report)])
        except subprocess.CalledProcessError as error:
            # pip peut avoir remplacé des fichiers avant d'échouer.
            sys._darija_restart_required = True
            raise RuntimeError("Installation incomplète. Vérifier Internet et l'espace disque, redémarrer le kernel puis relancer la première cellule. Commande : " + subprocess.list2cmdline(command)) from error
        changed = {item["metadata"]["name"].lower().replace("_", "-")
                   for item in json.loads(report.read_text(encoding="utf-8"))["install"]}
    importlib.invalidate_caches()
    if changed & loaded_distributions:
        sys._darija_restart_required = True
        raise RuntimeError(restart_message + " Bibliothèques remplacées déjà chargées : " + ", ".join(sorted(changed & loaded_distributions)))
