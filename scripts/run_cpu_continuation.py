"""Exécuter seulement la continuation, en conservant toutes les anciennes sorties."""
from pathlib import Path
import argparse
import datetime
import shutil
import sys
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plots-only', action='store_true')
    args = parser.parse_args()
    path = PROJECT / 'notebooks/TP2_Darija.ipynb'
    backup = PROJECT.parent / 'backups' / ('cpu_execution_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    backup.mkdir(parents=True)
    for source in PROJECT.joinpath('notebooks').glob('TP2_Darija*.ipynb'):
        shutil.copy2(source, backup / source.name)
    notebook = nbformat.read(path, as_version=4)
    setup_index = next(i for i, c in enumerate(notebook.cells) if c.metadata.get('darija_continuation'))
    notebook.cells[setup_index].source = "from pathlib import Path\nimport runpy\nglobals().update(runpy.run_path(str(Path(" + repr(str(PROJECT)) + ") / 'scripts/cpu_setup.py')))"
    notebook.cells[66].source = notebook.cells[66].source.replace('"CDE"', '"CDEP"')
    notebook.metadata.kernelspec = {'name': 'darija-cpu', 'display_name': 'Darija Sentiment (CPU)', 'language': 'python'}
    client = NotebookClient(notebook, timeout=None, kernel_name='darija-cpu', resources={'metadata': {'path': str(PROJECT)}})

    def save():
        temporary = path.with_suffix('.cpu.tmp')
        nbformat.write(notebook, temporary)
        temporary.replace(path)
        templates = str(Path(sys.base_prefix) / 'share/jupyter/nbconvert/templates')
        exporter = HTMLExporter(extra_template_basedirs=[templates], extra_template_paths=[templates])
        html, _ = exporter.from_notebook_node(notebook)
        path.with_suffix('.html').write_text(html, encoding='utf-8')

    indices = [setup_index, 64, 66, 72]
    if not args.plots_only:
        indices += [i for i in range(74, len(notebook.cells)) if notebook.cells[i].cell_type == 'code']
    try:
        with client.setup_kernel():
            for index in indices:
                cell = notebook.cells[index]
                previous = list(cell.get('outputs', []))
                print(f'CPU : cellule {index + 1}/{len(notebook.cells)} : {cell.source.splitlines()[0]}', flush=True)
                try:
                    client.execute_cell(cell, index)
                finally:
                    # Conserver aussi les sorties des précédentes exécutions.
                    cell.outputs = previous + list(cell.get('outputs', []))
                    save()
    finally:
        save()
    print('Continuation CPU terminée. Notebook et HTML sauvegardés.', flush=True)


if __name__ == '__main__':
    main()
