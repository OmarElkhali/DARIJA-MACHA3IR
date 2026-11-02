"""Exécute le notebook et synchronise son HTML après une exécution réussie."""
from pathlib import Path
import json
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "notebooks/TP2_Darija.ipynb"
notebook = nbformat.read(path, as_version=4)
for index, cell in enumerate(notebook.cells):
    if cell.cell_type == "code":
        compile(cell.source, f"cell_{index}", "exec")
client = NotebookClient(notebook, timeout=None, kernel_name="darija-sentiment", resources={"metadata": {"path": str(ROOT)}})
try:
    client.execute()
finally:
    # En cas d'erreur, conserver la sortie réelle pour le diagnostic.
    nbformat.write(notebook, path)
code = [cell for cell in notebook.cells if cell.cell_type == "code"]
errors = [output for cell in code for output in cell.get("outputs", []) if output.output_type == "error"]
assert not errors, errors
executed = sum(cell.execution_count is not None for cell in code)
assert executed == len(code)
html, _ = HTMLExporter().from_notebook_node(notebook)
html_path = path.with_suffix(".html")
html_path.write_text(html, encoding="utf-8")
print(json.dumps({"code_cells": len(code), "executed": executed, "errors": len(errors), "html": str(html_path)}, indent=2))
