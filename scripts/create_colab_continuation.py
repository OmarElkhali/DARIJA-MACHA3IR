"""Créer une reprise Colab F/G autonome sans modifier les sorties du notebook principal."""
from pathlib import Path
import copy
import nbformat

PROJECT = Path(__file__).resolve().parents[1]


def main():
    original = nbformat.read(PROJECT / 'notebooks/TP2_Darija.ipynb', as_version=4)
    cells = copy.deepcopy(original.cells[:13])
    cells[0].source = '# Reprise Colab T4 · évaluations F/G\n\nSauvegardes existantes obligatoires. Aucun entraînement lancé. Exécuter dans l’ordre ; arrêter à la première erreur.\n\nExclusion demandée avant F/G : `P_darijabert_without_both_s42`, dont les poids sont absents. Ses résultats dev sont conservés ; aucun score test ne lui est attribué. Déclaration dans `results/evaluation_exclusions.json` et dans le verrou de sélection.'
    cells[2].source = cells[2].source.replace('RUN_TRAINING = True', 'RUN_TRAINING = False')
    cells[2].source = cells[2].source.replace('/content/drive/MyDrive/campaign_v4',
        '/content/drive/.shortcut-targets-by-id/167fJzC1cDy4JHz-BSzRSbwg_gyr3XeOu/campaign_v4')
    # Valider le code et le protocole archivés ; seuls le matériel et les chemins changent.
    setup = (PROJECT / 'scripts/cpu_setup.py').read_text(encoding='utf-8')
    setup = setup[setup.index('from pathlib import Path'):]
    setup = setup.replace("PROJECT = Path(__file__).resolve().parents[1]\nOUTPUT = PROJECT / 'experiments/campaign_v4'", "assert Path('/content').is_dir(), 'Cette reprise est destinée à Colab.'")
    setup = setup.replace("device='cpu'", "device='cuda'")
    setup = setup.replace("DEVICE, ARAVEC_PATH = 'cpu'", "DEVICE, ARAVEC_PATH = 'cuda'")
    setup = setup.replace("DARIJA_EVAL_DEVICE='cpu'", "DARIJA_EVAL_DEVICE='cuda'")
    # Les scripts de continuation ne sont pas dans les anciennes archives intégrées.
    setup = setup.replace("exec(compile((PROJECT / 'scripts/continue_campaign.py').read_text(encoding='utf-8'),\n             str(PROJECT / 'scripts/continue_campaign.py'), 'exec'), globals())",
                          (PROJECT / 'scripts/continue_campaign.py').read_text(encoding='utf-8'))
    setup = setup.replace("exec(compile((PROJECT / 'scripts/reduce_remaining.py').read_text(encoding='utf-8'),\n             str(PROJECT / 'scripts/reduce_remaining.py'), 'exec'), globals())",
                          (PROJECT / 'scripts/reduce_remaining.py').read_text(encoding='utf-8'))
    setup = setup[:setup.index("provenance = ")] + "\nassert torch.cuda.is_available(), 'Un GPU Colab est requis.'\nprint('GPU actif :', torch.cuda.get_device_name(0))\nprint('Évaluations uniquement ; checkpoints conservés.')\n"
    setup += '\n' + (PROJECT / 'scripts/skip_missing_p.py').read_text(encoding='utf-8') + '\nskip_missing_p(globals())\n'
    cells += [nbformat.v4.new_markdown_cell('## Restaurer la configuration et vérifier les sauvegardes'),
              nbformat.v4.new_code_cell(setup),
              nbformat.v4.new_code_cell('status = campaign.status(CONFIG)\ndisplay(status)\nassert status.status.eq("exécuté").all(), "Sauvegardes incomplètes : arrêter ici."\nshow_training_times(globals())')]
    cache = (PROJECT / 'scripts/cache_pretrained.py').read_text(encoding='utf-8')
    cache = cache.replace('sys.path.insert(0, str(Path(__file__).resolve().parents[1]))', '')
    finalize = 'from src.resources import preflight\npreflight(CONFIG, check_models=False, download_aravec=True)\n'
    finalize += cache + '\nassert all(item["status"] == "cached" for item in status), "Ressources publiques incomplètes."\nfinal = campaign.finalize(CONFIG)\ndisplay(final)\n'
    cells += [nbformat.v4.new_markdown_cell('## Ressources publiques, verrouillage et évaluation finale'),
              nbformat.v4.new_code_cell(finalize)]
    start = next(i for i, c in enumerate(original.cells) if c.cell_type == 'code' and 'campaign.finalize(CONFIG)' in c.source)
    cells += copy.deepcopy(original.cells[start + 1:])
    for cell in cells:
        if cell.cell_type == 'code':
            cell.outputs, cell.execution_count = [], None
            compile(cell.source, '<colab-continuation>', 'exec')
    code = '\n'.join(c.source for c in cells if c.cell_type == 'code')
    assert 'campaign.run_part(' not in code
    assert 'D:\\\\SetupGame' not in code and 'scripts/cpu_setup.py' not in code
    assert 'runpy.run_path(str(PROJECT / "scripts/cache_pretrained.py")' not in code
    assert "if actual != expected:" in setup and "if json.loads((OUTPUT / 'campaign.json')" in setup
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
        'accelerator': 'GPU', 'colab': {'gpuType': 'T4'}})
    target = PROJECT / 'notebooks/TP2_Darija_colab_FG.ipynb'
    nbformat.write(notebook, target)
    print(target)
    print(f'{sum(c.cell_type == "code" for c in cells)} cellules compilées ; aucun entraînement exécuté.')


if __name__ == '__main__':
    main()
