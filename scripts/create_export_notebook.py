from pathlib import Path
import nbformat

target = Path(__file__).resolve().parents[1] / 'notebooks/EXPORT_P.ipynb'
source = '''from pathlib import Path
import shutil
p = Path('/content/drive/MyDrive/campaign_v4/models/P_darijabert_without_both_s42/weights.pt')
out = Path('/content/P_weights.pt')
shutil.copyfile(p, out)
print(out, out.stat().st_size)
'''
nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)]), target)
