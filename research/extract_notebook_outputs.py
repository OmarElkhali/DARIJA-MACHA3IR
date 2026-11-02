"""Lire les sorties sauvegardées uniquement ; ne jamais exécuter le notebook."""
import base64
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/TP2_Darija_colab_FG.ipynb'


class Tables(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables, self.table, self.row, self.cell = [], None, None, None
        self.section = None

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.table = {'header_rows': [], 'body_rows': []}
        elif tag in ('thead', 'tbody'):
            self.section = tag
        elif tag == 'tr' and self.table is not None:
            self.row = []
        elif tag in ('th', 'td') and self.row is not None:
            self.cell = {'tag': tag, 'text': '', **dict(attrs)}

    def handle_data(self, data):
        if self.cell is not None:
            self.cell['text'] += data

    def handle_endtag(self, tag):
        if tag in ('th', 'td') and self.cell is not None:
            self.cell['text'] = self.cell['text'].strip()
            self.row.append(self.cell)
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            self.table['header_rows' if self.section == 'thead' else 'body_rows'].append(self.row)
            self.row = None
        elif tag == 'table' and self.table is not None:
            self.tables.append(self.table)
            self.table = None


def text(value):
    return ''.join(value) if isinstance(value, list) else value


def main():
    raw = NOTEBOOK.read_bytes()
    nb = json.loads(raw)
    figures = ROOT / 'research/notebook_figures'
    figures.mkdir(exist_ok=True)
    snapshot = {'notebook': str(NOTEBOOK.relative_to(ROOT)), 'notebook_sha256': hashlib.sha256(raw).hexdigest(),
                'precision_policy': 'Displayed strings preserved exactly; no hidden/truncated rows or columns reconstructed.',
                'tables': [], 'figures': [], 'streams': [], 'errors': []}
    heading = ''
    for ci, cell in enumerate(nb['cells']):
        source = text(cell.get('source', ''))
        if cell['cell_type'] == 'markdown':
            heading = source
            continue
        for oi, output in enumerate(cell.get('outputs', [])):
            location = {'cell_index': ci, 'cell_number': ci + 1, 'output_index': oi,
                        'execution_count': cell.get('execution_count'), 'heading': heading, 'source': source}
            data = output.get('data', {})
            if 'text/html' in data:
                html = text(data['text/html'])
                parser = Tables()
                parser.feed(html)
                for ti, table in enumerate(parser.tables):
                    values = [c['text'] for row in table['header_rows'] + table['body_rows'] for c in row]
                    table.update(location, table_index=ti, columns=[c['text'] for c in table['header_rows'][-1]] if table['header_rows'] else [],
                                 has_ellipsis=any(v in ('...', '…') or v.endswith(('...', '…')) for v in values),
                                 reported_shapes=re.findall(r'(\d+)\s+rows?\s*[×x]\s*(\d+)\s+columns?', html),
                                 text_plain=text(data.get('text/plain', '')))
                    snapshot['tables'].append(table)
            if 'image/png' in data:
                content = base64.b64decode(text(data['image/png']))
                assert content.startswith(b'\x89PNG\r\n\x1a\n')
                target = figures / f'cell_{ci:03d}_output_{oi:03d}.png'
                target.write_bytes(content)
                snapshot['figures'].append({**location, 'path': str(target.relative_to(ROOT)),
                    'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
            if output.get('output_type') == 'stream':
                snapshot['streams'].append({**location, 'name': output.get('name'), 'text': text(output.get('text', ''))})
            if output.get('output_type') == 'error':
                snapshot['errors'].append({**location, 'ename': output.get('ename'), 'evalue': output.get('evalue')})
    assert snapshot['tables'], 'Aucun tableau HTML enregistré.'
    target = ROOT / 'research/notebook_tables_snapshot.json'
    target.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding='utf-8')
    (figures / 'manifest.json').write_text(json.dumps({'notebook_sha256': snapshot['notebook_sha256'],
        'figures': snapshot['figures']}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'tables': len(snapshot['tables']), 'figures': len(snapshot['figures']), 'errors': len(snapshot['errors'])}))
    for i, table in enumerate(snapshot['tables']):
        print(i, 'cell', table['cell_index'], 'output', table['output_index'], 'rows', len(table['body_rows']),
              'ellipsis', table['has_ellipsis'], 'columns', table['columns'])


if __name__ == '__main__':
    main()
