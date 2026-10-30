"""Export offline S6 reference previews from measured T4 homographies."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np

from tools.s6_reference.reference import render


def export(campaign: Path, destination: Path) -> dict:
    audit_path = campaign / 'sweep/canonical-sweep-audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    runs_path = campaign / 'sweep/runs.csv'
    assert hashlib.sha256(runs_path.read_bytes()).hexdigest() == audit['inputs_sha256']['runs']
    manifest_path = campaign / 'images/pairs.json'
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == audit['inputs_sha256']['manifest']
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    with runs_path.open(encoding='utf-8-sig', newline='') as handle:
        records = {(r['pair_id'], r['config_id']): r for r in csv.DictReader(handle)
                   if r['is_warmup'] == '0' and r['repeat_idx'] == '0'}
    oracle = {r['pair_id']: r['config_id'] for r in audit['oracle']}
    destination.mkdir(parents=True, exist_ok=False)
    entries, sections = [], []
    for pair, paths in manifest.items():
        folder = destination / pair
        folder.mkdir()
        images = []
        for side in ('a', 'b'):
            image = cv2.imread(str(manifest_path.parent / paths[side]))
            if image is None:
                raise ValueError(f'cannot decode {pair}/{side}')
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            assert cv2.imwrite(str(folder / f'input_{side}.png'), image)
        cells = [f'<figure><a href="{pair}/input_{s}.png"><img loading="lazy" src="{pair}/input_{s}.png"></a><figcaption>Input {s.upper()}</figcaption></figure>' for s in ('a', 'b')]
        entry = {'pair_id': pair, 'outputs': []}
        for label, config in (('Reference', 'S1.00_K2000_R1024_Bfeather'),
                              ('Fixed-Best', audit['fixed_best']['config_id']),
                              ('Oracle', oracle[pair])):
            output = {'label': label, 'config_id': config, 'repeat_idx': 0}
            caption = f'{label}: {config or "no canonical feasible configuration"}'
            picture = ''
            if config is None:
                output['status'] = 'no_canonical_feasible_configuration'
            else:
                row = records[pair, config]
                output.update(status=row['status'], fail_stage=row['fail_stage'])
                if row['status'] == 'OK' and not row['fail_stage']:
                    h = np.array([float(row[f'h{i}']) for i in range(9)]).reshape(3, 3)
                    output['H_B_to_A'] = h.tolist()
                    try:
                        result = render(*images, h, mode='feather')
                        filename = label.lower() + '.png'
                        assert cv2.imwrite(str(folder / filename), cv2.cvtColor(result.panorama, cv2.COLOR_RGB2BGR))
                        output['preview'] = f'{pair}/{filename}'
                        output['preview_sha256'] = hashlib.sha256((folder / filename).read_bytes()).hexdigest()
                        picture = f'<a href="{pair}/{filename}"><img loading="lazy" src="{pair}/{filename}"></a>'
                    except ValueError as exc:
                        output['preview_error'] = str(exc)
                caption += f' · {row["status"]} {row["fail_stage"]}'
            if not picture:
                picture = '<p class="failure">Không có panorama hợp lệ</p>'
            cells.append(f'<figure>{picture}<figcaption>{html.escape(caption)}</figcaption></figure>')
            entry['outputs'].append(output)
        entries.append(entry)
        sections.append(f'<section id="{pair}"><h2>{pair}</h2><div class="grid">'+''.join(cells)+'</div></section>')
    overview = np.full((250 * ((len(entries) + 4) // 5), 320 * 5, 3), 24, dtype=np.uint8)
    for index, entry in enumerate(entries):
        x, y = (index % 5) * 320, (index // 5) * 250
        reference = entry['outputs'][0]
        filename = reference.get('preview')
        if filename:
            preview = cv2.imread(str(destination / filename))
            scale = min(310 / preview.shape[1], 210 / preview.shape[0])
            thumb = cv2.resize(preview, (max(1, round(preview.shape[1] * scale)), max(1, round(preview.shape[0] * scale))))
            overview[y:y+thumb.shape[0], x:x+thumb.shape[1]] = thumb
        else:
            cv2.putText(overview, 'REFERENCE FAILED', (x+8, y+100), cv2.FONT_HERSHEY_SIMPLEX, .55, (80, 100, 255), 1)
        cv2.putText(overview, entry['pair_id'], (x+5, y+235), cv2.FONT_HERSHEY_SIMPLEX, .48, (240, 240, 240), 1)
    assert cv2.imwrite(str(destination / 'overview.jpg'), overview)
    receipt = {'schema_version': 'udis-preview-v1', 'pairs': entries,
               'renderer': 'existing CPU FP64 S6 feather reference from T4 measured H, repeat 0; not GPU pixel output or a new timing run',
               'audit_sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest(),
               'runs_sha256': audit['inputs_sha256']['runs']}
    (destination / 'preview-manifest.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    page = '''<!doctype html><html lang="vi"><meta charset="utf-8"><title>UDIS · 25 mẫu development</title>
<style>body{font:16px system-ui;margin:24px;background:#101820;color:#eee}section{border-top:1px solid #52616d;padding:16px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}figure{margin:0;background:#1d2c38;padding:10px}img{width:100%;height:240px;object-fit:contain}figcaption{font-size:13px;overflow-wrap:anywhere}a{color:#8ddcff}.failure{height:210px;color:#ffb0a8}p{max-width:1000px}</style>
<h1>UDIS: 25 mẫu development</h1><p>Ảnh train thật; grid đã đo trên Colab T4. Bấm ảnh để xem PNG đầy đủ. Panorama dưới đây được dựng bằng S6 CPU FP64 reference có sẵn, dùng homography đo trên T4 (repeat 0), feather. Đây là preview để xem hình học; không phải pixel output GPU hoặc lượt đo latency mới.</p>
<p>Canonical gate toàn cohort chưa xác lập: 23/25 cặp có cấu hình khả thi; hai reference thất bại được giữ nguyên. Oracle là lựa chọn offline từ grid, chưa phải Controller adaptive.</p><p><a href="overview.jpg">Xem tổng quan 25 reference</a></p>'''
    (destination / 'index.html').write_text(page + ''.join(sections) + '</html>', encoding='utf-8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = export(args.campaign, args.output)
    print(json.dumps({'pairs': len(receipt['pairs']), 'previews': sum('preview' in o for p in receipt['pairs'] for o in p['outputs']), 'index': str(args.output / 'index.html')}))


if __name__ == '__main__':
    main()
