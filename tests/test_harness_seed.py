"""Exercise the real harness CLI with a multi-configuration grid."""
import csv
import json
import random
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    binary = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        random_source = random.Random(20261101)
        pixels = bytes(channel for _ in range(128*128)
                       for channel in (random_source.randrange(256),)*3)
        image = root/'input.ppm'
        image.write_bytes(b'P6\n128 128\n255\n'+pixels)
        configs = [dict(config_id=f'fixture_{r}', S=1., K=500, R=r, blend_mode='feather')
                   for r in [128, 256]]
        (root/'configs.json').write_text(json.dumps(configs))
        for mode, seed in [('fixed', 12345), ('oracle', 67890)]:
            output = root/(mode+'.csv')
            subprocess.run([str(binary), '--img-a', str(image), '--img-b', str(image),
                '--configs', str(root/'configs.json'), '--mode', mode, '--seed', str(seed),
                '--warmup', '0', '--repeats', '1', '--out', str(output)],check=True)
            with output.open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == 2
            assert {row['config_id'] for row in rows} == {'fixture_128', 'fixture_256'}
            assert all(row['seed'] == str(seed) for row in rows), (mode, rows)
            assert all(row['status'] == 'OK' for row in rows), (mode, rows)
            print('HARNESS_GRID_SEED_VERIFIED', mode, seed)


if __name__ == '__main__':
    main()
