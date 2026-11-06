"""Convert the complete official UDIS-D test split to lossless P6 inputs."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2

from tools.controller.p3_io import sha256, write_json


def prepare(root: Path, output: Path, expected_pairs: int = 1106) -> dict:
    if output.exists():
        raise ValueError('output directory must be new')
    sides = [{p.stem: p for p in (root / side).glob('*.jpg')} for side in ('input1', 'input2')]
    if set(sides[0]) != set(sides[1]) or len(sides[0]) != expected_pairs:
        raise ValueError('UDIS-D source cohort is incomplete')
    pairs, hashes = {}, {}
    for pair in sorted(sides[0]):
        pairs[pair] = {}
        hashes[pair] = {}
        for index, side in enumerate(('a', 'b')):
            source = sides[index][pair]
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None or max(image.shape[:2]) > 2048:
                raise ValueError('unreadable or oversized input')
            destination = output / 'images' / f'{pair}_{side}.ppm'
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination), image):
                raise OSError('PPM conversion failed')
            pairs[pair][side] = destination.relative_to(output).as_posix()
            hashes[pair][side] = dict(source_sha256=sha256(source), ppm_sha256=sha256(destination),
                                     width=image.shape[1], height=image.shape[0])
    write_json(output / 'pairs.json', pairs)
    write_json(output / 'preparation.json', dict(pairs=len(pairs), images=hashes,
                                                manifest_sha256=sha256(output / 'pairs.json')))
    return dict(pairs=len(pairs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.root, args.output))
