"""Exercise real input-file branches with tiny generated files, not benchmark data."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]


def run(module, *arguments):
    subprocess.run([sys.executable, '-m', module, *map(str, arguments)], cwd=ROOT,
                   env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2'),
                   check=True, capture_output=True, text=True, timeout=60)


def csv_file(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class FileRecipeTests(unittest.TestCase):
    def test_tabular_unlabelled_csv_keeps_row_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for split, size in [('train', 20), ('valid', 8)]:
                rows = [{'id': f'{split}-{i}', 'amount': i % 4, 'kind': 'a' if i % 2 else 'b', 'target': i % 2} for i in range(size)]
                csv_file(base / f'{split}.csv', ['id', 'amount', 'kind', 'target'], rows)
            csv_file(base / 'test.csv', ['id', 'amount', 'kind'], [
                {'id': 'z', 'amount': 1, 'kind': 'new'}, {'id': 'a', 'amount': 2, 'kind': 'a'}])
            run('ml.tabular', '--train', base/'train.csv', '--valid', base/'valid.csv', '--predict', base/'test.csv',
                '--numeric', 'amount', '--categorical', 'kind', '--out', base/'out')
            with (base/'out/predictions.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([r['id'] for r in rows], ['z', 'a'])
            self.assertTrue((base/'out/model.joblib').exists())

    def test_audio_manifest_paths_and_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for split, size in [('train', 4), ('valid', 2)]:
                rows = []
                for i in range(size):
                    key = f'{split}-{i}'
                    wave = (12000 * np.sin(np.arange(1600) * 2 * np.pi * (300 + 600*(i % 2)) / 16000)).astype(np.int16)
                    wavfile.write(base/f'{key}.wav', 16000, wave)
                    rows.append({'id': key, 'path': f'{key}.wav', 'label': i % 2})
                csv_file(base/f'{split}.csv', ['id', 'path', 'label'], rows)
            run('audio.classify', '--train', base/'train.csv', '--valid', base/'valid.csv', '--epochs', 1, '--out', base/'out')
            self.assertTrue((base/'out/model.pt').is_file())
            with (base/'out/predictions.csv').open() as handle:
                self.assertEqual([r['id'] for r in csv.DictReader(handle)], ['valid-0', 'valid-1'])

    def test_restoration_pairs_and_scale(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for split in ['train', 'valid']:
                Image.fromarray(np.full((8, 8, 3), 100, dtype=np.uint8)).save(base/f'{split}-lr.png')
                Image.fromarray(np.full((16, 16, 3), 100, dtype=np.uint8)).save(base/f'{split}-hr.png')
                csv_file(base/f'{split}.csv', ['id', 'degraded', 'target'], [
                    {'id': split, 'degraded': f'{split}-lr.png', 'target': f'{split}-hr.png'}])
            run('vision.restore', '--train', base/'train.csv', '--valid', base/'valid.csv', '--scale', 2, '--epochs', 1, '--out', base/'out')
            with Image.open(base/'out/prediction_0000.png') as image:
                self.assertEqual(image.size, (16, 16))

    def test_boundary_labels_separate_and_inference_unlabelled(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            human = 'The rain fell. He left the house. '
            machine = 'Furthermore this is a comprehensive perspective. In conclusion this highlights the significance. '
            rows = [{'id': f'train-{i}', 'text': human + machine} for i in range(4)]
            labels = [{'id': r['id'], 'boundary_char_index': len(human)} for r in rows]
            valid = [{'id': 'unlabelled', 'text': human + machine}]
            for filename, data in [('train', rows), ('labels', labels), ('valid', valid)]:
                (base/f'{filename}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in data), encoding='utf-8')
            run('nlp.boundary', '--train', base/'train.jsonl', '--labels', base/'labels.jsonl', '--valid', base/'valid.jsonl', '--out', base/'out')
            result = json.loads((base/'out/answers.jsonl').read_text())
            self.assertEqual(result['id'], 'unlabelled')
            self.assertGreaterEqual(result['boundary_char_index'], 0)
            self.assertLessEqual(result['boundary_char_index'], len(valid[0]['text']))

    def test_order_cached_scores_export_rank_convention(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            scores = np.full((4, 4), -10.0)
            scores[2, 0] = scores[0, 3] = scores[3, 1] = 10
            np.save(base/'scores.npy', scores)
            run('nlp.order', '--scores', base/'scores.npy', '--prefix', 2, 0, '--dialogue-id', '17', '--out', base/'answers.json')
            self.assertEqual(json.loads((base/'answers.json').read_text()), {'17': [1, 3, 0, 2]})


if __name__ == '__main__':
    unittest.main()
