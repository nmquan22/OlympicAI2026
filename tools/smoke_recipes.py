"""Run offline fixture entry points in separate processes; no benchmark claims."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    modules = ['ml.tabular', 'ml.forecast', 'ml.q_learning', 'nlp.order', 'nlp.boundary',
               'audio.classify', 'vision.restore', 'nlp.train', 'vision.detect', 'vision.segment',
               'vision_ml.retrieve', 'vision_ml.anomaly']
    with tempfile.TemporaryDirectory() as temporary:
        for module in modules:
            command = [sys.executable, '-m', module]
            if module != 'ml.q_learning':
                destination = Path(temporary) / module
                if module == 'nlp.order':
                    destination /= 'answers.json'
                command += ['--out', str(destination)]
            print(f'Running {module}', flush=True)
            subprocess.run(command, cwd=root, env=environment, check=True, timeout=120)


if __name__ == '__main__':
    main()
