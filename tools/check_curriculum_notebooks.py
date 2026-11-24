"""Execute every curriculum notebook's code cells in order, using CPU fixtures."""

from contextlib import redirect_stdout
import gc
import io
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    directory = ROOT / "vision/curriculum/notebooks"
    paths = sorted(directory.glob("*.ipynb"))
    total = 0
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        namespace = {"__name__": "__main__"}
        count = 0
        output = io.StringIO()
        try:
            with redirect_stdout(output):
                for cell in notebook["cells"]:
                    if cell["cell_type"] != "code":
                        continue
                    count += 1
                    source = "".join(cell["source"])
                    exec(
                        compile(source, f"{path.name}:code-cell-{count}", "exec"),
                        namespace,
                    )
                    loss = namespace.get("loss")
                    if torch.is_tensor(loss) and not torch.isfinite(loss).all():
                        raise ValueError(f"Non-finite loss in code cell {count}")
        except Exception:
            print(output.getvalue(), file=sys.stderr)
            print(f"Failed {path.name}, code cell {count}", file=sys.stderr)
            raise
        print(f"{path.name}: {count} code cells OK", flush=True)
        total += count
        namespace.clear()
        gc.collect()
    print(f"{len(paths)} notebooks, {total} code cells: OK")


if __name__ == "__main__":
    main()
