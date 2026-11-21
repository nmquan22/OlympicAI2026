"""Compile Python, validate curated registry, and check repository-local MD links."""
import ast
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    for path in ROOT.rglob('*.py'):
        if '.git' not in path.parts:
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    rows = json.loads((ROOT / 'catalog/workshops_2026.json').read_text(encoding='utf-8'))['entries']
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append('Duplicate catalog IDs')
    for row in rows:
        for path in row['blocks']:
            if not (ROOT / path).is_file():
                errors.append(f"{row['id']}: missing block {path}")
        if row['year'] != 2026 or row['priority'] not in ['A', 'B', 'C']:
            errors.append(f"{row['id']}: invalid year or priority")
    for path in ROOT.rglob('*.md'):
        for target in re.findall(r'\]\(([^\s)]+)\)', path.read_text(encoding='utf-8')):
            parsed = urlsplit(target)
            if parsed.scheme or not parsed.path:
                continue
            if not (path.parent / unquote(parsed.path)).exists():
                errors.append(f'{path.relative_to(ROOT)}: broken local link {target}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Python syntax, {len(rows)} catalog entries and Markdown local links: OK')


if __name__ == '__main__':
    main()
