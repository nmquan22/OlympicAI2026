"""Readers for whitespace-separated sentence files and CoNLL-U."""
from pathlib import Path


def read_conll(path, word_column=0, tag_column=-1):
    text = Path(path).read_text(encoding="utf-8").strip()
    sentences = []
    for paragraph in text.split("\n\n"):
        rows = [line.split() for line in paragraph.splitlines() if line.strip() and not line.startswith("-DOCSTART-")]
        if rows:
            sentences.append(([row[word_column] for row in rows], [row[tag_column] for row in rows]))
    return sentences


def read_conllu(path):
    """Integer IDs are words. Ranges represent multiword tokens, decimals empty nodes."""
    text = Path(path).read_text(encoding="utf-8").strip()
    sentences = []
    for paragraph in text.split("\n\n"):
        rows = [line.split("\t") for line in paragraph.splitlines() if not line.startswith("#")]
        sentences.append([(r[1], int(r[6]), r[7]) for r in rows if r[0].isdigit()])
    return sentences
