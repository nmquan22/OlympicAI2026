import numpy as np


def normalize(rows):
    return rows / np.maximum(np.linalg.norm(rows, axis=-1, keepdims=True), 1e-12)


def recall_at_k(scores, query_groups, gallery_groups, k):
    """Any matching group in the top k counts; supports several captions per image."""
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    hits = np.asarray(gallery_groups)[order] == np.asarray(query_groups)[:, None]
    return float(hits.any(axis=1).mean())


def prototypes(support, labels, query):
    classes = np.unique(labels)
    centers = np.stack([support[labels == label].mean(0) for label in classes])
    return classes[(normalize(query) @ normalize(centers).T).argmax(1)]
