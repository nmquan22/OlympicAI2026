"""Sparse retrieval and rank fusion; scores from different models need calibration."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class SparseRetriever:
    def fit(self, documents):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.documents = self.vectorizer.fit_transform(documents)
        return self

    def search(self, queries, k=10):
        scores = (self.vectorizer.transform(queries) @ self.documents.T).toarray()
        ranks = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        return ranks, np.take_along_axis(scores, ranks, axis=1)


def reciprocal_rank_fusion(rankings, constant=60, k=10):
    """Each input is a ranked list of document IDs for ONE query."""
    scores = {}
    for ranking in rankings:
        for rank, item in enumerate(dict.fromkeys(ranking), start=1):
            scores[item] = scores.get(item, 0) + 1 / (constant + rank)
    return sorted(scores, key=scores.get, reverse=True)[:k]


def ndcg_at_k(ranking, relevance, k=10):
    values = np.asarray([relevance.get(item, 0) for item in ranking[:k]])
    ideal = np.asarray(sorted(relevance.values(), reverse=True)[:k])
    dcg = lambda v: ((2.0**v - 1) / np.log2(np.arange(len(v)) + 2)).sum()
    return float(dcg(values) / max(dcg(ideal), 1e-12))
