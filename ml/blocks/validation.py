"""Split at the unit that can leak: person, document, machine, scene or time."""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold


def grouped_holdout(groups, valid_fraction=0.2, seed=7):
    indices = np.arange(len(groups))
    splitter = GroupShuffleSplit(n_splits=1, test_size=valid_fraction, random_state=seed)
    return next(splitter.split(indices, groups=groups))


def stratified_folds(labels, folds=5, seed=7):
    return list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(len(labels)), labels))


def chronological_holdout(target_times, cutoff, gap=0):
    """Train target timestamps < cutoff-gap, validate >= cutoff; numeric times."""
    times = np.asarray(target_times)
    return np.flatnonzero(times < cutoff - gap), np.flatnonzero(times >= cutoff)


def worst_group_accuracy(labels, predictions, groups):
    labels, predictions, groups = map(np.asarray, (labels, predictions, groups))
    scores = {str(g): float((labels[groups == g] == predictions[groups == g]).mean()) for g in np.unique(groups)}
    return {"worst_group_accuracy": min(scores.values()), "groups": scores}
