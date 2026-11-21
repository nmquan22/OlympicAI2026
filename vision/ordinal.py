"""RetinaMNIST validation, using the 28-pixel NPZ from MedMNIST v2."""
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def features(images):
    # Average 4x4 neighborhoods; keep coarse spatial position and RGB channels.
    count, height, width, channels = images.shape
    pooled = images.reshape(count, height//4, 4, width//4, 4, channels).mean(axis=(2, 4))
    return pooled.reshape(count, -1)


def classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500))


def ordinal_probabilities(train, labels, valid):
    thresholds = range(4)
    cumulative = np.column_stack([
        classifier().fit(train, labels > cut).predict_proba(valid)[:, 1] for cut in thresholds
    ])
    # P(y>k) must decrease as k rises. Subtract successive survival probabilities.
    survival = np.minimum.accumulate(cumulative, axis=1)
    return -np.diff(survival, prepend=1.0, append=0.0, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz")
    args = parser.parse_args()
    with np.load(args.npz) as data:
        train, valid = features(data["train_images"]), features(data["val_images"])
        y_train, y_valid = data["train_labels"].ravel(), data["val_labels"].ravel()
    softmax = classifier().fit(train, y_train).predict_proba(valid)
    ordinal = ordinal_probabilities(train, y_train, valid)
    for name, probabilities in (("softmax", softmax), ("ordinal", ordinal)):
        pred = probabilities.argmax(1)
        print(name, {"accuracy": accuracy_score(y_valid, pred),
                     "auc_ovr_macro": roc_auc_score(y_valid, probabilities, multi_class="ovr"),
                     "mae": mean_absolute_error(y_valid, pred)})


if __name__ == "__main__":
    main()
