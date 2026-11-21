"""Explicit numeric/categorical column selection; preprocessing fits only on train."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def tabular_pipeline(numeric, categorical, task="classification", model="linear", seed=7):
    estimators = {
        ("classification", "linear"): LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        ("classification", "trees"): ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        ("regression", "linear"): Ridge(alpha=10.0, solver="lsqr"),
        ("regression", "trees"): ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=seed),
    }
    number = Pipeline([("fill", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
    category = Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                         ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])
    features = ColumnTransformer([("numeric", number, numeric), ("categorical", category, categorical)])
    return Pipeline([("features", features), ("model", estimators[task, model])])
