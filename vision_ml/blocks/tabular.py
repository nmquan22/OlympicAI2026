"""Fit on training rows. Split by site or product ID before fitting transformers."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def metadata_classifier(numeric, categorical):
    numbers = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categories = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
    prepare = ColumnTransformer([("numbers", numbers, numeric), ("categories", categories, categorical)])
    return make_pipeline(prepare, LogisticRegression(class_weight="balanced", max_iter=500))
