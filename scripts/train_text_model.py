"""Script to classify products based on the description"""
import tempfile
import typing
from pathlib import Path
from typing import Literal

import attrs
import click
import joblib
import mlflow
import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from product_prediction.definitions import RANDOM_STATE
from product_prediction.machine_learning.text import (
    is_only_whitespace,
    normalize_whitespace,
)
from product_prediction.machine_learning.utils import Metrics
from product_prediction.utils import map_opt

ClassifierType = Literal["svc", "logistic_regression", "random_forest"]


@click.command()
@click.option(
    "--data-file",
    "-d",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Parquet file containing the train data, output from `extract_intersport_products.py`",
)
@click.option(
    "--experiment-name",
    default="text_category_prediction",
    help="MlFlow experiment name to use",
)
@click.option(
    "--classifier-type",
    "-c",
    default="svc",
    type=click.Choice(typing.get_args(ClassifierType)),
    help="Classifier used to classify",
)
@click.option(
    "--sentence-transformer",
    default="distiluse-base-multilingual-cased-v1",
    type=str,
    help="Sentence transformer to use for features used by classifier, see: https://www.sbert.net/docs/pretrained_models.html, by default a model is used that is able to deal with dutch",
)
def main(
    data_file: Path,
    experiment_name: str,
    classifier_type: ClassifierType,
    sentence_transformer: str,
):
    # Initiate mlflow
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # log params
        mlflow.log_params(
            {
                "classifier_type": classifier_type,
                "sentence_transformer": sentence_transformer,
            }
        )

        # load data
        data = pl.read_parquet(data_file)

        # Normalize the whitespace in the descriptions
        data = data.with_columns(
            pl.Series(
                "description",
                [map_opt(normalize_whitespace, s) for s in data["description"]],
            )
        )

        # Collect descriptions and categories
        descriptions = []
        categories = []

        for row in data.iter_rows(named=True):
            desc = row["description"]
            if desc is None or is_only_whitespace(desc):
                continue

            descriptions.append(desc)
            categories.append(row["category"])

        # Create embeddings
        sentence_encoder = SentenceTransformer(sentence_transformer)

        embeddings: np.ndarray = sentence_encoder.encode(
            descriptions, show_progress_bar=True
        )

        # Encode labels
        label_encoder = LabelEncoder().fit(categories)
        labels: np.ndarray = label_encoder.transform(categories)

        # Do train test split
        train_feats, test_feats, train_labels, test_labels = train_test_split(
            embeddings, labels, stratify=labels, random_state=RANDOM_STATE
        )

        # Select model
        match classifier_type:
            case "svc":
                classifier = SVC(probability=True)
            case "logistic_regression":
                classifier = LogisticRegression()
            case "random_forest":
                classifier = RandomForestClassifier()
            case _:
                raise ValueError("Unsupported model")

        # Fit model
        classifier.fit(train_feats, train_labels)

        # Store model
        with tempfile.TemporaryDirectory() as d:
            model_path = Path(d, "model.joblib")
            joblib.dump(classifier, model_path)
            mlflow.log_artifact(str(model_path))

        # log model metrics, for train and test samples
        for features, labels, type_ in [
            (train_feats, train_labels, "train"),
            (test_feats, test_labels, "test"),
        ]:
            # Compute probabilities
            probs = classifier.predict_proba(features)

            # Compute metrics
            metrics = Metrics.compute(probs, labels)

            # Log metrics to mlflow
            for k, v in attrs.asdict(metrics).items():
                if k != "confusion_matrix":
                    mlflow.log_metric(f"{type_}_{k}", v)

            # Log confusion matrix to mlflow
            with tempfile.TemporaryDirectory() as d:
                fig = metrics.plot_confusion_matrix(list(label_encoder.classes_))

                fig_path = Path(d, f"{type_}_confusion_matrix.html")
                fig.write_html(fig_path)
                mlflow.log_artifact(str(fig_path))


if __name__ == "__main__":
    main()
