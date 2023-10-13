from pathlib import Path

import click
import grpc
import polars as pl
from itsdangerous import base64_decode
from loguru import logger
from sklearn.calibration import LabelEncoder
from tqdm import tqdm

from product_prediction.machine_learning.text import is_only_whitespace
from product_prediction.machine_learning.utils import Metrics
from product_prediction.prediction_service import (
    shoe_category_prediction_pb2,
    shoe_category_prediction_pb2_grpc,
)
from product_prediction.utils import json_load


@click.command()
@click.option(
    "--data-file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to parquet datafile (output of `extract_intersport_products.py`)",
)
@click.option(
    "--dataset-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to dataset directory (output of `create_ml_dataset.py`)",
)
@click.option(
    "--prediction-server-uri",
    "-u",
    default="localhost:50051",
    type=str,
    help="URI to category prediction server",
)
def main(data_file: Path, dataset_dir: Path, prediction_server_uri: str):
    # Load parquet data
    data = pl.read_parquet(data_file)

    # Create labelencoder
    label_encoder = LabelEncoder().fit(data["category"])

    # Selects texts with labels to test the server
    texts_and_labels_to_try = [
        (text, cat)
        for text, cat in zip(data["description"], data["category"])
        if text is not None and not is_only_whitespace(text)
    ]

    # Select images with labels
    imgs_and_labels_to_try = []

    for data_file in dataset_dir.glob("**/*.json"):
        data = json_load(data_file)

        # images are sent over as bytes
        imgs_and_labels_to_try.append(
            (base64_decode(data["main_image_base64"]), data["category"])
        )

    text_preds = []
    text_labels = []

    img_preds = []
    img_labels = []

    # Obtain predictions over the grps channel
    with grpc.insecure_channel(prediction_server_uri) as chan:
        stub = shoe_category_prediction_pb2_grpc.CategoryPredictionStub(chan)

        for txt, label in tqdm(texts_and_labels_to_try):
            response = stub.GetTextPrediction(
                shoe_category_prediction_pb2.TextPredictionRequest(text=txt)
            )
            text_preds.append(response.prediction)
            text_labels.append(label)

        for img, label in tqdm(imgs_and_labels_to_try):
            response = stub.GetImagePrediction(
                shoe_category_prediction_pb2.ImagePredictionRequest(image=img)
            )
            img_preds.append(response.prediction)
            img_labels.append(label)

    # No performance measure, though this is useful to check whether predictitions make sense
    for preds, labels, type_ in [
        (text_preds, text_labels, "text"),
        (img_preds, img_labels, "image"),
    ]:
        metrics = Metrics.compute_w_preds(
            label_encoder.transform(preds), label_encoder.transform(labels)
        )

        logger.info(f"{type_} metrics:\n{metrics}")


if __name__ == "__main__":
    main()
