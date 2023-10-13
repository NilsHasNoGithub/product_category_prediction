import multiprocessing
from concurrent import futures
from pathlib import Path

import click
import grpc
import joblib
import mlflow
import mlflow.artifacts
import polars as pl
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.calibration import LabelEncoder

from product_prediction.machine_learning.vision_model import ClassifierModule
from product_prediction.prediction_service import shoe_category_prediction_pb2_grpc
from product_prediction.prediction_service.prediction_server import PredictionServer


@click.command()
@click.option(
    "--text-model-run-id",
    "-t",
    required=True,
    type=str,
    help="MlFlow run id of the text based prediction model to be used",
)
@click.option(
    "--vision-model-run-id",
    "-v",
    required=True,
    type=str,
    help="MlFlow run id of the image based prediction model to be used",
)
@click.option(
    "--data-file",
    "-d",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to parquet data file, output of `extract_intersport_products`",
)
@click.option(
    "--port", "-p", default=50051, type=int, help="Port on which to run the grpc server"
)
@click.option(
    "--mlflow-tracking-uri",
    default="http://localhost:5000",
    type=str,
    help="Tracking URI of mlflow to retrieve models from",
)
@click.option(
    "--server-workers",
    "-w",
    default=multiprocessing.cpu_count(),
    type=int,
    help="Maximum server workers",
)
def main(
    text_model_run_id: str,
    vision_model_run_id: str,
    data_file: Path,
    port: int,
    mlflow_tracking_uri: str,
    server_workers: int,
) -> None:
    # load data, mainly for categories
    data = pl.read_parquet(data_file)

    # Create labelencoder for categories
    label_encoder = LabelEncoder().fit(data["category"])

    logger.debug("Loading vision model...")
    # Load vision model from MLFlow
    vision_model_path = mlflow.artifacts.download_artifacts(
        artifact_path="best_model.ckpt",
        run_id=vision_model_run_id,
        tracking_uri=mlflow_tracking_uri,
    )
    vision_model = ClassifierModule.load_from_checkpoint(vision_model_path)

    logger.debug("Loading text models...")
    # Load sentence transformer and text prediction model
    sentence_transformer_type = mlflow.get_run(text_model_run_id).data.params[
        "sentence_transformer"
    ]
    sentence_transformer = SentenceTransformer(sentence_transformer_type)

    text_prediction_model_path = mlflow.artifacts.download_artifacts(
        artifact_path="model.joblib",
        run_id=text_model_run_id,
        tracking_uri=mlflow_tracking_uri,
    )
    text_classifier_model = joblib.load(text_prediction_model_path)

    # Create the grpc server
    servicer = PredictionServer(
        vision_model, sentence_transformer, text_classifier_model, label_encoder
    )

    logger.info("Starting grpc service")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=server_workers))
    shoe_category_prediction_pb2_grpc.add_CategoryPredictionServicer_to_server(
        servicer, server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    main()
