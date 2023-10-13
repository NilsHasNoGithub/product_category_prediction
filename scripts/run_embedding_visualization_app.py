from pathlib import Path

import click
import mlflow.artifacts
import numpy as np
import PIL.Image as pil_img
import torchvision.transforms as T
from embedding_inspector import run_embedding_inspection_app

from product_prediction.machine_learning.data import VisionProductCategoryDataset
from product_prediction.machine_learning.vision_model import ClassifierModule


@click.command()
@click.option(
    "--also-train-samples",
    "-t",
    is_flag=True,
    default=False,
    help="Also include train samples in visualization",
)
@click.option(
    "--vision-model-run-id",
    "-v",
    required=True,
    type=str,
    help="MlFlow run id of the image based prediction model to be used",
)
@click.option(
    "--dataset-dir",
    "-d",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to dataset directory (output of `create_ml_dataset.py`)",
)
@click.option(
    "--mlflow-tracking-uri",
    default="http://localhost:5000",
    type=str,
    help="Tracking URI of mlflow to retrieve models from",
)
@click.option(
    "--port", "-p", default=5050, type=int, help="Port on which to run the dash server"
)
def main(
    also_train_samples: bool,
    vision_model_run_id: str,
    dataset_dir: Path,
    port: int,
    mlflow_tracking_uri: str,
):
    # Load dataset
    dataset = VisionProductCategoryDataset(dataset_dir)

    # split between train and test
    train_ds, test_ds = dataset.train_test_split()

    # Define image loading procedure
    def load_img(idx: int) -> pil_img.Image:
        img_t, _ = dataset[idx]
        return T.ToPILImage()(img_t)

    # Load model from MlFlow
    vision_model_path = mlflow.artifacts.download_artifacts(
        artifact_path="best_model.ckpt",
        run_id=vision_model_run_id,
        tracking_uri=mlflow_tracking_uri,
    )
    vision_model = ClassifierModule.load_from_checkpoint(vision_model_path)

    img_transform = vision_model.get_transform()

    def load_embedding(idx: int) -> np.ndarray:
        img = load_img(idx)
        img = img_transform(img).to(vision_model.device)

        embed = vision_model.create_embedding(img[None, ...])[0, ...]
        return embed.detach().cpu().numpy()

    # Define ids
    if also_train_samples:
        unique_ids = range(len(dataset))
    else:
        unique_ids = list(test_ds.indices)

    # Get required labels
    labels = [
        dataset.label_encoder().inverse_transform([l])[0]
        for _, l in [dataset[i] for i in unique_ids]
    ]

    run_embedding_inspection_app(
        unique_ids,
        labels,
        load_embedding,
        load_img,
        dash_app_run_kwargs=dict(port=port),
    )


if __name__ == "__main__":
    main()
