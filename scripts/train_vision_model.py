import multiprocessing
import shutil
import tempfile
from pathlib import Path

import albumentations as A
import click
import pytorch_lightning as ptl
from loguru import logger as logu
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from product_prediction.machine_learning.data import (
    VisionProductCategoryDataset,
    create_albumentation_aug_fn,
)
from product_prediction.machine_learning.vision_model import ClassifierModule

AUG_TRANSFORM = A.Compose(
    [A.HueSaturationValue(p=0.2), A.RandomBrightnessContrast(p=0.2)]
)


# A lot more hyperparameters can be added if necessary
@click.command()
@click.option(
    "--timm-model",
    "-m",
    default="efficientnet_b1",
    type=str,
    help="Timm model to use as backbone",
)
@click.option(
    "--batch-size",
    "-b",
    default=8,
    type=int,
    help="Batch size to use during model training",
)
@click.option(
    "--num-epochs",
    "-n",
    default=25,
    type=int,
    help="Number of epochs to train the model for",
)
@click.option(
    "--exp-name",
    default="visual_category_prediction",
    type=str,
    help="MLFlow experiment name to use",
)
@click.option(
    "--dataset-dir",
    "-d",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--cache-dir",
    default=".cache",
    type=click.Path(file_okay=False, path_type=Path),
    help="Cache directory to store model checkpoints",
)
def main(
    timm_model: str,
    batch_size: int,
    num_epochs: int,
    exp_name: str,
    dataset_dir: Path,
    cache_dir: Path,
):
    cache_dir.mkdir(exist_ok=True, parents=True)

    # Load the dataset, from the output of the `create_ml_dataset.py` script
    dataset = VisionProductCategoryDataset(dataset_dir)

    # Load Classification model to train
    model = ClassifierModule(
        timm_model,
        dataset.num_classes(),
        categories=list(dataset.label_encoder().classes_),
    )

    # Set the appropriate image transformation for the model
    dataset.set_transform(model.get_transform())

    # Split dataset in train and test
    train_ds, test_ds = dataset.train_test_split()

    # Set augmentation on train dataset
    train_ds.dataset.set_augmentation(create_albumentation_aug_fn(AUG_TRANSFORM))

    # Create an MlFlowLogger instance to log script results
    logger = MLFlowLogger(experiment_name=exp_name)

    # Open temporary on-disk directory to store intermediate model
    with tempfile.TemporaryDirectory(dir=str(cache_dir)) as tmpdir:
        tmpdir = Path(tmpdir)

        # Define a callback to store the model with the lowest loss
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",  # As defined in `ClassifierModule`
            dirpath=tmpdir,
            mode="min",
        )

        # Create ptl trainer, this selects compute device automatically
        trainer = ptl.Trainer(
            max_epochs=num_epochs,
            logger=logger,
            callbacks=[checkpoint_callback],
            accelerator="auto",
            devices=1,
        )

        # Define dataloaders
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, num_workers=multiprocessing.cpu_count()
        )
        val_loader = DataLoader(
            test_ds, batch_size=batch_size, num_workers=multiprocessing.cpu_count()
        )

        # Fit model
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Check if best model was saved
        if Path(checkpoint_callback.best_model_path).is_dir():
            logu.error("Run did not exit succesfully")
            return

        # Test best model
        best_model = ClassifierModule.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )

        trainer.test(best_model, val_loader)
        # Results are stored in MLFlow

        new_best_model_path = Path(checkpoint_callback.best_model_path).with_name(
            "best_model.ckpt"
        )
        shutil.move(checkpoint_callback.best_model_path, str(new_best_model_path))

        # Copy the best model to the artifact directory
        logger.experiment.log_artifact(logger.run_id, new_best_model_path)


if __name__ == "__main__":
    main()
