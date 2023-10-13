import tempfile
from os.path import join as pjoin
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import attrs
import PIL.Image as pil_img
import pytorch_lightning as ptl
import timm
import timm.optim
import torch
import torch.nn as nn
from attr import asdict
from pytorch_lightning.loggers import MLFlowLogger
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from ..definitions import OptKwargs
from .utils import Metrics

C = TypeVar("C", dict, list, Any)


def detach_tensors(collection: C, to_cpu: bool = True, to_numpy: bool = False) -> C:
    """Recursively detach tensors in a collection"""

    def operation(t: torch.Tensor) -> torch.Tensor:
        if to_cpu:
            t = t.cpu()
        t = t.detach()
        if to_numpy:
            t = t.numpy()
        return t

    if isinstance(collection, dict):
        return {k: detach_tensors(v) for k, v in collection.items()}
    elif isinstance(collection, list):
        return [detach_tensors(v) for v in collection]
    elif isinstance(collection, torch.Tensor):
        return operation(collection)
    else:
        return collection


class Head(nn.Module):
    """Simple head for the neural network"""

    def __init__(self, n_feats: int, n_classes: int, droprate: float) -> None:
        super().__init__()

        self._dropout = nn.Dropout(droprate)
        self._linear = nn.Linear(n_feats, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._dropout(x)
        x = self._linear(x)

        return x


class ClassifierModule(ptl.LightningModule):
    """Pytorch lightning wrapper around `timm` image model"""

    def __init__(
        self,
        timm_model: str,
        n_classes: int,
        optimizer: str = "AdamW",
        learning_rate: float = 10e-5,
        weight_decay: float = 10e-6,
        extra_model_params: OptKwargs = None,
        pretrained: bool = True,
        head_drop_rate: float = 0.5,
        categories: Optional[List[str]] = None,
    ) -> None:
        """Pytorch lightning wrapper around `timm` backbone, with classfication head

        ## Parameters:
        - `timm_model` (`str`): The name of the timm model used
        - `n_classes` (`int`): Number of classes
        - `optimizer` (`str`, optional): Optimizer used. Defaults to "AdamW".
        - `learning_rate` (`float`, optional): Learning rate used. Defaults to 10e-3.
        - `weight_decay` (`float`, optional): Weight decay used. Defaults to 10e-6.
        - `extra_model_params` (`OptKwargs`, optional): Extra nodel parameters used. Defaults to None.
        - `pretrained` (`bool`, optional): Whether to use a pretrained model. Defaults to True.
        - `head_drop_rate` (`float`, optional): Dropout rate for the classification head. Defaults to 0.5.
        - `categories` (`Optional[List[str]]`, optional): Categories to use for plotting confusion matrix. Defaults to None.

        """
        super().__init__()

        if extra_model_params is None:
            extra_model_params = dict()

        self._timm_model = timm_model
        self._n_classes = n_classes
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._extra_model_params = (
            extra_model_params if extra_model_params is not None else dict()
        )
        self._categories = categories

        self._model = timm.create_model(
            timm_model,
            pretrained=pretrained,
            num_classes=0,
            **extra_model_params,
        )

        self._head = Head(self._model.num_features, n_classes, head_drop_rate)

        self._loss_fn = nn.CrossEntropyLoss()

        self._train_outputs = []
        self._val_outputs = []
        self._test_outputs = []

        self.save_hyperparameters()

    def embedding_size(self) -> int:
        return self._model.num_features

    def get_transform(self) -> Callable[[pil_img.Image], torch.Tensor]:
        config = resolve_data_config({}, model=self._model)
        transform = create_transform(**config)

        return transform

    def configure_optimizers(self):
        optimizer = timm.optim.create_optimizer_v2(
            self.parameters(),
            opt=self._optimizer,
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )

        return optimizer

    def create_embedding(self, imgs: torch.Tensor) -> torch.Tensor:
        return self._model(imgs)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        embedding = self._model(imgs)
        predictions = self._head(embedding)

        return predictions

    def step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        logt: str,
        outputs_collection=Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        # Get image and label
        img, label = batch

        # Get embedding and prediction
        embedding = self._model(img)
        prediction = self._head(embedding)

        # Compute loss
        loss = self._loss_fn(prediction, label)

        self.log(f"{logt}_loss", loss)

        outputs = dict(
            loss=loss, embeddings=embedding, predictions=prediction, labels=label
        )

        # Store outputs for later analysis
        if outputs_collection is not None:
            outputs_collection.append(outputs)

        return outputs

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        return self.step(
            batch, batch_idx, "train", outputs_collection=self._train_outputs
        )

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        return self.step(batch, batch_idx, "val", outputs_collection=self._val_outputs)

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        return self.step(
            batch, batch_idx, "test", outputs_collection=self._test_outputs
        )

    def epoch_end(
        self,
        outputs: List[dict],
        type_: str,
        make_confusion_matrix: bool = False,
        do_log: bool = True,
    ) -> Metrics:
        probs = torch.cat([o["predictions"] for o in outputs]).detach().cpu().numpy()
        labels = torch.cat([o["labels"] for o in outputs]).detach().cpu().numpy()
        metrics: Metrics = Metrics.compute(probs, labels)

        # log all metrics that are single numbers
        for name, val in attrs.asdict(metrics).items():
            if name != "confusion_matrix" and do_log:
                self.log(f"{type_}_{name}", val)

        if make_confusion_matrix and isinstance(self.logger, MLFlowLogger) and do_log:
            with tempfile.TemporaryDirectory() as d:
                cm_plot = metrics.plot_confusion_matrix(categories=self._categories)
                fpath = pjoin(d, "confusion_matrix.html")
                cm_plot.write_html(fpath)
                self.logger.experiment.log_artifact(self.logger.run_id, fpath)

        return metrics

    def on_validation_epoch_end(self) -> None:
        self.epoch_end(self._val_outputs, "val")

    def on_validation_epoch_start(self) -> None:
        self._val_outputs.clear()

    def on_train_epoch_end(self) -> None:
        self.epoch_end(self._train_outputs, "train")

    def on_train_epoch_start(self) -> None:
        self._train_outputs.clear()

    def on_test_epoch_end(self) -> None:
        self.epoch_end(
            self._test_outputs,
            "test",
            make_confusion_matrix=True,
        )

    def on_test_epoch_start(self) -> None:
        self._test_outputs.clear()
