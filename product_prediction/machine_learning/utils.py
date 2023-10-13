from typing import List, Optional

import attrs
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@attrs.define
class Metrics:
    f1_score: float
    accuracy_score: float
    precision_score: float
    recall_score: float
    confusion_matrix: np.ndarray

    @classmethod
    def compute(cls, probs: np.ndarray, targets: np.ndarray) -> "Metrics":
        preds = np.argmax(probs, axis=1)
        return cls.compute_w_preds(preds, targets)

    @classmethod
    def compute_w_preds(cls, preds: np.ndarray, targets: np.ndarray) -> "Metrics":
        return cls(
            f1_score(targets, preds, average="macro"),
            balanced_accuracy_score(targets, preds),
            precision_score(targets, preds, average="macro"),
            recall_score(targets, preds, average="macro"),
            confusion_matrix(targets, preds),
        )

    def plot_confusion_matrix(
        self, categories: Optional[List[str]] = None
    ) -> go.Figure:
        if categories is None:
            categories = [str(i) for i in range(self.confusion_matrix.shape[0])]

        # Create confusion matrix
        fig = ff.create_annotated_heatmap(
            z=self.confusion_matrix,
            x=categories,
            y=categories,
            annotation_text=[[str(n) for n in row] for row in self.confusion_matrix],
        )
        fig.update_layout(
            {
                "xaxis": {"title": "Predicted category"},
                "yaxis": {"title": "True category"},
            }
        )

        return fig
