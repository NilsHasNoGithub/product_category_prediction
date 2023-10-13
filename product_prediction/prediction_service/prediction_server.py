import io
from typing import Union

import PIL.Image as pil_img
from sentence_transformers import SentenceTransformer
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

from ..machine_learning.vision_model import ClassifierModule
from . import shoe_category_prediction_pb2, shoe_category_prediction_pb2_grpc


class PredictionServer(shoe_category_prediction_pb2_grpc.CategoryPredictionServicer):
    def __init__(
        self,
        vision_model: ClassifierModule,
        sentence_transformer: SentenceTransformer,
        text_classifier_model: Union[RandomForestClassifier, SVC, LinearRegression],
        label_encoder: LabelEncoder,
    ) -> None:
        """Initializes grpc server for generating category predictions based on text or images

        ## Parameters:
        - `vision_model` (`ClassifierModule`): Model which makes prediction based on images
        - `sentence_transformer` (`SentenceTransformer`): Model which generates embeddings for text using the `encode` function
        - `text_classifier_model` (`Union[RandomForestClassifier, SVC, LinearRegression]`): Model to predict category based on sentence embedding
        - `label_encoder` (`LabelEncoder`): Label encoder used to decode integer labels to categories

        """
        super().__init__()

        vision_model.eval()

        self._vision_model = vision_model
        self._img_transform = vision_model.get_transform()
        self._sentence_transformer = sentence_transformer
        self._text_classifier = text_classifier_model
        self._label_encoder = label_encoder

    def _predict_text(self, text: str) -> str:
        """Predict the category based on description text"""

        # create embedding
        embedding = self._sentence_transformer.encode(text)

        # make prediction
        pred = self._text_classifier.predict(embedding[None, ...])

        return self._label_encoder.inverse_transform(pred)[0]

    def _predict_image(self, img_bytes: bytes) -> str:
        """Predict category based on image"""

        # load and transform image
        img = pil_img.open(io.BytesIO(img_bytes))
        img = img.convert("RGB")
        img_t = self._img_transform(img)

        img_t = img_t.to(self._vision_model.device)
        pred = (
            self._vision_model.forward(img_t[None, ...]).detach().cpu().argmax(dim=-1)
        )

        return self._label_encoder.inverse_transform(pred.numpy())[0]

    def GetImagePrediction(self, request, context):
        result = self._predict_image(request.image)
        return shoe_category_prediction_pb2.PredictionReply(prediction=result)

    def GetTextPrediction(self, request, context):
        result = self._predict_text(request.text)
        return shoe_category_prediction_pb2.PredictionReply(prediction=result)
