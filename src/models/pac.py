
from typing import List

import warnings
import numpy as np

from joblibspark import register_spark

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from pyspark.sql.dataframe import DataFrame

warnings.filterwarnings('ignore')
register_spark()

class PACModel:
    def __init__(self):
        self.model = PassiveAggressiveClassifier(
            max_iter=2,          # số vòng lặp trên mỗi batch
            warm_start=True,     # giữ lại trạng thái mô hình qua các batch
            early_stopping=False
        )
        self.is_fitted = False  # chỉ truyền classes trong lần fit đầu tiên

    def train(self, df: DataFrame) -> List:
        X = np.array(df.select("image").collect()).reshape(-1, 3072)
        y = np.array(df.select("label").collect()).reshape(-1)

        if not self.is_fitted:
            self.model.partial_fit(X, y, classes=np.arange(10))
            self.is_fitted = True
        else:
            self.model.partial_fit(X, y)

        predictions = self.model.predict(X)

        accuracy = self.model.score(X, y)
        precision = precision_score(y, predictions, labels=np.arange(0, 10), average="macro")
        recall = recall_score(y, predictions, labels=np.arange(0, 10), average="macro")
        f1 = 2 * precision * recall / (precision + recall)

        return predictions, accuracy, precision, recall, f1

    def predict(self, df: DataFrame) -> List:
        X = np.array(df.select("image").collect()).reshape(-1, 3072)
        y = np.array(df.select("label").collect()).reshape(-1)

        predictions = self.model.predict(X)

        accuracy = self.model.score(X, y)
        precision = precision_score(y, predictions, labels=np.arange(0, 10), average="macro")
        recall = recall_score(y, predictions, labels=np.arange(0, 10), average="macro")
        f1 = 2 * precision * recall / (precision + recall)
        cm = confusion_matrix(y, predictions)

        return predictions, accuracy, precision, recall, f1, cm
