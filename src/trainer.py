import pyspark

from pyspark.context import SparkContext
from pyspark.streaming.context import StreamingContext
from pyspark.sql.context import SQLContext
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import IntegerType, StructField, StructType
from pyspark.ml.linalg import VectorUDT


class SparkConfig:
    appName = "CIFAR"
    receivers = 4
    host = "local"
    stream_host = "localhost"
    port = 6100
    batch_interval = 2

from dataloader import DataLoader

class Trainer:
    def __init__(self, 
                 model, 
                 split:str, 
                 spark_config:SparkConfig
                 ) -> None:

        self.model = model
        self.split = split
        self.sparkConf = spark_config
        #self.transforms = transforms
        self.sc = SparkContext(f"{self.sparkConf.host}[{self.sparkConf.receivers}]",f"{self.sparkConf.appName}")
        self.ssc = StreamingContext(self.sc,self.sparkConf.batch_interval)
        self.sqlContext = SQLContext(self.sc)
        self.dataloader = DataLoader(self.sc,self.ssc,self.sqlContext,self.sparkConf)
        self.sc.setLogLevel("ERROR")
    def train(self):
        stream = self.dataloader.parse_stream()
        stream.foreachRDD(self.__train__)

        self.ssc.start()
        self.ssc.awaitTermination()

    def __train__(self, timestamp, rdd: pyspark.RDD) -> DataFrame:
        try:
            print(f"[Spark] New RDD at {timestamp}")
            if not rdd.isEmpty():
                count = rdd.count()
                print(f"[Spark] RDD has {count} records")

                # sample = rdd.take(1)
                # print(f"[Spark] Sample: {sample}")
                # print(f"[Spark] Sample type: {type(sample[0])} | len = {len(sample[0])}")

                df = rdd.map(lambda x: (x[:-1], int(x[-1]))) \
                    .toDF(["image", "label"])

                # Huấn luyện
                preds, acc, prec, rec, f1 = self.model.train(df)

                print(
                    f"[Spark] Train metrics - Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
            else:
                print("[Spark] Empty RDD")

        except Exception as e:
            print(f"[Spark ERROR] Exception in __train__: {e}")


    def predict(self):
        stream = self.dataloader.parse_stream()
        stream.foreachRDD(self.__predict__)

        self.ssc.start()
        self.ssc.awaitTermination()

    def __predict__(self, timestamp, rdd: pyspark.RDD) -> DataFrame:
        try:
            print(f"[Spark] New RDD at {timestamp}")
            if not rdd.isEmpty():
                count = rdd.count()
                print(f"[Spark] RDD has {count} records")

                sample = rdd.take(1)
                print(f"[Spark] Sample: {sample}")
                print(f"[Spark] Sample type: {type(sample[0])} | len = {len(sample[0])}")

                df = rdd.map(lambda x: (x[:-1], int(x[-1]))) \
                    .toDF(["image", "label"])

                # Huấn luyện
                preds, acc, prec, rec, f1 = self.model.predict(df)

                print(
                    f"[Spark] Train metrics - Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
            else:
                print("[Spark] Empty RDD")

        except Exception as e:
            print(f"[Spark ERROR] Exception in __predict__: {e}")
    