import numpy as np
from pyspark.context import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.streaming.context import StreamingContext
from pyspark.streaming.dstream import DStream
from pyspark.ml.linalg import DenseVector
import torchvision.transforms as T
from transforms_scr import ApplyTransform
from trainer import SparkConfig
from PIL import Image
import json
# def transform_image(x):
#     img = np.array(x[:-1], dtype=np.uint8).reshape(3, 32, 32).transpose(1,2,0)
#     label = x[-1]
#
#     # # Tạo transform trong hàm Worker
#     # transform_ops = T.Compose([
#     #     T.RandomHorizontalFlip(),
#     #     T.ToTensor(),
#     #     T.Normalize(
#     #         mean=(0.4913997551666284, 0.48215855929893703, 0.4465309133731618),
#     #         std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628)
#     #     )
#     # ])
#     #
#     img = Image.fromarray(img)
#     # img = transform_ops(img)
#     img = img.permute(1,2,0).numpy()
#     img = img.transpose(2,0,1).reshape(-1)
#
#
#     return list(img) + [label]
def transform_image(x):
    if x is None or len(x) != 3073:
        print("BAD DATA:", x)
        return None

class DataLoader:
    def __init__(self, 
                 sparkContext:SparkContext, 
                 sparkStreamingContext: StreamingContext, 
                 sqlContext: SQLContext,
                 sparkConf: SparkConfig) -> None:
        
        self.sc = sparkContext
        self.ssc = sparkStreamingContext
        self.sparkConf = sparkConf
        self.sql_context = sqlContext
        self.stream = self.ssc.socketTextStream(
            hostname=self.sparkConf.stream_host, 
            port=self.sparkConf.port
        )
        # self.transforms = ApplyTransform()

    # def parse_stream(self) -> DStream:
    #     """
    #     preprocess data from tcp
    #     """
    #     pp
    #     json_stream = self.stream.map(lambda line: json.loads(line))
    #     print("[DataLoader] Received new JSON batch from TCP stream")
    #
    #     json_stream_exploded = json_stream.flatMap(lambda x: x.values())
    #     json_stream_exploded = json_stream_exploded.map(
    #         lambda x: [x[f"feature-{i}"] for i in range(3072)] + [x["label"]]
    #     )
    #     pixels = json_stream_exploded.map(lambda x: [np.array(x[:-1]).reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8),x[-1]])
    #     pixels = DataLoader.preprocess(pixels, self.transforms)
    #     return pixels
    def parse_stream(self) -> DStream:
        # Chỉ đọc từ socket
        raw_stream = self.stream
        parsed_stream = raw_stream.map(lambda line: json.loads(line))
        parsed_stream = parsed_stream.flatMap(lambda x: x.values())

        parsed_stream = parsed_stream.map(
                lambda x: [x[f"feature-{i}"] for i in range(3072)] + [x["label"]]
        )
        pixels = parsed_stream.map(
            lambda x: [np.array(x[:-1], dtype=np.uint8).reshape(3, 32, 32).transpose(1, 2, 0), x[-1]]
        )
        # pixels = pixels.map(
        #     lambda x: [
        #         (T.Compose([
        #             T.RandomHorizontalFlip(),
        #             T.ToTensor(),
        #             T.Normalize(
        #                 mean=(0.4913997551666284, 0.48215855929893703, 0.4465309133731618),
        #                 std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628)
        #             )
        #         ])(Image.fromarray(x[0]))),
        #         x[1]
        #     ]
        # )

        pixels = parsed_stream
        # pixels = parsed_stream.map(
        #     lambda x: self.transforms.__call__(x)
        # )
        return pixels