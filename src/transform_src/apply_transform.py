# apply_transform.py

import numpy as np
import torchvision.transforms as T
from PIL import Image

class ApplyTransform:
    def __init__(self):
        # Khởi tạo các phép biến đổi
        self.transforms = T.Compose([
            T.RandomHorizontalFlip(),  # Lật ngẫu nhiên
            T.ToTensor(),              # [0,255] -> [0,1]
            T.Normalize(
                mean=(0.4913997551666284, 0.48215855929893703, 0.4465309133731618),
                std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628)
            )
        ])

    def __call__(self, x: list) -> list:
        """
        x: list, trong đó x[:-1] là ảnh flatten (3072 phần tử), x[-1] là label.
        """
        img = np.array(x[:-1], dtype=np.uint8).reshape(3, 32, 32).transpose(1, 2, 0)  # (32,32,3)
        label = x[-1]

        img = Image.fromarray(img)
        img = self.transforms(img)
        # Chuyển tensor (C,H,W) -> (H,W,C), rồi flatten ra lại thành vector 1D
        img = img.permute(1, 2, 0).numpy()  # tensor -> numpy
        img = img.transpose(2, 0, 1).reshape(-1)  # (3,32,32) -> (3072,)
        return list(img) + [label]