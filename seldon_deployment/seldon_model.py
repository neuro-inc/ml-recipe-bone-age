import os
import pathlib
import logging
from typing import Iterable, Dict, Union, List

import numpy as np
import cv2

from src.transforms import get_transform
from src.model import m46, convert_checkpoint

MOUNTED_MODELS_ROOT = pathlib.Path("/storage")


class SeldonModel:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        model_path = self._find_model()
        self.logger.info(f"Loading model at '{str(model_path)}'")
        h, w = 2000, 1500
        scale = 0.25
        input_shape = (1, int(h * scale), int(w * scale))
        crop_dict = {"crop_center": (1040, 800), "crop_size": (h, w)}
        self.test_transform = get_transform(
            augmentation=False, crop_dict=crop_dict, scale=scale
        )
        checkpoint = convert_checkpoint(
            model_path, {"input_shape": input_shape, "model_type": "age"}
        )
        self.model = m46.from_ckpt(checkpoint)
        self.logger.info("Model loaded.")

    def predict(
        self, img_bytes: bytes, names: Iterable[str], meta: Dict = None
    ) -> Union[np.ndarray, List, str, bytes]:
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        image = self.test_transform({"image": image, "label": np.ones(1)})["image"]
        image = image.unsqueeze(0)
        pred = self.model(image)
        result = 120 * (1 + pred.flatten())
        return result.detach().numpy()

    def _find_model(self) -> pathlib.Path:
        env_path = os.environ.get("MODEL_PATH")
        if not env_path:
            model_path = list(MOUNTED_MODELS_ROOT.glob("**/*.pth"))[-1]
        else:
            model_path = pathlib.Path(env_path)
            if not model_path.is_file():
                model_path = list(model_path.glob("**/*.pth"))[-1]

        return model_path
