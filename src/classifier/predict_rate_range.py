import importlib
import argparse

import cv2
import torch
import numpy as np

from .config import Config
from .models.model import EfficientNet, LightningModel

from PIL import Image


def load_config(config_name):
    """
    Dynamically load the Config class from the specified config module.
    """
    try:
        # Import the module
        module = importlib.import_module(f"configs.{config_name}")
        # Get the Config class
        Config = getattr(module, "Config")
        # Instantiate the Config class
        return Config()
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Failed to load config: {config_name}. Error: {e}")


def preprocess(image:np.ndarray, crop_size:int):
    if crop_size > 0:
        if image.shape[0] < crop_size or image.shape[1] < crop_size:
            raise ValueError("Image is too small")

        # option 1: center crop image
        offset_h = image.shape[0] // 2 - crop_size // 2
        offset_w = image.shape[1] // 2 - crop_size // 2
        offset_h = offset_h - offset_h % 8
        offset_w = offset_w - offset_w % 8

        # # option 2: just crop the left top corner
        # offset_h = 0
        # offset_w = 0

        image = image[offset_h:offset_h+crop_size, offset_w:offset_w+crop_size]

    # normalize image
    if np.ndim(image) == 3:
        image = np.transpose(image, (2, 0, 1)) # hwc->chw
    elif np.ndim(image) == 2:
        image = np.stack([image] * 3, axis=0)
    image = image / 255
    image = torch.Tensor(image)
    return image


def predict_rate_range(image:np.ndarray, config:Config):
    # Initialize model
    # model = ConstNet(config)
    model = EfficientNet(config.model, config.num_class)

    lightning_model = LightningModel.load_from_checkpoint(config.ckpt_path, model=model)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    lightning_model = lightning_model.to(device)
    lightning_model.eval()

    # load image
    image = preprocess(image, crop_size=-1)

    # predict
    image = image.to(device)[None, ...]
    # set useless value for d
    with torch.no_grad():
        logits, output = lightning_model.forward(image)
    logits = logits.detach().cpu().numpy()[0]
    # print("logits: ", logits)


    args_sorted = np.argsort(logits, axis=0)[::-1]
    logits_sorted = logits[args_sorted]
    ranges_sorted = [config.r_range_list[i] for i in args_sorted]
    # print("sorted ranges: ", ranges_sorted)
    # print("args_sorted:")
    # print(args_sorted)
    # range_predicted = config.r_range_list[label_predicted]
    # print("Predicted range: ", range_predicted)

    return ranges_sorted, logits_sorted
