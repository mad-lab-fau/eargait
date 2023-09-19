"""Helper functions to load example data."""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from eargait.utils.helper_gaitmap import MultiSensorData

HERE = Path(__file__).parent.parent.parent


def get_mat_example_data_path() -> Path:
    # data directory
    data_dir = HERE.joinpath("example_data/mat_files/normal")
    return data_dir


def get_txt_example_data_path() -> Path:
    # data directory
    data_dir = HERE.joinpath("example_data/txt_file/walking_50Hz_BMI270.txt")
    return data_dir


def get_example_data() -> Tuple[MultiSensorData, int]:
    # returns example data for a single walking,  50Hz
    # sensor data is aligned, calibrated
    csv_data = pd.read_csv(
        HERE.joinpath("example_data", "aligned_calibrated_earframe_50Hz_walking_data_single_walkingbout.csv")
    )
    left = csv_data.loc[csv_data.side == "left"].drop(columns=["side"]).set_index("n_samples")
    right = csv_data.loc[csv_data.side == "right"].drop(columns=["side"]).set_index("n_samples")
    return {"left_sensor": left, "right_sensor": right}, 50


def plot_image(image_path):
    pil_im = Image.open(image_path)
    im_array = np.asarray(pil_im)
    plt.axis("off")
    plt.imshow(im_array)
    plt.show()
