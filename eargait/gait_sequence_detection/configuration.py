"""Configuration object for data loading and pruning settings."""

import math
from pathlib import Path


class Config:
    """Config object, holds information about how the data should be loaded and which pruning should be executed.

    :param har_data_path: path of dataset repo
    :param sample_rate_hz: sampling rate with which the data should be loaded
    :param selected_coords: coordinate axis which should be loaded
    :param window_length_in_ms: window length in ms
    :param step_size_in_ms: sept size in ms
    :param body_frame_coords: defines if coordinate values should be transformed into body frame
    :param imbalance_method: defines which method to tackle class imbalance should be used
    options: "WRS", "WCE", "ROS", "SMTOE"
    :param model: model which should be run
    :param use_gravity_alignment: Decide weather or not to align the data to Gravity or not
    """

    def __init__(
        self,
        har_data_path: str = Path(__file__).resolve().parent.parent.parent,
        use_gravity_alignment: bool = True,
        sample_rate_hz: int = 50,
        selected_coords: list[str] = None,
        window_length_in_ms: int = 3000,
        step_size_in_ms: int = 1500,
        body_frame_coords: bool = True,
        model: str = "conv_gru",
    ):
        self.data_base_path = har_data_path
        self.sample_rate_hz = sample_rate_hz
        self.selected_coords = selected_coords if selected_coords is not None else ["x", "y", "z"]
        self.window_length_in_ms = window_length_in_ms
        self.window_length = int((window_length_in_ms / 1000) * self.sample_rate_hz)
        self.step_size_in_ms = step_size_in_ms
        self.step_size = int((step_size_in_ms / 1000) * 200)
        self.frequency_step = int(200 / sample_rate_hz)
        self.body_frame_coords = body_frame_coords
        self.model = model
        self.window_length_in_samples = math.ceil(self.window_length / self.frequency_step)
        self.use_gravity_alignment = use_gravity_alignment
