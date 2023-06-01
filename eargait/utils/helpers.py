"""Set of helper functions."""
import pickle
from pathlib import Path

import pandas as pd
from scipy import signal

from eargait.utils.helper_gaitmap import SensorData, is_sensor_data


def load_pickle(path: Path):
    """Load a pickle file."""
    with open(path, "rb") as handle:
        b = pickle.load(handle)
    return b


def save_pickle(path: Path, variable):
    """Save a variable as pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(variable, handle, protocol=pickle.HIGHEST_PROTOCOL)


def butter_lowpass_filter(data: SensorData, cutoff_freq: int, nyq_freq: int, order: int) -> SensorData:
    """Butterworth Low Pass Filter."""
    is_sensor_data(data, check_gyr=False)
    normal_cutoff = float(cutoff_freq) / nyq_freq
    if normal_cutoff == 1:
        normal_cutoff = 10 / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype="lowpass")
    return _butter_filter(data, a, b)


def butter_highpass_filter(data: SensorData, cutoff_freq: int, nyq_freq: int, order: int) -> SensorData:
    """Butterworth High Pass Filter."""
    is_sensor_data(data, check_gyr=False)
    normal_cutoff = float(cutoff_freq) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype="highpass")
    return _butter_filter(data, a, b)


def _butter_filter(data: SensorData, a, b) -> SensorData:
    kind = is_sensor_data(data, check_gyr=False)
    if kind == "single":
        data_filtered = _single_sensor_butter_filter(data, a, b)
    elif kind == "multi":
        data_filtered = {}
        for key, val in data.items():
            data_filtered[key] = _single_sensor_butter_filter(val, a, b)
    return data_filtered


def _single_sensor_butter_filter(data: pd.DataFrame, a, b) -> pd.DataFrame:
    assert isinstance(data, pd.DataFrame)
    data_filt = data.copy()
    for col in data.columns:
        try:
            data_filt[col] = signal.filtfilt(b, a, data[col])
        except ValueError:
            data_filt[col] = signal.filtfilt(b, a, data[col], padlen=len(data) - 5)
    return data_filt
