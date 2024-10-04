"""Set of helper functions."""
import pickle
from pathlib import Path

import numba
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler

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


def get_standardized_data(scalars: StandardScaler, data: np.ndarray) -> np.ndarray:
    """Standardizes sensor data.

    :param scalars: previously fitted scalar
    :param data: data to be transformed
    :return: transformed standardized data
    """
    for i in range(data.shape[0]):
        # transform each window based on the scalar
        standardized_sample = scalars.transform(data[i])
        data[i] = standardized_sample
    return data


def _solve_overlap(input_array: np.ndarray, gap_size: int) -> numba.typed.List:
    """Merge intervals that are overlapping and that are a distance less or equal to gap_size from each other."""
    stack = numba.typed.List()
    stack.append(input_array[0])

    for i in range(1, len(input_array)):
        if stack[-1][0] <= input_array[i][0] <= (stack[-1][1] + gap_size) <= (input_array[i][1] + gap_size):
            stack[-1][1] = input_array[i][1]
        else:
            stack.append(input_array[i])

    return stack


def merge_intervals(input_array: np.ndarray, gap_size: int = 0) -> np.ndarray:
    """Merge intervals that are overlapping and that are a distance less or equal to gap_size from each other.

    This is actually a wrapper for _solve_overlap that is needed because numba can not compile np.sort().

    Parameters
    ----------
    input_array : (n, 2) np.ndarray
        The np.ndarray containing the intervals that should be merged
    gap_size : int
        Integer that sets the allowed gap between intervals.
        For examples see below.
        Default is 0.

    Returns
    -------
    merged intervals array
        (n, 2) np.ndarray containing the merged intervals

    Examples
    --------
    >>> tests = np.array([[1, 3], [2, 4], [6, 8], [5, 7], [10, 12], [11, 15], [18, 20]])
    >>> merge_intervals(tests)
    array([[ 1,  4],
           [ 5,  8],
           [10, 15],
           [18, 20]])

    >>> merge_intervals(tests, 2)
    array([[ 1, 15],
           [18, 20]])

    """
    return np.array(_solve_overlap(np.sort(input_array, axis=0, kind="stable"), gap_size))
