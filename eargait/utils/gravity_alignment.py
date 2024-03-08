"""Gravity alignment algorithms."""

from typing import Dict, Hashable, Union

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.stats import trim_mean
from tpcp import Algorithm
from typing_extensions import Literal

from eargait.utils.consts import SF_ACC, SF_GYR
from eargait.utils.helper_gaitmap import (
    SensorData,
    _bool_fill,
    find_shortest_rotation,
    get_multi_sensor_names,
    is_sensor_data,
    normalize,
    rotate_dataset,
    sliding_window_view,
)
from eargait.utils.helpers import butter_lowpass_filter

METRIC_FUNCTION_NAMES = Literal["maximum", "variance", "mean", "median"]
GRAV_VEC = np.array([0.0, 0.0, 9.81])


class BaseGravityAlignment(Algorithm):
    """Base class for gravity alignment algorithms."""

    _action_methods = "align_to_gravity"

    def _compute_rotation_and_align_to_gravity(
        self, dataset: SensorData, acc_vector: np.ndarray, gravity: np.array
    ) -> SensorData:
        dataset_type = is_sensor_data(dataset, check_gyr=False)

        if dataset_type == "single":
            # build rotation for dataset from acc and gravity
            rotation = self.get_gravity_rotation(acc_vector, gravity)
            rot_in_degrees = rotation.as_euler("xyz", degrees=True)
        else:
            # build rotation dict for each dataset from acc dict and gravity
            rotation = {
                name: self.get_gravity_rotation(acc_vector[name], gravity) for name in get_multi_sensor_names(dataset)
            }
            rot_in_degrees = {name: rot.as_euler("xyz", degrees=True) for name, rot in rotation.items()}
        return rotate_dataset(dataset, rotation), rot_in_degrees

    @staticmethod
    def get_gravity_rotation(gravity_vector: np.ndarray, expected_gravity: np.ndarray = GRAV_VEC) -> Rotation:
        """Find the rotation matrix needed to align actual z-axis with expected gravity.

        Parameters
        ----------
        gravity_vector : vector with shape (3,)
            axis ([x, y ,z])
        expected_gravity : vector with shape (3,)
            axis ([x, y ,z])

        Returns
        -------
        rotation
            rotation between given gravity vector and the expected gravity

        Examples
        --------
        >>> goal = np.array([0, 0, 1])
        >>> start = np.array([1, 0, 0])
        >>> rot = get_gravity_rotation(start)
        >>> rotated = rot.apply(start)
        >>> rotated
        array([0., 0., 9.81])

        """
        gravity_vector = normalize(gravity_vector)
        expected_gravity = normalize(expected_gravity)
        return find_shortest_rotation(gravity_vector, expected_gravity)


class StaticWindowGravityAlignment(BaseGravityAlignment):
    """Gravity alignment algorithm based on static window detection."""

    acc_vector: Union[np.ndarray, Dict[Hashable, np.ndarray]]

    dataset_aligned_: SensorData
    rotation_: Rotation

    def __init__(
        self,
        sampling_rate_hz,
        window_length_s: float = 0.7,
        static_signal_th: float = 2.5,
        metric: METRIC_FUNCTION_NAMES = "median",
        gravity: np.ndarray = GRAV_VEC,
        force_usage_acc=False,
    ):
        self.sampling_rate_hz = sampling_rate_hz
        self.window_length_s = window_length_s
        self.static_signal_th = static_signal_th
        self.metric = metric
        self.gravity = gravity
        self.force_usage_acc = force_usage_acc
        super().__init__()

    def align_to_gravity(self, dataset: SensorData):
        self.acc_vector = self._get_acc_grav_vector_static_period(dataset)
        self.dataset_aligned_, self.rotation_ = self._compute_rotation_and_align_to_gravity(
            dataset, self.acc_vector, self.gravity
        )

    def _get_acc_grav_vector_static_period(self, dataset: SensorData) -> Union[np.ndarray, Dict[Hashable, np.ndarray]]:
        dataset_type = is_sensor_data(dataset, check_gyr=False)
        window_length = int(round(self.window_length_s * self.sampling_rate_hz))
        if dataset_type == "single":
            # get static acc vector
            acc_vector = self._get_static_acc_vector(dataset, window_length)
            if acc_vector is None:
                raise ValueError(
                    "No static window was found. Please use the class TrimMeanGravityAlignment or an alternative "
                    "gravity alignment class or check your data."
                )
        else:
            # build dict with static acc vectors for each sensor in dataset
            acc_vector = {
                name: self._get_static_acc_vector(dataset[name], window_length)
                for name in get_multi_sensor_names(dataset)
            }
        return acc_vector

    def _get_static_acc_vector(self, data: pd.DataFrame, window_length) -> np.ndarray:
        """Extract the mean accelerometer vector describing the static position of the sensor."""
        # find static windows within the gyro data
        if self.force_usage_acc or SF_GYR[0] not in data.columns:
            cols = SF_ACC
            print("Acceleration signal is used for gravity alignment.")
        else:
            cols = SF_GYR
            print("Gyroscope signal is used for gravity alignment.")

        static_bool_array = find_static_samples(
            data[cols].to_numpy(), window_length, self.static_signal_th, self.metric
        )

        # raise error if no static windows could be found with given user settings
        if not any(static_bool_array):
            static_bool_array = find_static_samples(
                data[cols].to_numpy(), int(window_length / 2), self.static_signal_th, self.metric
            )
        if not any(static_bool_array):
            raise ValueError(
                "No static window was found. Please use the class TrimMeanGravityAlignment, an alternative gravity "
                "alignment class, or check your data."
            )

        # get mean acc vector indicating the sensor offset orientation from gravity from static sequences
        return np.median(data[SF_ACC].to_numpy()[static_bool_array], axis=0)


class TrimMeanGravityAlignment(BaseGravityAlignment):
    """Gravity alignment algorithm based on trimmed mean of accelerometer data."""

    acc_vector: Union[np.ndarray, Dict[Hashable, np.ndarray]]

    dataset_aligned_: SensorData
    rotation_: Rotation

    def __init__(
        self,
        sampling_rate_hz: int,
        trim_mean_prop: float = 0.2,
        cut_off_freq: int = 1.25,
        order: int = 4,
        gravity: np.ndarray = GRAV_VEC,
    ):
        self.sampling_rate_hz = sampling_rate_hz
        self.trim_mean_prop = trim_mean_prop
        self.cut_off_freq = cut_off_freq
        self.order = order
        self.gravity = gravity
        self.nyq = 0.5 * self.sampling_rate_hz
        super().__init__()

    def align_to_gravity(self, dataset: SensorData):
        acc_vector = self._get_acc_grav_vector_trimmed_mean(dataset)
        self.dataset_aligned_, self.rotation_ = self._compute_rotation_and_align_to_gravity(
            dataset, acc_vector, self.gravity
        )

    def _get_acc_grav_vector_trimmed_mean(self, dataset: SensorData) -> Union[np.ndarray, Dict[Hashable, np.ndarray]]:
        acc_vector: Union[np.ndarray, Dict[Hashable, np.ndarray]]
        dataset_type = is_sensor_data(dataset, check_gyr=False)
        if dataset_type == "single":
            acc_vector = self._get_trimmed_mean_acc_vector(dataset)
        else:
            acc_vector = {
                name: self._get_trimmed_mean_acc_vector(dataset[name]) for name in get_multi_sensor_names(dataset)
            }
        return acc_vector

    def _get_trimmed_mean_acc_vector(self, dataset: pd.DataFrame) -> np.ndarray:
        # filter data
        data_lowpass = butter_lowpass_filter(dataset, self.cut_off_freq, self.nyq, self.order)
        mean_vector = trim_mean(data_lowpass[SF_ACC].to_numpy(), self.trim_mean_prop, axis=0)
        return mean_vector


def find_static_samples(
    signal: np.ndarray,
    window_length: int,
    inactive_signal_th: float,
    metric: METRIC_FUNCTION_NAMES = "mean",  # noqa
    overlap: int = None,
) -> np.ndarray:
    """Search for static samples within given input signal, based on windowed L2-norm thresholding.

    .. warning::
        Due to edge cases at the end of the input data where window size and overlap might not fit your data, the last
        window might be discarded for analysis and will therefore always be considered as non-static!

    Parameters
    ----------
    signal : array with shape (n, 3)
        3D signal on which static moment detection should be performed (e.g. 3D-acc or 3D-gyr data)

    window_length : int
        Length of desired window in units of samples

    inactive_signal_th : float
       Threshold to decide whether a window should be considered as active or inactive. Window will be tested on
       <= threshold

    metric : str, optional
        Metric which will be calculated per window, one of the following strings

        'mean' (default)
            Calculates mean value per window
        'maximum'
            Calculates maximum value per window
        'median'
            Calculates median value per window
        'variance'
            Calculates variance value per window

    overlap : int, optional
        Length of desired overlap in units of samples. If None (default) overlap will be window_length - 1

    Returns
    -------
    Boolean array with length n to indicate static (=True) or non-static (=False) for each sample

    Examples
    --------
    >>> test_data = load_gyro_data(path)
    >>> get_static_moments(gyro_data, window_length=128, overlap=64, inactive_signal_th = 5, metric = 'mean')

    See Also
    --------
    gaitmap.utils.array_handling.sliding_window_view: Details on the used windowing function for this method.

    """
    # test for correct input data shape
    if np.shape(signal)[-1] != 3:
        raise ValueError("Invalid signal dimensions, signal must be of shape (n,3).")
    _METRIC_FUNCTIONS = {  # noqa
        "maximum": np.nanmax,
        "variance": np.nanvar,
        "mean": np.nanmean,
        "median": np.nanmedian,
    }  # noqa
    if metric not in _METRIC_FUNCTIONS:
        raise ValueError(f"Invalid metric passed! {metric} as metric is not supported.")

    # check if minimum signal length matches window length
    if window_length > len(signal):
        raise ValueError(
            f"Invalid window length, window must be smaller or equal than given signal length. Given signal length: "
            f"{len(signal)} with given window_length: {window_length}."
        )

    # add default overlap value
    if overlap is None:
        overlap = window_length - 1

    # allocate output array
    inactive_signal_bool_array = np.zeros(len(signal))

    # calculate norm of input signal (do this outside of loop to boost performance at cost of memory!)
    signal_norm = np.linalg.norm(signal, axis=1)

    mfunc = _METRIC_FUNCTIONS[metric]

    # Create windowed view of norm
    windowed_norm = np.atleast_2d(sliding_window_view(signal_norm, window_length, overlap, nan_padding=False))
    is_static = np.broadcast_to(mfunc(windowed_norm, axis=1) <= inactive_signal_th, windowed_norm.shape[::-1]).T

    # create the list of indices for sliding windows with overlap
    windowed_indices = np.atleast_2d(
        sliding_window_view(np.arange(0, len(signal)), window_length, overlap, nan_padding=False)
    )

    # iterate over sliding windows
    inactive_signal_bool_array = _bool_fill(windowed_indices, is_static, inactive_signal_bool_array)

    return inactive_signal_bool_array.astype(bool)
