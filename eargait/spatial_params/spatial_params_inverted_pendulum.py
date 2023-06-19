"""Class spatial parameter using the inverted pendulum method by Zjilstra [1]."""

from pathlib import Path
from typing import Dict, TypeVar, Union

import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import cumtrapz

from eargait.spatial_params.spatial_params_base import SpatialParamsBase
from eargait.utils.helper_datatype import EventList, is_event_list
from eargait.utils.helper_gaitmap import SensorData
from eargait.utils.helpers import butter_highpass_filter, butter_lowpass_filter

HERE = Path(__file__).parent

Self = TypeVar("Self", bound="SpatialParamsInvertedPendulum")


class SpatialParamsInvertedPendulum(SpatialParamsBase):
    """Class spatial parameter using the inverted pendulum method by Zjilstra [1].

    Parameters
    ----------
    data
       ToDo

    Attributes
    ----------
    event_list_
        ToDo

    Examples
    --------
    Get gait events from single sensor signal

    >>> spatial_method = SpatialParamsInvertedPendulum()
    >>> spatial_method.calculate(data=data, sampling_rate_hz=200.0)

    Notes
    -----
    proprocessgin: 4th order  Butterworth lowpass, cut off 20hz

    acc --> double integration --> high pass filter (4th order butterworht cutoff 0.1Hz) --> distance estimation


    [1] Zijlstra, W. and Hof, A.L. Assessment of spatio-temporal gait parameters from trunk accelerations during human
        walking. Gait & Posture 18.2, 2003.

    """

    sample_rate_hz: int
    com_height: float  # in m

    model_path: Path

    velocity: pd.DataFrame  # MultiIndex DataFrame with index: (s_id, [stride_index])
    position: pd.DataFrame  # MultiIndex DataFrame with index: (s_id, [stride_index])

    step_length_: None

    def __init__(self, sample_rate_hz: int, com_height: float):
        self.sample_rate_hz = sample_rate_hz
        self.com_height = com_height
        super().__init__()

    def estimate(self, data: SensorData, event_list: EventList) -> Union[Dict, pd.DataFrame]:
        if self.com_height > 2.0:
            raise ValueError("COM Height greater 1m. Must be provided in Meters.")

        self._check_sensor_frame_orientation(data)
        kind = is_event_list(event_list)

        if kind == "single":
            tmp = self._get_step_length_single(data, event_list)
            tmp.name = "step_length"
            spatial_params = pd.merge(tmp, event_list["side"], left_index=True, right_index=True)
        else:
            spatial_params = {}
            for sensor_pos in event_list.keys():
                tmp = self._get_step_length_single(data[sensor_pos], event_list[sensor_pos])
                tmp.name = "step_length"
                spatial_params[sensor_pos] = pd.merge(
                    tmp, event_list[sensor_pos]["side"], left_index=True, right_index=True
                )
        self.step_length_ = spatial_params
        return self.step_length_

    def _get_step_length_single(self, data: pd.DataFrame, event_list: pd.DataFrame) -> pd.DataFrame:
        filtered_data = self._low_pass_filter_single_sensor(data, 20, self.sample_rate_hz / 2, 4)

        # todo: not sure whether gravity needs to be subtracted
        # filtered_data = self._subtract_gravity(filtered_data)

        spatio = pd.Series(index=event_list.index)
        start_index = filtered_data.index[0]

        velocity = cumtrapz(filtered_data["acc_si"].to_numpy(), dx=1 / self.sample_rate_hz)
        position = cumtrapz(velocity, dx=1 / self.sample_rate_hz)
        position_filtered = self._high_pass_filter_array(position, cut_off=0.1, nyq_fs=self.sample_rate_hz / 2, order=4)
        for s_id in range(1, event_list.shape[0]):
            start = event_list["ic"].to_numpy()[s_id - 1] - start_index
            ende = event_list["ic"].to_numpy()[s_id] - start_index
            if np.isnan(start) or np.isnan(ende):
                spatio.iloc[s_id] = np.nan
                continue
            sl = self._calculate_step_length(position_filtered[int(start) : int(ende)], self.com_height)
            spatio.iloc[s_id] = sl
        return spatio

    @staticmethod
    def _calculate_step_length(position_data: np.array, com_height: float) -> float:
        assert position_data.shape[0] != 0
        h = max(position_data) - min(position_data)
        sl = 2 * np.sqrt(2 * com_height * h - (h * h))
        return sl

    @staticmethod
    def _low_pass_filter_single_sensor(data: pd.DataFrame, cut_off: int, nyq_fs: int, order: int):
        assert isinstance(data, pd.DataFrame)
        return butter_lowpass_filter(data, cut_off, nyq_fs, order)

    @staticmethod
    def _high_pass_filter_single_sensor(data: pd.DataFrame, cut_off: int, nyq_fs: int, order: int):
        assert isinstance(data, pd.DataFrame)
        return butter_highpass_filter(data, cut_off, nyq_fs, order)

    @staticmethod
    def _high_pass_filter_array(data: np.ndarray, cut_off: int, nyq_fs: int, order: int):
        # assert isinstance(data, pd.DataFrame)
        normal_cutoff = float(cut_off) / nyq_fs
        b, a = signal.butter(order, normal_cutoff, btype="highpass")
        try:
            data_filt = signal.filtfilt(b, a, data)
        except ValueError:
            data_filt = signal.filtfilt(b, a, data, padlen=len(data) - 5)
        return data_filt

    @staticmethod
    def _subtract_gravity(data):
        col = "acc_si"
        if col not in data.columns:
            raise ValueError("Sensor data not in body frame.")
        grav = -9.81 if data[col].mean() > 0 else 9.81
        data = data.copy()
        data.loc[:, col] += grav
        return data

    def _check_sensor_frame_orientation(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]):
        """Raise error if orientation does not fit."""
        if isinstance(data, dict):
            for _, val in data.items():
                self._check_single_sensor_frame_format(val)
                self._check_single_sensor_orientation(val)
        else:
            self._check_single_sensor_frame_format(data)
            self._check_single_sensor_orientation(data)

    @staticmethod
    def _check_single_sensor_frame_format(data: pd.DataFrame):
        """Raise error if sensor frame format does not fit."""
        bool_columns = set(["acc_si", "acc_ml", "acc_pa"]).issubset(data.columns)
        if not bool_columns:
            raise ValueError(
                "The sensor data seems to be in an incorrect format. "
                "Columns ['acc_si', 'acc_ml', 'acc_pa'] are required, but "
                "columns are: ",
                data.columns,
            )

    @staticmethod
    def _check_single_sensor_orientation(data: pd.DataFrame):
        """Raise error if orientation of third axis does not match direction of gravtiy."""
        bool_not_gravity = data["acc_si"].mean() > 5.5
        if bool_not_gravity:
            raise ValueError(
                "The sensor data seems to be in an incorrect format. 'acc_z' need to point in the "
                "direction of gravity."
            )
