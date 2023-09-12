"""Data Vector Loader for Spatial Parameter Estimation Using DL."""
from typing import Dict, Union

import numpy as np
import pandas as pd

from eargait.utils.consts import BF_ACC
from eargait.utils.helper_datatype import EventList, SingleSensorEventList, is_event_list
from eargait.utils.helper_gaitmap import SensorData, SingleSensorData


class DataVectorLoader:
    """Data vector loader for spatial parameter estimation using DL."""

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def get_data_vector(self, data: SensorData, gait_events: EventList) -> Union[Dict, pd.DataFrame]:
        kind = is_event_list(gait_events)
        if kind == "single":
            data_vector, realistic_stride_time = self._get_data_vector_single(data, gait_events)
        else:
            data_vector = {}
            realistic_stride_time = {}
            for sensor, events in gait_events.items():
                data_vector[sensor], realistic_stride_time[sensor] = self._get_data_vector_single(data[sensor], events)
                assert data_vector[sensor].shape[0] != 0
        return data_vector, realistic_stride_time.to_numpy()

    def _get_data_vector_single(self, data: SingleSensorData, gait_events: SingleSensorEventList) -> pd.DataFrame:
        vectors = self._cut_data_into_vectors_using_event_list(data, gait_events)
        realtistic_stride_time = self._consistency_check_stride_time(vectors)
        vectors_padded = self._get_padded_vectors(vectors[realtistic_stride_time])
        return vectors_padded, realtistic_stride_time

    @staticmethod
    def _cut_data_into_vectors_using_event_list(data: SingleSensorData, event_list: SingleSensorEventList):
        ic_start = event_list.ic.iloc[0:-1]
        ic_stop = event_list.ic.iloc[1::]
        data_list = [data.loc[start:stop] for start, stop in zip(ic_start, ic_stop)]
        data_vectors = pd.DataFrame({"data": data_list}, index=ic_start.index)
        return data_vectors.squeeze()

    def _get_padded_vectors(self, data_vector: pd.Series):
        data_vector_acc = data_vector.apply(lambda x: x[BF_ACC])
        data_new = self._transform_data_with_padding(data_vector_acc)
        data_new = data_new.apply(lambda x: np.reshape(x, (1, x.shape[0], x.shape[1])))
        return np.vstack(list(data_new))

    def _transform_data_with_padding(self, data):
        pad_size = self.sample_rate
        data = data.apply(lambda x: _padding(x, pad_size))
        return data

    def _consistency_check_stride_time(self, vectors_series):
        realistic_stride_time = vectors_series.apply(lambda x: self._check_stride_time(x))  # noqa
        return realistic_stride_time

    def _check_stride_time(self, x):
        if x.shape[0] > self.sample_rate:
            return False
        return True


def _padding(df, pad_size):
    diff = pad_size - df.shape[0]
    pad1 = diff // 2
    if diff % 2 == 0:
        pad2 = pad1
    else:
        pad2 = pad1 + 1
    data = df.to_numpy()
    padded_data = np.apply_along_axis(_pad, 0, data, pad1, pad2)
    return padded_data


def _pad(array, pad1, pad2):
    return np.pad(array, pad_width=(pad1, pad2), mode="constant", constant_values=(0, 0))
