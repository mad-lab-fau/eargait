"""Class spatial parameter using a pre-trained CNN.

A 5-fold cross validation was applied to train and evaluate the model.
A hyperparameter estimation was performed within each cross-validation split.
Out of the 5 trained models, we chose the model that performed best on the entire dataset to be the production model.

More information on the dataset, training etc., can be found in the paper XY.

The final 'production' model was created by following the guidlines in
https://tpcp.readthedocs.io/en/latest/guides/algorithm_evaluation.html.
"""
from pathlib import Path
from typing import Dict, Optional, TypeVar, Union

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models

from eargait.spatial_params.data_vector_loader import DataVectorLoader
from eargait.spatial_params.spatial_params_base import SpatialParamsBase
from eargait.utils.consts import BF_ACC
from eargait.utils.helper_datatype import EventList
from eargait.utils.helper_gaitmap import SensorData

HERE = Path(__file__).parent

Self = TypeVar("Self", bound="SpatialParamsCNN")


class SpatialParamsCNN(SpatialParamsBase):
    """Spatial Parameter Estimation Class using a pretrained CNN."""

    sample_rate_hz: int

    model_path: Path
    model = None  # todo: Define data type for model
    scaler = None  # eventual auch verstekcen
    memory: Optional[Memory]

    step_length_ = None

    def __init__(
        self,
        sample_rate_hz,
        model_path=None,
        memory: Optional[Memory] = None,
    ):
        self.sample_rate_hz = sample_rate_hz
        self.dataloader = DataVectorLoader(sample_rate_hz)
        self.model_path = model_path
        self.memory = memory

        super().__init__()

    def estimate(self, data: SensorData, event_list: EventList) -> Union[Dict, pd.DataFrame]:
        self._load_model(memory=self.memory)
        self._load_scaler(memory=self.memory)
        data_vector, realistic_stride_time = self.dataloader.get_data_vector(data, event_list)
        data_vector_normalized = self.normalize(data_vector)
        step_length = self.predict(data_vector_normalized)
        self.step_length_ = self._restructed_step_length_array(step_length, realistic_stride_time)
        return self.step_length_

    def predict(self, data_vector):
        if isinstance(data_vector, np.ndarray):
            step_length = self.model.predict(data_vector)
        else:
            step_length = {}
            for sensor, data in data_vector.items():
                step_length[sensor] = self.model.predict(data)
        return step_length

    def _load_model(self, memory: Memory):  # noqa: unused-argument
        if not self.model_path:
            # find model path
            self.model_path = HERE.joinpath("trained_models", "dl_cnn", str(self.sample_rate_hz) + "hz_acc")
        self.model = models.load_model(self.model_path)

    def _load_scaler(self, memory: Memory):  # noqa: unused-argument
        scaler_df = pd.read_csv(self.model_path.joinpath("scaler.csv"), index_col=0)[BF_ACC]
        scalers = []
        for col in scaler_df.columns:
            sca = StandardScaler()
            sca.mean_ = scaler_df[col]["mean"]
            sca.var_ = scaler_df[col]["var"]
            sca.scale_ = scaler_df[col]["scale"]
            scalers.append(sca)
        self.scaler = scalers

    def normalize(self, data):
        if isinstance(data, np.ndarray):
            data_vector = self._normalize_single(data)
        else:
            data_vector = {}
            for sensor, da in data.items():
                data_vector[sensor] = self._normalize_single(da)
        return data_vector

    def _normalize_single(self, data):
        data_copy = data.copy()
        for i in range(data_copy.shape[2]):
            shape = data_copy[:, :, i].shape
            tmp = self.scaler[i].transform(data[:, :, i].reshape(-1, 1))
            data_copy[:, :, i] = tmp.reshape(shape)
        return data_copy

    def _restructed_step_length_array(self, step_length, realistic_bool):
        if isinstance(step_length, np.ndarray):
            step_series = self._restructed_step_length_array_single(step_length, realistic_bool)
        else:
            step_series = {}
            for sensor, sl in step_length.items():
                step_series[sensor] = self._restructed_step_length_array_single(sl, realistic_bool[sensor])
        return step_series

    @staticmethod
    def _restructed_step_length_array_single(sl, bool_array) -> pd.Series:
        sl_series = pd.Series(index=bool_array.index)
        sl_series[bool_array] = np.squeeze(sl)
        return sl_series
