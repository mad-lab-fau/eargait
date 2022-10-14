"""Example Class for Spatial Parameter Estimation methods."""
import warnings
from pathlib import Path
from typing import Dict, TypeVar, Union

import numpy as np
import pandas as pd

from eargait.spatial_params.spatial_params_base import SpatialParamsBase
from eargait.utils.helper_datatype import EventList
from eargait.utils.helper_gaitmap import SensorData

HERE = Path(__file__).parent

Self = TypeVar("Self", bound="SpatialParamsExample")


class SpatialParamsExample(SpatialParamsBase):
    """Spatial Parameter Estimation Class using a pretrained Random Forest."""

    sample_rate_hz: int

    def __init__(self, sample_rate_hz):
        self.sample_rate_hz = sample_rate_hz
        super().__init__()

    def estimate(self, data: SensorData, event_list: EventList) -> Union[Dict, pd.DataFrame]:
        """Caution: Example function! Sets all parameter to NaN."""
        warnings.warn(
            "Example class for spatial parameter estimation is used. No spatial parameters are calculated, "
            "step length and stride length are set to NaN."
        )
        spatial_params = event_list.copy()
        spatial_params = spatial_params.rename(columns={"ic": "step_length", "tc": "stride_length"})
        spatial_params.step_length = np.nan
        spatial_params.stride_length = np.nan
        return spatial_params
