"""from pathlib import Path
from unittest import TestCase

import pandas as pd
from pandas.testing import assert_frame_equal

from eargait import EarGait
from eargait.event_detection import DiaoAdaptedEventDetection
from eargait.spatial_params.spatial_params_cnn import SpatialParamsCNN
from eargait.spatial_params.spatial_params_example_class import SpatialParamsExample
from eargait.spatial_params.spatial_params_rf import SpatialParamsRandomForest, SpatialParamsRandomForestDemographics
from eargait.utils.gait_parameters import get_temporal_params
from eargait.utils.helpers import load_pickle

HERE = Path(__file__).parent
TEST_DATA = HERE.joinpath("test_data", "short_example_data_acc_50hz.csv")


class TestImport(TestCase):
    def test_spatial_method_cnn(self):
        data = pd.read_csv(TEST_DATA, index_col=0)
        sample_rate = 50
        diao = DiaoAdaptedEventDetection(sample_rate)

        spatial = SpatialParamsCNN(sample_rate)
        ear_gait = EarGait(
            sample_rate_hz=sample_rate,
            event_detection_method=diao,
            spatial_params_method=spatial,
            bool_use_event_list_consistent=True,
        )

        ear_gait.detect(data)
        gait_params = ear_gait.get_gait_parameters()
"""