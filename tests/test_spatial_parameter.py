import warnings
from pathlib import Path
from unittest import TestCase

import pandas as pd

from eargait import EarGait
from eargait.event_detection import DiaoAdaptedEventDetection
from eargait.spatial_params.spatial_params_cnn import SpatialParamsCNN
from eargait.spatial_params.spatial_params_rf import SpatialParamsRandomForest, SpatialParamsRandomForestDemographics
from eargait.utils.helpers import load_pickle, save_pickle


HERE = Path(__file__).parent
TEST_DATA = HERE.joinpath("test_data", "short_example_data_acc_50hz.csv")


class TestImport(TestCase):
    # def test_spatial_method_cnn(self):
    #    data = pd.read_csv(TEST_DATA, index_col=0)
    #    sample_rate = 50
    #    diao = DiaoAdaptedEventDetection(sample_rate)

    #     spatial = SpatialParamsCNN(sample_rate, grav_alignment_method="static")
    #     ear_gait = EarGait(
    #         sample_rate_hz=sample_rate,
    #         event_detection_method=diao,
    #         spatial_params_method=spatial,
    #         bool_use_event_list_consistent=True,
    #     )

    #    ear_gait.detect(data)
    #     gait_params = ear_gait.get_gait_parameters()
    #     ref_gait_params = load_pickle(HERE.joinpath("test_data/spatial_cnn_50Hz.pkl"))
    #     assert gait_params.equals(ref_gait_params)

    def test_spatial_method_randomforest_static(self):
        data = pd.read_csv(TEST_DATA, index_col=0)
        sample_rate = 50
        diao = DiaoAdaptedEventDetection(sample_rate)

        spatial = SpatialParamsRandomForest(sample_rate, grav_alignment_method="static")
        ear_gait = EarGait(
            sample_rate_hz=sample_rate,
            event_detection_method=diao,
            spatial_params_method=spatial,
            bool_use_event_list_consistent=True,
        )
        ear_gait.detect(data)
        gait_params = ear_gait.get_gait_parameters()

        ref_gait_params = load_pickle(HERE.joinpath("test_data/spatial_rf_50Hz_static.pkl"))
        assert gait_params.equals(ref_gait_params)

    def test_spatial_method_randomforest_loading(self):
        for rate in [50, 200]:
            for method in ["static", "trim"]:
                spatial = SpatialParamsRandomForest(rate, grav_alignment_method=method)
                spatial._load_model()

    def test_spatial_method_randomforest_trim(self):
        data = pd.read_csv(TEST_DATA, index_col=0)
        sample_rate = 50  # TEST_DATA is 50Hz
        diao = DiaoAdaptedEventDetection(sample_rate)
        spatial = SpatialParamsRandomForest(sample_rate, grav_alignment_method="trim")
        ear_gait = EarGait(
            sample_rate_hz=sample_rate,
            event_detection_method=diao,
            spatial_params_method=spatial,
            bool_use_event_list_consistent=True,
        )
        ear_gait.detect(data)
        gait_params = ear_gait.get_gait_parameters()
        ref_gait_params = load_pickle(HERE.joinpath("test_data/spatial_rf_50Hz_trim.pkl"))
        assert gait_params.equals(ref_gait_params)
