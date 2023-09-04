from pathlib import Path
from unittest import TestCase

import pandas as pd
from pandas.testing import assert_frame_equal

from eargait import EarGait
from eargait.spatial_params.spatial_params_example_class import SpatialParamsExample
from eargait.utils.gait_parameters import get_temporal_params
from eargait.utils.helpers import load_pickle

HERE = Path(__file__).parent


def _dummy_event_list():
    r = pd.DataFrame()
    r = r.assign(ic=[1000, 2000, 3000, 5000, 8000])
    r = r.assign(tc=[500, 1500, 2500, 4500, 7000])
    r = r.assign(side=["ipsilateral", "contralateral", "ipsilateral", "contralateral", "ipsilateral"])
    r.index.name = "s_id"
    return r


def _dummy_spatiotemp_list():
    r = pd.DataFrame()
    r = r.assign(stride_time=[2.0, 1.5, 2.5, 1.5])
    r = r.assign(swing_time=[1.0, 0.75, 1.25, 0.70])
    r = r.assign(stance_time=[1.0, 0.75, 1.25, 0.80])
    r = r.assign(side=["ipsilateral", "contralateral", "ipsilateral", "contralateral"])
    r = r.assign(step_length=[0.85, 0.78, 0.33, 0.67])
    r = r.assign(stride_length=[1.5, 1.3, 1.4, 1.5])
    r.index.name = "s_id"
    return r


class TestImport(TestCase):
    def test_temporal_parameter_estimation(self):
        event_list = load_pickle(HERE.joinpath("test_data/gait_events_diao_adapted.pkl"))
        ref_temp_params = pd.read_csv(HERE.joinpath("test_data/temporal_params_right_sensor.csv"), index_col=0)
        estimated_temp_params = get_temporal_params(event_list, 50)
        assert_frame_equal(estimated_temp_params["right_sensor"], ref_temp_params)

    def test_get_asymmetry(self):
        spatial_method = SpatialParamsExample(50)
        eargait = EarGait(sample_rate_hz=50, spatial_params_method=spatial_method)
        eargait.event_list = _dummy_event_list()

        # single sensor
        eargait._spatiotemporal_params_memory = _dummy_spatiotemp_list()
        asy = eargait.get_asymmetry()
        asy_manual = pd.Series(
            index=asy.index,
            data=[
                0.75,
                0.4,
                0.35,
                0.13500000000000012,
                0.050000000000000044,
                0.4,
                0.43243243243243246,
                0.3684210526315789,
                0.20532319391635,
                0.03508771929824565,
            ],
        )
        pd.testing.assert_series_equal(asy, asy_manual)

    def test_symmetry_index(self):
        spatial_method = SpatialParamsExample(50)
        eargait = EarGait(sample_rate_hz=50, spatial_params_method=spatial_method)
        eargait.event_list = _dummy_event_list()

        # single sensor
        eargait._spatiotemporal_params_memory = _dummy_spatiotemp_list()
        asy = eargait.get_symmetry_index()
        asy_manual = pd.Series(
            index=asy.index, data=[40.0, 43.24324324324324, 36.84210526315789, 20.532319391635, 3.5087719298245648]
        )
        pd.testing.assert_series_equal(asy, asy_manual)

    def test_variability(self):
        spatial_method = SpatialParamsExample(50)
        eargait = EarGait(sample_rate_hz=50, spatial_params_method=spatial_method)
        eargait.event_list = _dummy_event_list()

        # single sensor
        eargait._spatiotemporal_params_memory = _dummy_spatiotemp_list()
        asy = eargait.get_variability()
        print(asy)
        asy_manual = pd.Series(
            index=asy.index,
            data=[
                0.47871355387816905,
                0.25331140255951107,
                0.2273030282830976,
                0.23056091024571648,
                0.0957427107756338,
                0.25531389540169014,
                0.27385016492920117,
                0.23926634556115536,
                0.35066298136230645,
                0.0671878672109711,
            ],
        )
        pd.testing.assert_series_equal(asy, asy_manual)
