from pathlib import Path
from unittest import TestCase

import numpy as np

from eargait.eargait import EarGait
from eargait.event_detection.diao_adapted_event_detection import DiaoAdaptedEventDetection
from tests.helpers_test import load_data_of_interval

HERE = Path(__file__).parent


class TestImport(TestCase):
    def test_temporal_gait_parameters(self):
        ear_data_short = load_data_of_interval()
        event_detection_algorithm = DiaoAdaptedEventDetection(sample_rate_hz=50, window_length=50)
        eargait = EarGait(
            sample_rate_hz=50, event_detection_method=event_detection_algorithm, bool_use_event_list_consistent=True
        )
        eargait.detect(ear_data_short)
        temporal_stride_params = eargait.temporal_params
        for _, tmp in temporal_stride_params.items():
            tmp["check"] = tmp["swing_time"] + tmp["stance_time"]
            assert (np.sum(tmp["stride_time"] - tmp["check"]), 0.0)

    def test_step_consistency(self):
        ear_data_short = load_data_of_interval()
        event_detection_algorithm = DiaoAdaptedEventDetection(sample_rate_hz=50, window_length=50)
        eargait = EarGait(
            sample_rate_hz=50, event_detection_method=event_detection_algorithm, bool_use_event_list_consistent=False
        )
        eargait.detect(ear_data_short)
