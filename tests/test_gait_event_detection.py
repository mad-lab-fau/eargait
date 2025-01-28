from pathlib import Path
from unittest import TestCase

from pandas.testing import assert_frame_equal

from eargait.event_detection.diao_adapted_event_detection import DiaoAdaptedEventDetection
from eargait.event_detection.diao_event_detection import DiaoEventDetection
from eargait.utils.helpers import load_pickle
from tests.helpers_test import load_data_of_interval

HERE = Path(__file__).parent


class TestImport:
    def test_diao_event_detection(self, snapshot):
        ear_data_short = load_data_of_interval()
        event_detection_algorithm = DiaoEventDetection(sample_rate_hz=50, window_length=50)
        event_detection_algorithm.detect(ear_data_short)
        gait_events = event_detection_algorithm.event_list_
        # gait_events_ground_truth = load_pickle(HERE.joinpath("test_data/gait_events_diao_orig.pkl"))
        # assert_frame_equal(gait_events["left_sensor"], gait_events_ground_truth["left_sensor"])
        # assert_frame_equal(gait_events["right_sensor"], gait_events_ground_truth["right_sensor"])
        snapshot.assert_match(gait_events["left_sensor"], "gait_events_diao_orig_left")
        snapshot.assert_match(gait_events["right_sensor"], "gait_events_diao_orig_right")

    def test_diao_adapted_event_detection(self, snapshot):
        ear_data_short = load_data_of_interval()
        event_detection_algorithm = DiaoAdaptedEventDetection(sample_rate_hz=50, window_length=50)
        event_detection_algorithm.detect(ear_data_short)
        gait_events = event_detection_algorithm.event_list_
        # gait_events_ground_truth = load_pickle(HERE.joinpath("test_data/gait_events_diao_adapted.pkl"))
        # assert_frame_equal(gait_events["left_sensor"], gait_events_ground_truth["left_sensor"])
        # assert_frame_equal(gait_events["right_sensor"], gait_events_ground_truth["right_sensor"])
        snapshot.assert_match(gait_events["left_sensor"], "gait_events_diao_adapted_left")
        snapshot.assert_match(gait_events["right_sensor"], "gait_events_diao_adapted_right")
