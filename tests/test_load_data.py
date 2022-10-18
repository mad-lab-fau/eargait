import pickle
from pathlib import Path
from unittest import TestCase

import pandas as pd
from signialib import Session

from eargait.preprocessing.load_data_helpers import load
from eargait.preprocessing.rotations import aling_gravity_and_convert_ear_to_ebf, convert_ear_to_ebf

HERE = Path(__file__).parent


class TestImport(TestCase):
    def test_is_signia_session(self):
        session = load(HERE.joinpath("test_data/subject01"))
        assert isinstance(session, Session)

    def test_is_signia_session_with_target_frequency(self):
        session = load(HERE.joinpath("test_data/subject01"), 50)
        self.assertEqual(session.info.sampling_rate_hz[0], 50)

    def test_convert_to_ebf(self):
        with open(HERE.joinpath("test_data/converted_to_ebf.pickle"), "rb") as handle:
            reference_data_ebf = pickle.load(handle)
        session = load(HERE.joinpath("test_data/subject01"), skip_calibration=True)
        data_ebf = convert_ear_to_ebf(session)
        for key, val in data_ebf.items():
            pd.testing.assert_frame_equal(val, reference_data_ebf[key])

    def test_align_to_gravity_convert_to_ebf(self):
        with open(HERE.joinpath("test_data/align_to_gravity_converted_to_ebf.pickle"), "rb") as handle:
            reference_data_ebf = pickle.load(handle)
        session = load(HERE.joinpath("test_data/subject01"))
        data_ebf = aling_gravity_and_convert_ear_to_ebf(session)
        for key, val in data_ebf.items():
            pd.testing.assert_frame_equal(val, reference_data_ebf[key])
