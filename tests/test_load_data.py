import pickle
from pathlib import Path
from unittest import TestCase

import pandas as pd
from signialib import Session

from eargait.preprocessing.load_data_helpers import load
from eargait.preprocessing.rotations import (
    align_dataset_to_gravity,
    align_gravity_and_convert_ear_to_ebf,
    convert_ear_to_ebf,
    convert_ear_to_esf,
)
from eargait.utils import StaticWindowGravityAlignment, TrimMeanGravityAlignment

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

    def test_static_gravity_alignment_and_convert_ebf(self):
        with open(HERE.joinpath("test_data/align_to_gravity_converted_to_ebf.pickle"), "rb") as handle:
            reference_data_ebf = pickle.load(handle)
        session = load(HERE.joinpath("test_data/subject01"))
        gravity_alignment_method = StaticWindowGravityAlignment(sampling_rate_hz=session.info.sampling_rate_hz[0])
        data_ebf = align_gravity_and_convert_ear_to_ebf(session, gravity_alignment_method)
        for key, val in data_ebf.items():
            pd.testing.assert_frame_equal(val, reference_data_ebf[key])

    def test_static_gravity_alignment(self):
        with open(HERE.joinpath("test_data/align_to_gravity_static_method.pickle"), "rb") as handle:
            reference_data_ebf = pickle.load(handle)
        session = load(HERE.joinpath("test_data/subject01"))
        dataset_sf = convert_ear_to_esf(session)
        gravity_alignment_method = StaticWindowGravityAlignment(sampling_rate_hz=session.info.sampling_rate_hz[0])
        dataset_aligned = align_dataset_to_gravity(dataset_sf, gravity_alignment_method)
        for key, val in dataset_aligned.items():
            pd.testing.assert_frame_equal(val, reference_data_ebf[key])

    def test_static_gravity_alignment_forced_acc(self):
        session = load(HERE.joinpath("test_data/subject01"))
        dataset_sf = convert_ear_to_esf(session)
        gravity_alignment_method = StaticWindowGravityAlignment(
            sampling_rate_hz=session.info.sampling_rate_hz[0], static_signal_th=10, force_usage_acc=True
        )
        dataset_aligned = align_dataset_to_gravity(dataset_sf, gravity_alignment_method)

    def test_trim_gravity_alignment(self):
        with open(HERE.joinpath("test_data/align_to_gravity_trim_method.pickle"), "rb") as handle:
            reference_data_ebf = pickle.load(handle)
        session = load(HERE.joinpath("test_data/subject01"))
        dataset_sf = convert_ear_to_esf(session)
        gravity_alignment_method = TrimMeanGravityAlignment(sampling_rate_hz=session.info.sampling_rate_hz[0])
        dataset_aligned = align_dataset_to_gravity(dataset_sf, gravity_alignment_method)
        for key, val in dataset_aligned.items():
            pd.testing.assert_frame_equal(val, reference_data_ebf[key])
