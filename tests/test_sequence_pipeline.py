import pandas as pd
from pandas.testing import assert_frame_equal
from pathlib import Path
from unittest import TestCase

from eargait.gait_sequence_analyzer_pipeline import GaitSequenceAnalyzerPipeline
from eargait.preprocessing import load, align_gravity_and_convert_ear_to_ebf
from eargait.utils import TrimMeanGravityAlignment

HERE = Path(__file__).parent

class TestGaitSequenceAnalyzerPipeline(TestCase):

    @classmethod
    def setUpClass(cls):
        session_path = HERE / "test_data" / "subject01"
        target_sample_rate = 50
        session = load(session_path, target_sample_rate_hz=target_sample_rate, skip_calibration=True)
        trim_method = TrimMeanGravityAlignment(sampling_rate_hz=target_sample_rate)
        ear_data_dict = align_gravity_and_convert_ear_to_ebf(session, trim_method)
        cls.ear_data = ear_data_dict['left_sensor']
        cls.manual_sequences = pd.read_csv(HERE / "test_data" / "manual_sequences.csv", index_col=0)
        cls.expected_auto_sequences = pd.read_csv(HERE / "test_data" / "expected_auto_sequences.csv", index_col=0)
        cls.expected_auto_params = pd.read_csv(HERE / "test_data" / "expected_auto_params.csv", index_col=0)
        cls.expected_manual_sequences = pd.read_csv(HERE / "test_data" / "expected_manual_sequences.csv", index_col=0)

    # --- Initialization Tests (input validation) ---
    # Check valid input parameters are set correctly
    def test_init_with_valid_inputs(self):
        pipeline = GaitSequenceAnalyzerPipeline(sample_rate=50, strictness=1, min_seq_length=2)
        self.assertEqual(pipeline.sample_rate, 50)
        self.assertEqual(pipeline.strictness, 1)
        self.assertEqual(pipeline.min_seq_length, 2)

    # Verify sample_rate can't be None or invalid
    def test_init_invalid_sample_rate(self):
        with self.assertRaises(ValueError):
            GaitSequenceAnalyzerPipeline(sample_rate=None)

    # Check strictness cannot be negative
    def test_init_negative_strictness(self):
        with self.assertRaises(ValueError):
            GaitSequenceAnalyzerPipeline(sample_rate=50, strictness=-1)

    # Check min_seq_length must be >= 1
    def test_init_invalid_min_seq_length(self):
        with self.assertRaises(ValueError):
            GaitSequenceAnalyzerPipeline(sample_rate=50, min_seq_length=0)

    # --- Input Data Validation Tests ---
    # Ensure data must be provided as DataFrame (check incorrect type)
    def test_assert_data_correctness_non_dataframe(self):
        pipeline = GaitSequenceAnalyzerPipeline(sample_rate=50)
        with self.assertRaises(AssertionError):
            pipeline.assert_data_correctness("not a dataframe", activity="walking")

    # DataFrame provided must not be empty
    def test_assert_data_correctness_empty_dataframe(self):
        pipeline = GaitSequenceAnalyzerPipeline(sample_rate=50)
        with self.assertRaises(AssertionError):
            pipeline.assert_data_correctness(pd.DataFrame(), activity="walking")

    # DataFrame values must be numeric
    def test_assert_data_correctness_non_numeric_dataframe(self):
        pipeline = GaitSequenceAnalyzerPipeline(sample_rate=50)
        invalid_data = pd.DataFrame({"acc": ["a", "b", "c"]})
        with self.assertRaises(AssertionError):
            pipeline.assert_data_correctness(invalid_data, activity="walking")

    # Confirm valid data passes without errors
    def test_assert_data_correctness_valid_dataframe(self):
        pipeline = GaitSequenceAnalyzerPipeline(sample_rate=50)
        pipeline.assert_data_correctness(self.ear_data, activity="walking")  # Should pass without error

    # --- Manual Sequences Input Validation Tests ---
    # Columns must be exactly ["start", "end"]
    def test_assert_predef_seq_correctness_invalid_columns(self):
        pipeline = GaitSequenceAnalyzerPipeline(sample_rate=50)
        invalid_seq = pd.DataFrame({"begin": [1000], "finish": [2000]})
        with self.assertRaises(AssertionError):
            pipeline.assert_predef_seq_correctness(invalid_seq, self.ear_data)

    # Values in sequences must be integers
    def test_assert_predef_seq_non_integer_values(self):
        pipeline = GaitSequenceAnalyzerPipeline(sample_rate=50)
        invalid_seq = pd.DataFrame({"start": [1000.5], "end": [1500.8]})
        with self.assertRaises(AssertionError):
            pipeline.assert_predef_seq_correctness(invalid_seq, self.ear_data)

    # Sequence must be at least 3 seconds long (e.g., 150 samples at 50 Hz)
    def test_assert_predef_seq_short_sequences(self):
        pipeline = GaitSequenceAnalyzerPipeline(sample_rate=50)
        short_seq = pd.DataFrame({"start": [0], "end": [50]})  # 1 second at 50 Hz
        with self.assertRaises(ValueError):
            pipeline.assert_predef_seq_correctness(short_seq, self.ear_data)

    # Overlapping sequences trigger a warning
    def test_assert_predef_seq_overlapping_warns(self):
        pipeline = GaitSequenceAnalyzerPipeline(sample_rate=50)
        overlapping_seq = pd.DataFrame({"start": [1000, 1200], "end": [1500, 1600]})
        with self.assertWarns(UserWarning):
            pipeline.assert_predef_seq_correctness(overlapping_seq, self.ear_data)

    # --- Output Correctness Tests ---
    # Check automatic sequence detection matches expected output
    def test_compute_auto_sequences_correct_output(self):
        pipeline = GaitSequenceAnalyzerPipeline(sample_rate=50, strictness=0, min_seq_length=2)
        pipeline.compute(self.ear_data, activity="walking")

        assert_frame_equal(
            pipeline.auto_sequence_list_.reset_index(drop=True),
            self.expected_auto_sequences.reset_index(drop=True),
        )
        assert_frame_equal(
            pipeline.auto_average_params_.reset_index(drop=True),
            self.expected_auto_params.reset_index(drop=True),
        )

    # Verify manual sequences are computed as expected
    def test_compute_manual_sequences_correct_output(self):
        pipeline = GaitSequenceAnalyzerPipeline(sample_rate=50, strictness=0, min_seq_length=2)
        pipeline.compute_predef_seq(self.ear_data, self.manual_sequences)

        assert_frame_equal(
            pipeline.manual_sequence_list_.reset_index(drop=True),
            self.expected_manual_sequences.reset_index(drop=True),
        )

    # --- Acceptable Input Variables Tests ---
    # Check invalid activities raise ValueError
    def test_activity_parameter_validation(self):
        pipeline = GaitSequenceAnalyzerPipeline(sample_rate=50)
        for invalid_activity in [1, None, ["running"], {"activity": "walking"}]:
            with self.assertRaises((AssertionError, ValueError)):
                pipeline.compute(self.ear_data, activity=invalid_activity)

        # Valid activity input
        try:
            pipeline.compute(self.ear_data, activity="walking")
        except Exception as e:
            self.fail(f"compute() raised {e} unexpectedly with valid activity input 'walking'!")

    # Check various invalid manual sequence inputs
    def test_sequence_input_types_validation(self):
        pipeline = GaitSequenceAnalyzerPipeline(sample_rate=50)
        invalid_inputs = [None, "sequence", [0, 100], {"start": 0, "end": 100}]
        for seq in invalid_inputs:
            with self.assertRaises(AssertionError):
                pipeline.assert_predef_seq_correctness(seq, self.ear_data)

    # --- Other Tests for run_eargait Method ---
    # run_eargait must fail clearly if sequence_list isn't a DataFrame
    def test_run_eargait_invalid_sequence_list_type(self):
        pipeline = GaitSequenceAnalyzerPipeline(sample_rate=50)
        invalid_sequence_list = ["not", "a", "dataframe"]
        with self.assertRaises(TypeError):
            pipeline.run_eargait(self.ear_data, invalid_sequence_list)

    # run_eargait should fail clearly if provided sequence_list is empty
    def test_run_eargait_empty_sequence_list(self):
        pipeline = GaitSequenceAnalyzerPipeline(sample_rate=50)
        empty_sequence_list = pd.DataFrame()
        with self.assertRaises(ValueError):
            pipeline.run_eargait(self.ear_data, empty_sequence_list)

    # --- Other Tests for get_combined_df Method ---
    # get_combined_df without arguments returns the combined_df_
    def test_get_combined_df_default_behavior(self):
        pipeline = GaitSequenceAnalyzerPipeline(sample_rate=50)

        # What get_combined_df reads in 'auto' mode
        pipeline.auto_sequence_list_ = pd.DataFrame({"start": [0, 100], "end": [50, 150]})
        pipeline.auto_average_params_ = pd.DataFrame({"param1": [10, 20]})

        result = pipeline.get_combined_df()  # default mode='auto'
        expected_result = pd.DataFrame({"start": [0, 100], "end": [50, 150], "param1": [10, 20]})

        assert_frame_equal(result.reset_index(drop=True), expected_result.reset_index(drop=True))

    # get_combined_df with parameters returns a DataFrame including only specified columns
    def test_get_combined_df_with_parameters(self):
        pipeline = GaitSequenceAnalyzerPipeline(sample_rate=50)
        pipeline.auto_sequence_list_ = pd.DataFrame({"start": [0, 100], "end": [50, 150]})
        pipeline.auto_average_params_ = pd.DataFrame({"param1": [10, 20], "param2": [30, 40]})

        result = pipeline.get_combined_df(parameters=["param1"])
        expected_result = pd.DataFrame({
            "start": [0, 100],
            "end": [50, 150],
            "param1": [10, 20]
        })

        assert_frame_equal(result.reset_index(drop=True), expected_result.reset_index(drop=True))

