from typing import Union, Dict, List, Iterator, Type, TypeVar, Generic, Tuple , Optional
from dataclasses import dataclass
import warnings
import pandas as pd
import numpy as np
from tpcp.misc import TypedIterator

from eargait.gait_sequence_detection.gait_sequence_detector import GaitSequenceDetection
from eargait.eargait import EarGait
from eargait.event_detection import DiaoAdaptedEventDetection
from eargait.event_detection.jarchi_event_detection import JarchiEventDetection
from eargait.spatial_params import SpatialParamsRandomForest


# --------------------------------------------------------------
@dataclass
class DummyResult:
    dummy: int = 0


@dataclass # alias for Detected seq Index, pd series (start stop)
class SequenceRow:
    idx: int
    seq: pd.Series
# --------------------------------------------------------------------

class GaitSequenceAnalyzerPipeline:
    """End-to-end wrapper for gait analysis with ear-worn IMUs.

    Combines automatic gait sequence detection (GSD) with EarGait event detection
    and feature extraction. Supports auto-detected sequences from a continuous
    stream or user-provided `[start, end]` samples. Produces per-sequence
    average temporal and spatial parameters.

    Parameters
    ----------
    sample_rate : int
        IMU sampling rate (Hz).
    strictness : int, optional
        Forwarded to GSD (more conservative detection with higher values).
    min_seq_length : int, optional
        Minimum sequence length in samples for GSD.
    event_detection_algorithm : type, optional
        EarGait-compatible event detector class (default: DiaoAdaptedEventDetection).
    spatial_parameter_method : type, optional
        EarGait-compatible spatial estimator class (default: SpatialParamsRandomForest).

    Attributes
    ----------
    auto_sequence_list_ : pd.DataFrame
    manual_sequence_list_ : pd.DataFrame
    auto_average_params_ : pd.DataFrame
    manual_average_params_ : pd.DataFrame

    Methods
    -------
    compute(data, activity)
        Auto mode: run GSD, then EarGait per detected sequence.
    compute_predef_seq(data, sequences)
        Manual mode: validate sequences, then run EarGait per sequence.
    get_combined_df(parameters=None, mode='auto')
        Concatenate `start/end` with selected parameter columns.

    Notes
    -----
    - Indices in `start/end` are samples (not timestamps).
    - Short sequences are fragile for downstream processing:
        <175 samples (~3.50 s @ 50 Hz): may be skipped (too few IC/TC).
        175–199 samples (~3.50–3.99 s @ 50 Hz): features may be NaN.
      This is a current E2E limitation and is projected to be refined.
    """
    sample_rate: int
    strictness: int
    min_seq_length: int
    auto_sequence_list_: pd.DataFrame
    manual_sequence_list_: pd.DataFrame
    auto_average_params_: pd.DataFrame
    manual_average_params_: pd.DataFrame
    combined_df_: pd.DataFrame

    def __init__(self, sample_rate: int,
                 strictness: int = 0,
                 min_seq_length: int = 1,
                 event_detection_algorithm: type = DiaoAdaptedEventDetection,
                 spatial_parameter_method: type = SpatialParamsRandomForest,):

        if sample_rate is None or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer.")

        if strictness < 0:
            raise ValueError("strictness must be zero or positive integer.")

        if min_seq_length < 1:
            raise ValueError("min_seq_length must be greater than or equal to 1.")

        self.sample_rate = sample_rate
        self.strictness = strictness
        self.min_seq_length = min_seq_length
        self.event_detection_algorithm = event_detection_algorithm
        self.spatial_parameter_method = spatial_parameter_method

    def _assert_data_valid(self, data: pd.DataFrame) -> None:
        assert isinstance(data, pd.DataFrame), f"Data must be a pandas DataFrame, got {type(data)}."
        assert len(data) > 0, "Data is empty, must contain at least one row."
        assert data.apply(pd.to_numeric, errors='coerce').notna().all().all(), \
            "All DataFrame entries must be numeric."

    def assert_data_correctness(self, data: pd.DataFrame, activity: Union[str, List[str]]) -> None:
        self._assert_data_valid(data)
        if not isinstance(activity, (str, list)):
            raise ValueError("activity must be a string or list of strings.")
        if isinstance(activity, list) and not all(isinstance(a, str) for a in activity):
            raise ValueError("All elements in activity list must be strings.")

    def compute(self, data: pd.DataFrame, activity: Union[str, List[str]]) -> "GaitSequenceAnalyzerPipeline":
        self.assert_data_correctness(data, activity)
        gsd = self._create_gsd()
        gsd.detect(data, activity=activity)
        self.auto_sequence_list_ = gsd.sequence_list_
        self.auto_sequence_list_.index = [f"seq_id_{i}" for i in range(len(self.auto_sequence_list_))]
        self.auto_average_params_ = self.run_eargait(data, self.auto_sequence_list_)
        return self


    def assert_predef_seq_correctness(self,sequences: pd.DataFrame, data: pd.DataFrame) -> None:
        self._assert_data_valid(data)
        assert isinstance(sequences, pd.DataFrame), f"Self defined sequences must be in a DataFrame format, got {type(sequences)}"
        required_columns = ["start", "end"]
        assert list(sequences.columns) == required_columns, \
            f"Columns mismatch. Expected {required_columns}, got {list(sequences.columns)}."
        assert sequences.map(lambda x: isinstance(x, (int, np.integer))).all().all(), \
            "All entries in sequences must be integers."
        assert len(sequences) > 0, "At least one sequence is required."
        sequences_sorted = sequences.sort_values(by="start").reset_index(drop=True)
        overlaps = sequences_sorted["start"].iloc[1:].values < sequences_sorted["end"].iloc[:-1].values
        if overlaps.any():
            warnings.warn(
                "Detected overlapping sequences in manual input. "
                "Consider joining or correcting overlapping sequences for more accurate analysis.",
                UserWarning
            )
        if (sequences["start"] >= sequences["end"]).any():
            raise ValueError("Each sequence must have 'start' < 'end'.")
        if hasattr(self, 'sample_rate') and self.sample_rate is not None:
            min_length_in_samples = 3 * self.sample_rate
            sequence_lengths = sequences['end'] - sequences['start']
            if (sequence_lengths < min_length_in_samples).any():
                raise ValueError(
                    "All manually defined sequences must be at least 3 seconds in length."
                )
        if not (sequences['start'].isin(range(len(data))) & sequences['end'].isin(range(len(data)))).all():
            raise ValueError(
                f"Invalid 'start' indices:\n{sequences[~sequences['start'].isin(range(len(data)))][['start']]}\n"
                f"Invalid 'end' indices:\n{sequences[~sequences['end'].isin(range(len(data)))][['end']]}"
            )

        lengths = (sequences["end"] - sequences["start"]).astype(int)
        secs = (lengths / self.sample_rate).round(2)

        too_short_fail = lengths < 175
        too_short_partial = (lengths >= 175) & (lengths < 200)

        if too_short_fail.any():
            warnings.warn(
                "Some sequences are too short (<175 samples ≈ 3.50 s @ 50 Hz )"
                "This can cause downstream processing problems in Eargait in this End2End approach. "
                "We are assuming problems with sufficient IC/ TC events so the Process might encounter abortion."
                "This is a current Limitation of the E2E approach. exact cause and solution still unclear; "
                "projected to be worked on.",
                UserWarning,
            )

        if too_short_partial.any():
            warnings.warn(
                "Some sequences are too short (<200 samples ≈ 4.00 s @ 50 Hz )"
                "This can cause downstream processing problems in Eargait in this End2End approach. "
                "We are assuming problems with sufficient IC/ TC events so the Process might encounter NaN values in "
                "the gait parameter list.",
                UserWarning,
            )

    def compute_predef_seq(self, data: pd.DataFrame, sequences: pd.DataFrame) -> "GaitSequenceAnalyzerPipeline":
        self.assert_predef_seq_correctness(sequences, data)
        self.manual_sequence_list_ = sequences.copy()
        self.manual_sequence_list_.index = [f"seq_id_{i}" for i in range(len(self.manual_sequence_list_))]
        self.manual_average_params_ = self.run_eargait(data, sequences)
        return self

    def get_combined_df(self, parameters: Optional[List[str]] = None, mode: str="auto") -> pd.DataFrame:
        if mode == "auto":
            seq_df = self.auto_sequence_list_
            param_df = self.auto_average_params_
        elif mode == "manual":
            seq_df = self.manual_sequence_list_
            param_df = self.manual_average_params_
        else:
            raise ValueError(
                "Mode Should match the choosen compute method, auto for compute and manual for compute_predef_seq"
            )

        if parameters is None:
            return pd.concat(
                [
                    seq_df[['start', 'end']],
                    param_df
                ],
                axis=1
            )
        return pd.concat(
            [
                seq_df[['start', 'end']],
                param_df[parameters]
            ],
            axis=1
        )

    def _create_gsd(self) -> GaitSequenceDetection:
        return GaitSequenceDetection(
            sample_rate=self.sample_rate,
            strictness=self.strictness,
            minimum_seq_length=self.min_seq_length,
        )

    def create_eargait(self) -> EarGait:
        event_detection = self.event_detection_algorithm(sample_rate_hz=self.sample_rate, window_length=self.sample_rate)
        spatial_method = self.spatial_parameter_method(sample_rate_hz=self.sample_rate)
        return EarGait(
            sample_rate_hz=self.sample_rate,
            event_detection_method=event_detection,
            spatial_params_method=spatial_method,
            bool_use_event_list_consistent=True,
        )

    def run_eargait(self, data: pd.DataFrame,
                    sequence_list: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
                    ) -> pd.DataFrame:

        if not isinstance(sequence_list, pd.DataFrame):
            raise TypeError(f"Invalid type for sequence_list: expected pd.DataFrame, but got {type(sequence_list)}. "
                            f" Give only give single Sensor/Side Data as Input.")
        if sequence_list.empty:
            raise ValueError(
                "The provided sequence list is empty. \n"
                "Please provide a non empty Dataframe if manual Sequences are entered. \n"
                "If GSD (auto) is used make sure the examined activity is in the provided dataset."
            )

        seq_df = sequence_list

        # IMU data make, sure it is continuus df
        data_df = data if not isinstance(data, dict) else next(iter(data.values()))

        lengths = (seq_df["end"] - seq_df["start"]).astype(int)

        too_short_fail = lengths < 175
        too_short_partial = (lengths >= 175) & (lengths < 200)

        secs = (lengths / self.sample_rate).round(2)

        if too_short_fail.any():
            warnings.warn(
                "Some sequences are too short (<175 samples ≈ 3.50 s @ 50 Hz )"
                "This can cause downstream processing problems in Eargait in this End2End approach. "
                "We are assuming problems with sufficient IC/ TC events so the Process might encounter abortion."
                "This is a current Limitation of the E2E approach. exact cause and solution still unclear; "
                "projected to be worked on.",
                UserWarning,
            )

        if too_short_partial.any():
            warnings.warn(
                "Some sequences are too short (<200 samples ≈ 4.00 s @ 50 Hz )"
                "This can cause downstream processing problems in Eargait in this End2End approach. "
                "We are assuming problems with sufficient IC/ TC events so the Process might encounter NaN values in "
                "the gait parameter list.",
                UserWarning,
            )


        # Process each detected sequence row functionally
        iterator = TypedIterator(SequenceRow, None)
        processed_results = []
        for (idx, seq), _ in iterator.iterate(seq_df.iterrows()):
            seq_data = data_df.iloc[seq["start"]:seq["end"]]
            processed_results.append(
                self.compute_current_sequence_features(seq_data)
            )

        # create df out of lists
        parameters_df = pd.DataFrame.from_records(
            processed_results,
            index=[f"seq_id_{i}" for i in range(len(processed_results))]
        )

        return parameters_df

    def compute_current_sequence_features(self, seq_data: pd.DataFrame) -> pd.Series:
        ear_gait = self.create_eargait()
        ear_gait.detect({"single_sensor": seq_data})
        temporal_avg = self._extract_mean(ear_gait.average_temporal_params)
        spatial_avg = self._extract_mean(ear_gait.average_spatial_params)

        # concat temp & spatial
        combined_avg = pd.concat([temporal_avg, spatial_avg])
        return combined_avg

    @staticmethod
    def _extract_mean(avg_result: Union[pd.DataFrame, pd.Series, float, dict]) -> pd.Series:
        if isinstance(avg_result, dict):
            avg_result = avg_result.get("single_sensor")
        if isinstance(avg_result, pd.DataFrame) and "mean" in avg_result.index:
            return avg_result.loc["mean"]
        if not hasattr(avg_result, "index"):
            # if scalar -> series ?  falls none -> trotzdem concatable
            return pd.Series([avg_result], index=["value"])
        return avg_result

