from typing import Union, Dict, List, Iterator, Type, TypeVar, Generic, Tuple
from dataclasses import dataclass
import pandas as pd
import warnings

from eargait.gait_sequence_detection.gait_sequence_detector import GaitSequenceDetection
from eargait.eargait import EarGait
from eargait.event_detection import DiaoAdaptedEventDetection
from eargait.spatial_params import SpatialParamsRandomForest
from tpcp.misc import TypedIterator
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
    sample_rate: int
    strictness: int
    min_seq_length: int
    auto_sequence_list_: pd.DataFrame
    manual_sequence_list_: pd.DataFrame
    auto_average_params_: pd.DataFrame
    manual_average_params_: pd.DataFrame

    def __init__(self, sample_rate: int,
                 strictness: int = 0,
                 min_seq_length: int = 1,
                 event_detection_algorithm: type = DiaoAdaptedEventDetection,
                 spatial_parameter_method: type = SpatialParamsRandomForest):
        if sample_rate is None:
            raise ValueError("sample_rate must be provided and cannot be None")
        self.sample_rate = sample_rate
        self.strictness = strictness
        self.min_seq_length = min_seq_length
        self.event_detection_algorithm = event_detection_algorithm
        self.spatial_parameter_method = spatial_parameter_method

    def compute(self, data: pd.DataFrame, activity: Union[str, List[str]] = "walking") -> "GaitSequenceAnalyzerPipeline":
        gsd = self._create_gsd()
        gsd.detect(data, activity=activity)
        self.auto_sequence_list_ = gsd.sequence_list_
        self.auto_sequence_list_.index = [f"seq_id_{i}" for i in range(len(self.auto_sequence_list_))]
        self.auto_average_params_ = self.run_eargait(data, self.auto_sequence_list_)
        return self

    def compute_predef_seq(self, data: pd.DataFrame, sequences: pd.DataFrame) -> "GaitSequenceAnalyzerPipeline":
        sequences_sorted = sequences.sort_values(by="start").reset_index(drop=True)
        overlaps = sequences_sorted["start"].iloc[1:].values < sequences_sorted["end"].iloc[:-1].values
        if overlaps.any():
            warnings.warn(
                "Detected overlapping sequences in manual input. "
                "Consider joining or correcting overlapping sequences for more accurate analysis.",
                UserWarning
            )
        self.manual_sequence_list_ = sequences.copy()
        self.manual_sequence_list_.index = [f"seq_id_{i}" for i in range(len(self.manual_sequence_list_))]
        self.manual_average_params_ = self.run_eargait(data, sequences)
        return self

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

# TODO Mit Verscheidenen Probanden Daten probieren, kein Walking, sonderfälle, parameter ausuprobieren
# TODO Fehler dann hier aufschreiben und überelgen ob wir die abgfangen müssen
# TODO Unit Tests schreiben

