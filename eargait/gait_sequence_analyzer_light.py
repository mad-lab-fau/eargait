"""Combines Eargait and Gait Detection without the manual need for Looping"""
from typing import Union, Dict, List
import pandas as pd

from eargait.gait_sequence_detection.gait_sequence_detector import GaitSequenceDetection
from eargait.eargait import EarGait
from eargait.event_detection import DiaoAdaptedEventDetection
from eargait.spatial_params import SpatialParamsRandomForest


class GaitSequenceAnalyzerLight:
    """One-call combo of gait sequence detection (GSD) → EarGait.

    Runs GSD on a continuous single-sensor IMU stream and applies EarGait per
    detected segment to return per-sequence average temporal/spatial parameters.

    Parameters
    ----------
    sample_rate : int, default 50
    strictness : int, default 0
    min_seq_length : int, default 1
        All values are in samples.

    Attributes
    ----------
    sequence_list_ : pd.DataFrame
    average_params_ : pd.DataFrame

    Notes
    -----
    - Auto only (no manual `[start, end]` input).
    - Fixed methods: DiaoAdaptedEventDetection + SpatialParamsRandomForest.
    - Light validation; expects a single-sensor stream.
    - Prefer `GaitSequenceAnalyzerPipeline` unless you specifically need this
      lightweight one-call wrapper—the pipeline covers this and adds manual mode,
      stronger validation, and swappable methods.

    """

    sample_rate: int
    strictness: int
    min_seq_length: int

    sequence_list_: Union[pd.DataFrame, Dict[str, pd.DataFrame]]

    def __init__(
            self,
            sample_rate: int = 50,
            strictness: int = 0,
            min_seq_length: int = 1,):
        """Defaults"""
        self.sample_rate = sample_rate
        self.strictness = strictness
        self.min_seq_length = min_seq_length
        # self.average_params_ = pd.DataFrame()

        super().__init__()

    def detect(self, data: pd.DataFrame, activity: Union[str, List[str]] = "walking") -> "GaitSequenceAnalyzerLight":
        """"""
        # self.validate_data(data)
        gsd = self._create_gsd()
        gsd.detect(data, activity=activity)
        self.sequence_list_ = gsd.sequence_list_
        self.average_params_ = self.run_eargait(data, self.sequence_list_)
        return self

    def run_eargait(self, data: pd.DataFrame,
                    sequence_list: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
                    ) -> pd.DataFrame:
        """"""
        avg_params = {}

        if isinstance(sequence_list, dict):
            if "left_sensor" in sequence_list:
                seq_df = sequence_list["left_sensor"]
            elif "right_sensor" in sequence_list:
                seq_df = sequence_list["right_sensor"]
            else:
                seq_df = next(iter(sequence_list.values()))
        else:
            seq_df = sequence_list

        data_df = data if not isinstance(data, dict) else next(iter(data.values()))

        for idx, seq in seq_df.iterrows():
            start, end = seq["start"], seq["end"]
            seq_data = data_df.iloc[start:end]

            # run eargait on the sequence for sliced datastream
            event_detection = DiaoAdaptedEventDetection(sample_rate_hz=self.sample_rate, window_length=self.sample_rate)
            spatial_method = SpatialParamsRandomForest(sample_rate_hz=self.sample_rate)
            ear_gait = EarGait(
                sample_rate_hz=self.sample_rate,
                event_detection_method=event_detection,
                spatial_params_method=spatial_method,
                bool_use_event_list_consistent=True,
            )
            ear_gait.detect({"single_sensor": seq_data})

            temporal_avg = ear_gait.average_temporal_params
            spatial_avg = ear_gait.average_spatial_params

            if isinstance(temporal_avg, dict):
                temporal_avg = temporal_avg.get("single_sensor")
            if isinstance(spatial_avg, dict):
                spatial_avg = spatial_avg.get("single_sensor")

            # Hier noch per step, in sequence --> correct way, check NO NO NO
            if isinstance(temporal_avg, pd.DataFrame) and temporal_avg.ndim > 1:
                temporal_avg = temporal_avg.mean(axis=0)
            if isinstance(spatial_avg, pd.DataFrame) and spatial_avg.ndim > 1:
                spatial_avg = spatial_avg.mean(axis=0)

            # concat
            combined_avg = pd.concat([temporal_avg, spatial_avg])
            avg_params[f"Sequence_{idx + 1}"] = combined_avg
        #  make df
        result_df = pd.DataFrame(avg_params)
        return result_df

    # def validate_single_side_data(self):

    def _create_gsd(self) -> GaitSequenceDetection:
        gsd = GaitSequenceDetection(
            sample_rate=self.sample_rate,
            strictness=self.strictness,
            minimum_seq_length=self.min_seq_length,
        )
        return gsd
