"Combines Eargait and Gait Detection without the manual need for Looping"
from typing import Union, Dict, List
import pandas as pd

from eargait.gait_sequence_detection.gait_sequence_detector import GaitSequenceDetection
from eargait.eargait import EarGait
from eargait.event_detection import DiaoAdaptedEventDetection
from eargait.spatial_params import SpatialParamsRandomForest

"""
    - Accept a single continuous IMU data stream (single‐sided).
    - No looping by the user—handle everything inside your object.
    - Internally, call existing “EarGait detection” method or logic.
    - Detect where in the continuous stream the gait sequences are.
    - Enforce or check that the user only passes data for one hearing aid.
    - Throw an error if more than one sensor or channel is present
    - Check data correctness as early as possible (e.g., required columns, sample rate).
    - Return a clear error message if data doesn’t meet requirements.
    - For each detected gait sequence, compute “average” gait parameters (e.g., average step time, stride time) no sub stuff for step level
    - Return a concise result (gait sequence start/stop + average parameters).

    One method call (e.g., my_new_detector.detect(data)).
    Inside, run:
        Gait detection over the entire IMU data.
        Summarize the sequences.
    Store results (e.g., sequence_list_, average_params_) as class attributes.
"""


class Combi: # Namen überlegen -> gpt fragen
    """"""
    sample_rate: int
    strictness: int
    min_seq_length: int

    sequence_list_: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    def __init__(
            self,
            sample_rate: int = 50,
            strictness: int = 0,
            min_seq_length: int = 1,):
        """ Defaults"""
        self.sample_rate = sample_rate
        self.strictness = strictness
        self.min_seq_length = min_seq_length
        #self.average_params_ = pd.DataFrame()

        super().__init__()

    def detect(self, data: pd.DataFrame,activity: Union[str, List[str]] = "walking") -> "Combi":
        """"""
        #self.validate_data(data)  # TODO - only one sided, etc --> error schmeißen
        gsd = self._create_gsd()
        gsd.detect(data, activity=activity)
        self.sequence_list_ = gsd.sequence_list_
        self.average_params_ = self.run_eargait(data, self.sequence_list_)
        # TODO for loop so klein wie möglich -> for loop wenige code / function there
        # iterator object für eargait -> braucht keine for loop
        return self

    def run_eargait(self, data: pd.DataFrame,
                    sequence_list: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
                    ) -> pd.DataFrame:
        """"""
        avg_params = {}


        if isinstance(sequence_list, dict):
            # TODO change since we only want to accept single side but currently with the example dataset ok
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
            seq_data = data_df.iloc[start:end] # Start and end times as before done

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

            # TODO change with above, bc of dict behaviour
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

    #def validate_single_side_data(self):
    #    """"""

    def _create_gsd(self) -> GaitSequenceDetection:
        gsd = GaitSequenceDetection(
            sample_rate=self.sample_rate,
            strictness=self.strictness,
            minimum_seq_length=self.min_seq_length,
        )
        return gsd