"""Script for preparing participant data for machine learning and deep learning models."""
import numpy as np
import pandas as pd

from eargait.gait_sequence_detection.temp_config_oder_so.configuration import Config
from eargait.utils.global_variables import LABELS_TO_INT
from eargait.gait_sequence_detection.temp_utils_.helper import concat_participants


def pandas_data_to_numpy(
    dataframe: pd.DataFrame, window_length_in_samples: int, input_channels: int
) -> dict[str, np.ndarray]:
    """Transform sensor data and labels of a pd.Dataframe into numpy arrays.

    :param dataframe: pd.Dataframe with sensor_data and labels
    :param window_length_in_samples: window length in samples
    :param input_channels: number of input channels
    :return: dict with 'data' and 'labels' containing data as numpy arrays
    """
    data = np.empty([len(dataframe), window_length_in_samples, input_channels])
    labels = np.empty(len(dataframe), dtype=int)
    index = 0
    for _, row in dataframe.iterrows():
        current_sample = pd.DataFrame.from_dict(row["sensor_data"]).to_numpy()
        label = LABELS_TO_INT[row["label"]]
        data[index] = current_sample
        labels[index] = label
        index += 1
    return {"data": data, "labels": labels}


def prepare_participant_data(
    data_participants: list[dict], participants_idx: np.ndarray, run_config: Config
) -> dict[str, np.ndarray]:
    """Wrapper function which stacks participant data based on participant indices.

    :param data_participants: data of all participants
    :param participants_idx: indices which belong to the same data set
    :param run_config: the current run config
    :return: dict with 'data' and 'labels' containing data as numpy arrays
    """
    participants = [data_participants[i] for i in participants_idx]
    data = concat_participants(participants)
    data_numpy = pandas_data_to_numpy(
        data,
        window_length_in_samples=run_config.window_length_in_samples,
        input_channels=len(run_config.selected_coords),
    )
    return data_numpy



