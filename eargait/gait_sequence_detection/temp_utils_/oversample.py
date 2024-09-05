"""Docstring."""
import numpy as np
import torch
from imblearn.over_sampling import SMOTE, RandomOverSampler
from torch.utils.data import WeightedRandomSampler

from eargait.utils.global_variables import SEED
from eargait.gait_sequence_detection.temp_utils_.helper import get_class_distribution


def oversample(data: dict, selected_coords: list[str], window_length: int, mode: str = "smote") -> dict:
    """Oversample a given set of data.

    :param data: data and labels
    :param selected_coords: currently used coordinates
    :param window_length: window length in samples
    :param mode: oversampling method 'smote' or 'ros'
    :return: dict of oversamples data with 'data' and 'labels'
    """
    input_channels = len(selected_coords)
    sensor_data = data["data"]
    labels = data["labels"]
    # cast data to numpy 1D list for oversampling
    flattened_length = input_channels * window_length
    train_data = np.zeros([len(sensor_data), flattened_length])
    for i in range(len(sensor_data)):
        train_data[i] = sensor_data[i].flatten()

    # oversample the data
    if mode == "ros":
        ros = RandomOverSampler(random_state=SEED)
        X, y = ros.fit_resample(train_data, labels)
    elif mode == "smote":
        smote = SMOTE(random_state=SEED)
        X, y = smote.fit_resample(train_data, labels)
    else:
        raise ValueError("mode has to be either ros or smote!")

    # restore the data in previous format
    oversampled_length = len(X)
    sensor_data_oversampled = np.zeros([oversampled_length, window_length, input_channels])
    for i, entry in enumerate(X):
        sensor_data_oversampled[i] = np.array(entry.reshape((window_length, input_channels)))
    return {"data": sensor_data_oversampled, "labels": y}


def get_weighted_samples(labels: np.ndarray) -> WeightedRandomSampler:
    """Create WeightedRandomSampler.

    :param labels: labels of the current dataset
    :return: WeightedRandomSampler based on the given labels
    """
    target_list = torch.Tensor([label_id for label_id in labels]).long()
    class_count = [i for i in get_class_distribution(labels).values()]
    class_weights = 1.0 / torch.tensor(class_count, dtype=torch.float)
    class_weights_all = class_weights[target_list]
    return WeightedRandomSampler(weights=class_weights_all, num_samples=len(class_weights_all), replacement=True)
