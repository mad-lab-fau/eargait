"""Functions for data concatenation, class distribution, logging hyperparameters, and logging predictions."""
import numpy as np
import pandas as pd


def concat_participants(data: list[dict]) -> pd.DataFrame:
    """Helper_utils function which concatenates the data of multiple participants.

    :param data: list of participant activity windows
    :return: concatenated dataframe of the given data
    """
    concatenated_data = []
    for participant in data:
        concatenated_data.append(participant["data"])
    return pd.concat(concatenated_data, ignore_index=True)


def get_class_distribution(labels_array: np.ndarray) -> dict:
    """Count the amount of activity windows of each class.

    :param labels_array: numpy array of all activity labels
    :return: dict of labels and their count
    """
    unique, counts = np.unique(labels_array, return_counts=True)
    counted_labels = dict(zip(unique, counts))
    return counted_labels
