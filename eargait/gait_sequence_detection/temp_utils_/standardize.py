"""Author."""
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_standardized_scalars(data: np.ndarray) -> object:
    """Create fitted StandardScalars for each coordinate channel.

    :param data: sensor data to fit a scalar
    :return: fitted standard scalar
    """
    train_data = np.concatenate(data, axis=0)
    scalars = StandardScaler()
    trained_scalar = scalars.fit(train_data)
    return trained_scalar


def get_standardized_data(scalars: StandardScaler, data: np.ndarray) -> np.ndarray:
    """Standardizes sensor data.

    :param scalars: previously fitted scalar
    :param data: data to be transformed
    :return: transformed standardized data
    """
    for i in range(data.shape[0]):
        # transform each window based on the scalar
        standardized_sample = scalars.transform(data[i])
        data[i] = standardized_sample
    return data

