"""A set of functions to rotate and transform ear-worn sensor data."""
import warnings
from typing import Dict, Union

import numpy as np
import pandas as pd
from nilspodlib import SyncedSession
from signialib import Session

from eargait.utils.consts import BF_ACC, MAX_VALUES_GRAV_ALIGNED_BF_ACC, MIN_VALUES_GRAV_ALIGNED_BF_ACC, SF_ACC, SF_COLS
from eargait.utils.gravity_alignment import StaticWindowGravityAlignment, TrimMeanGravityAlignment
from eargait.utils.helper_gaitmap import (
    MultiSensorData,
    SensorData,
    convert_to_fbf,
    get_multi_sensor_names,
    is_multi_sensor_data,
    is_sensor_data,
    rotate_dataset,
    rotation_from_angle,
)


def convert_ear_to_esf(session: Union[Session, SyncedSession]) -> SensorData:
    """Convert sensor data from hearing aid frame (haf) into the ear sensor frame (esf).

    Parameters
    ----------
    session :
            Recording session of type SigniaSession

    Return
    ------
    SensorData
            Contains gyroscope and acceleration data in the ear body frame

    Notes
    -----
    This funktion can only be used if session data is in hearing aid frame. A different rotation is chosen if
    firmware version is D12. See user guide, coordinate systems for more information.

    """
    # data into sensor_data (single or multi)
    dataset, gyr_avail = get_ear_multi_sensor_data(session)

    # rotate to sensor frame
    rotation = _get_rotation(session)
    dataset_sf = rotate_dataset(dataset, rotation, check_gyr=gyr_avail)
    is_multi_sensor_data(dataset_sf, frame="sensor", raise_exception=True, check_gyr=gyr_avail)
    # rotate to body frame and return
    return dataset_sf


def convert_esf_to_ebf(datasets: SensorData) -> SensorData:
    """Convert sensor data from ear sensor frame (esf) into the ear body frame (ebf).

    Parameters
    ----------
    datasets:
            Dataset

    Return
    ------
    SensorData
            Contains gyroscope and acceleration data in the ear body frame

    Notes
    -----
    See user guide, coordinate systems for more information.

    """
    if len(datasets) == 1:
        sensor_pos = list(datasets.keys())[0]
        if sensor_pos == "left_sensor":
            return convert_to_fbf(datasets, left=["left_sensor"])
        return convert_to_fbf(datasets, right=["right_sensor"])
    return convert_to_fbf(datasets, left=["left_sensor"], right=["right_sensor"])


def convert_ear_to_ebf(session: Union[Session, SyncedSession]) -> MultiSensorData:
    """Convert sensor data from ear worn sensors into the ear body frame (ebf).

    Parameters
    ----------
    session :
            Recorsing session either of type SigniaSession or Nilspod SyncSession.

    Return
    ------
    MutliSensorData
            Contains gyroscope and acceleration data in the ear body frame

    Notes
    -----
    This funktion can only be used if session data is in hearing aid frame. A different rotation is chosen if
    firmware version is D12. See user guide, coordinate systems for more information.

    """
    dataset_sf = convert_ear_to_esf(session)
    return convert_esf_to_ebf(dataset_sf)


def align_dataset_to_gravity(
    dataset: SensorData,
    gravity_alignment_method: Union[TrimMeanGravityAlignment, StaticWindowGravityAlignment],
) -> SensorData:
    """Align dataset, so that each sensor z-axis (if multiple present in dataset) will be parallel to gravity.

    # ToDo : add more details

    Parameters
    ----------
    dataset : gaitmap.utils.dataset_helper.Sensordata
        dataframe representing a single or multiple sensors.
        In case of multiple sensors a df with MultiIndex columns is expected where the first level is the sensor name
        and the second level the axis names (all sensor frame axis must be present)

    gravity_alignment_method :
        Method to align the dataset to gravity. This can be either TrimMeanGravityAlignment or
        StaticWindowGravityAlignment

    Returns
    -------
    aligned dataset
        This will always be a copy. The original dataframe will not be modified.

    Examples
    --------
    >>> # pd.DataFrame containing one or multiple sensor data streams, each of containing all 3 (or 6) IMU
    ... # axis (acc_x, ..., gyr_z)
    >>> dataset_df = ...
    >>> gravity_alignment_method = TrimMeanGravityAlignment(trim_mean_prop=0.2)
    >>> align_dataset_to_gravity(dataset_df, gravity_alignment_method)
    <copy of dataset with all axis aligned to gravity>

    """
    gravity_alignment_method.align_to_gravity(dataset)
    dataset_aligned = gravity_alignment_method.dataset_aligned_
    return dataset_aligned


def align_gravity_and_convert_ear_to_ebf(
    session: Union[Session, SyncedSession],
    gravity_alignment_method: Union[TrimMeanGravityAlignment, StaticWindowGravityAlignment] = None,
) -> SensorData:
    """Convert sensor data from hearing aid frame into the ear body frame (ebf).

    Parameters
    ----------
    session :
            Recording session either of type SigniaSession or Nilspod SyncSession.

    gravity_alignment_method
            Method to align the dataset to gravity. This can be either TrimMeanGravityAlignment or
            StaticWindowGravityAlignment.
    Return
    ------
    SensorData
            Contains acceleration (and gyroscope) data in the ear body frame

    Notes
    -----
    This function can only be used if session data is in hearing aid frame. A different rotation is chosen if
    firmware version is D12. See user guide, coordinate systems for more information.

    """
    dataset_sf = convert_ear_to_esf(session)

    if not gravity_alignment_method:
        warnings.warn("No gravity alignment method provided. Using MeanTrimGravityAlignment.")
        gravity_alignment_method = TrimMeanGravityAlignment(session.info.sampling_rate_hz[0])

    dataset_aligned = align_dataset_to_gravity(dataset_sf, gravity_alignment_method)
    data_ebf = convert_esf_to_ebf(dataset_aligned)
    _sanity_check_gravity_aligned_data_ebf(data_ebf)
    return data_ebf


def _sanity_check_gravity_aligned_data_ebf(dataset: SensorData):
    dataset_type = is_sensor_data(dataset, check_gyr=False)
    if dataset_type == "single":
        _sanity_check_gravity_aligned_data_ebf_single(dataset)
    else:
        for name in get_multi_sensor_names(dataset):
            _sanity_check_gravity_aligned_data_ebf_single(dataset[name])


def _sanity_check_gravity_aligned_data_ebf_single(dataset):
    mean_acc = dataset[BF_ACC].mean()
    if (mean_acc < pd.Series(MIN_VALUES_GRAV_ALIGNED_BF_ACC)).sum() != 0:
        warnings.warn(
            f"Mean of gravity aligned dataset is {mean_acc} and thus smaller than expected minimum value of "
            f"{MIN_VALUES_GRAV_ALIGNED_BF_ACC}. This might indicate an error in the gravity alignment."
        )
    if (mean_acc > pd.Series(MAX_VALUES_GRAV_ALIGNED_BF_ACC)).sum() != 0:
        warnings.warn(
            f"Mean of gravity aligned dataset is {mean_acc} and thus larger than expected maximum value of "
            f"{MAX_VALUES_GRAV_ALIGNED_BF_ACC}. This might indicate an error in the gravity alignment."
        )


def _get_rotation(session: Union[SyncedSession, Session, str]) -> Dict:
    if isinstance(session, Session) or session == "signia":

        if "D12" in session.info.platform[0] or "D12" in session.info.firmware_version[0]:
            if "BMA400" == session.info.imu_sensor_type[0]:
                rot_matrices = _get_rot_matrix_d12_bma400()
            else:
                rot_matrices = _get_rot_matrix_d12()
        else:
            rot_matrices = _get_rot_matrix_default()

        rot = {}
        for side in session.info.sensor_position:
            side = side.split("_")[1]
            rot[side + "_sensor"] = rot_matrices[side]
    elif isinstance(session, SyncedSession) or session == "nilspod":
        left_rot = rotation_from_angle(np.array([1, 0, 0]), np.deg2rad(-90))
        right_rot = rotation_from_angle(np.array([0, 0, 1]), np.deg2rad(180)) * rotation_from_angle(
            np.array([1, 0, 0]), np.deg2rad(-90)
        )
        rot = {"left_sensor": left_rot, "right_sensor": right_rot}
    return rot


def _get_rot_matrix_default():
    rot_matrices = {}
    rot_matrices["left"] = rotation_from_angle(np.array([0, 0, 1]), np.deg2rad(-180)) * rotation_from_angle(
        np.array([0, 1, 0]), np.deg2rad(-90)
    )
    rot_matrices["right"] = rotation_from_angle(np.array([0, 0, 1]), np.deg2rad(-180)) * rotation_from_angle(
        np.array([0, 1, 0]), np.deg2rad(-90)
    )
    return rot_matrices


def _get_rot_matrix_d12():
    rot_matrices = {}
    rot_matrices["left"] = rotation_from_angle(np.array([1, 0, 0]), np.deg2rad(-90)) * rotation_from_angle(
        np.array([0, 1, 0]), np.deg2rad(-90)
    )
    rot_matrices["right"] = rotation_from_angle(np.array([1, 0, 0]), np.deg2rad(-90)) * rotation_from_angle(
        np.array([0, 1, 0]), np.deg2rad(-90)
    )
    return rot_matrices


def _get_rot_matrix_d12_bma400():
    rot_matrices = {}
    rot_matrices["left"] = rotation_from_angle(np.array([0, 1, 0]), np.deg2rad(90))
    rot_matrices["right"] = rotation_from_angle(np.array([0, 1, 0]), np.deg2rad(90))
    return rot_matrices


def get_ear_multi_sensor_data(session: Union[SyncedSession, Session]) -> Dict:
    if isinstance(session, SyncedSession):
        left = session.get_dataset_by_id("157e").imu_data_as_df()
        right = session.get_dataset_by_id("2541").imu_data_as_df()
        left, right = left[SF_COLS], right[SF_COLS]
        return {"left_sensor": left, "right_sensor": right}
    res = {}
    for dataset in session.datasets:
        pos = dataset.info.sensor_position.split("_")[1]
        if session.gyro[0]:
            res[pos + "_sensor"] = dataset.data_as_df()[SF_COLS]
            gyr_avail = True
        else:
            res[pos + "_sensor"] = dataset.data_as_df()[SF_ACC]
            gyr_avail = False
    return res, gyr_avail
