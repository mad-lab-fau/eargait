"""A set of functions to rotate and transform ear-worn sensor data."""
from typing import Dict, Union

import numpy as np
from nilspodlib import SyncedSession
from signialib import Session

from eargait.utils.consts import SF_ACC, SF_COLS
from eargait.utils.helper_gaitmap import (
    MultiSensorData,
    SensorData,
    align_dataset_to_gravity,
    convert_to_fbf,
    is_multi_sensor_data,
    rotate_dataset,
    rotation_from_angle,
)


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


def align_gravity_and_convert_ear_to_ebf(session: Union[Session, SyncedSession]) -> SensorData:
    """Convert sensor data from hearing aid frame into the ear body frame (ebf).

    Parameters
    ----------
    session :
            Recording session either of type SigniaSession or Nilspod SyncSession.

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
    aligned = {}
    if len(dataset_sf) == 1:
        sensor_pos = list(dataset_sf.keys())[0]
        aligned[sensor_pos] = align_dataset_to_gravity(dataset_sf[sensor_pos], session.info.sampling_rate_hz[0])
    else:
        aligned = align_dataset_to_gravity(dataset_sf, session.info.sampling_rate_hz[0])
    return convert_esf_to_ebf(aligned)


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


def _get_rotation(session: Union[SyncedSession, Session, str]) -> Dict:
    if isinstance(session, Session) or session == "signia":

        if "D12" in session.info.version_firmware[0]:
            if "BMA400" == session.info.imu_sensor_type[0]:
                rot_matrices = _get_rot_matrix_d12_bma400()
                print("BMA 400 and D12")
            else:
                rot_matrices = _get_rot_matrix_d12()
                print("D12")
        else:
            rot_matrices = _get_rot_matrix_default()
            print("D11, default")

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
