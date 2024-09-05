"""Functions for data loading, preprocessing, and gravity alignment in HAR tasks."""
import os
import os.path
import re
from typing import Dict, Optional

import numpy as np
import pandas as pd
from eargait.preprocessing.rotations import _get_rot_matrix_default
from eargait.utils.consts import SF_ACC, SF_GYR
from eargait.utils.gravity_alignment import StaticWindowGravityAlignment
from eargait.utils.helper_gaitmap import GRAV_VEC

original_get_static_acc_vector = StaticWindowGravityAlignment._get_static_acc_vector
from eargait.preprocessing.rotations import align_dataset_to_gravity as original_align_dataset_to_gravity
from eargait.utils.gravity_alignment import StaticWindowGravityAlignment, TrimMeanGravityAlignment, find_static_samples
from eargait.utils.helper_gaitmap import rotate_dataset
from pandas import DataFrame

from eargait.gait_sequence_detection.temp_config_oder_so.configuration import Config

from eargait.utils.global_variables import (
    ALL_COLUMNS,
    DEFECT_FILES,
    EXCLUDED_LABELS,
    PARTICIPANTS,
    PERIODIC_THRESHOLD,
    TRANSITION_THRESHOLD,
    TRANSITION_LABELS,
)
from eargait.utils.helper_gaitmap import convert_left_foot_to_fbf, convert_right_foot_to_fbf


def _get_label_of_window(
    label_start_idx: int, window_start: int, window_end: int, label_frame: DataFrame
) -> Optional[str]:
    """Helper_utils function to get the label of an activity window.

    :param label_start_idx: start index of the current activity label
    :param window_start: start index of the current data window
    :param window_end: end index of the current data window
    :param label_frame: data frame which contains the labels with their ranges
    :return: label of the activity window or None if no activity is above the threshold
    """
    labels = []
    cur_end = 0
    while cur_end <= window_end and label_start_idx < len(label_frame):
        cur_label = label_frame.iloc[[label_start_idx]]
        label_start_idx += 1
        cur_end = cur_label["end"].item()
        if cur_label["start"].item() > window_end:
            break
        if cur_label["description"].item() in EXCLUDED_LABELS:
            continue
        labels.append(cur_label)
    if len(labels) == 0:
        return None
    labels_in_window_df = pd.concat(labels)
    label_count = {}
    for _, row in labels_in_window_df.iterrows():
        label_start = row["start"]
        label_end = row["end"]
        label = row["description"]
        # group transition labels
        if label in TRANSITION_LABELS:
            label = "transition"
        if label_start > window_start:
            if label_end > window_end:
                count_label = window_end - label_start
            else:
                count_label = label_end - label_start
        else:
            if label_end > window_end:
                count_label = window_end - window_start
            else:
                count_label = label_end - window_start
        try:
            label_count[label] += count_label
        except KeyError:
            label_count[label] = count_label
    label = max(label_count, key=label_count.get)
    if label == "transition":
        if TRANSITION_THRESHOLD * (window_end - window_start) > label_count[label]:
            return None
    else:
        if PERIODIC_THRESHOLD * (window_end - window_start) > label_count[label]:
            return None
    return label

def new_align_dataset_to_gravity(
    data, sampling_rate, window_length_s, static_signal_th, metric, config, GravALignmentMethodStatic=True, grav_vec=GRAV_VEC
):
    """Aligns the dataset to gravity using an improved method for static acceleration vector extraction.

    Parameters
    ----------
    - data (pd.DataFrame): Input sensor readings.
    - sampling_rate (float): Data sampling rate.
    - window_length_s (float): Window length for identifying static samples.
    - static_signal_th (float): Threshold for static signal detection.
    - metric (str): Metric for evaluating static nature, e.g., "median".
    - grav_vec (array-like): Gravity vector for alignment.

    Returns
    -------
    - pd.DataFrame: Aligned dataset.

    Notes
    -----
    - Temporarily replaces the original static vector extraction with `new_get_static_acc_vector`.
    - Restores the original function post-alignment.
    https://github.com/mad-lab-fau/eargait/blob/master/eargait/utils/helper_gaitmap.py#L1265

    """
    # eargait.utils.helper_gaitmap._get_static_acc_vector = lambda *args: new_get_static_acc_vector(*args, config=config)

    if GravALignmentMethodStatic:
        gravity_alignment_method = StaticWindowGravityAlignment(
            sampling_rate_hz=sampling_rate,
            window_length_s=window_length_s,
            static_signal_th=static_signal_th,
            metric=metric,
        )
    else:
        gravity_alignment_method = TrimMeanGravityAlignment(
            sampling_rate_hz=sampling_rate, trim_mean_prop=0.2, cut_off_freq=1, order=4, gravity=GRAV_VEC
        )
    aligned_data = original_align_dataset_to_gravity(data, gravity_alignment_method)

    # restore org get static function
    # eargait.utils.helper_gaitmap._get_static_acc_vector = original_get_static_acc_vector

    return aligned_data


def convert_HAF_to_ESF_optional_gravity_align(cur_data_file: pd.DataFrame, run_config: Config) -> pd.DataFrame:
    """Convert the dataset from the Hearing Aid Frame (HAF) to the Ear Sensor Frame (ESF)
    with an optional gravity alignment.
    The given dataset is initially in the Hearing Aid Frame, which is tilted due to the placement of the hearing aid.
    This function transforms the data into the Ear Sensor Frame, ensuring the z-axis is approximately aligned with gravity.
    The optional gravity alignment is crucial for evaluating the performance difference between gravity-aligned and non-gravity-aligned deep learning models.

    Parameters
    ---------
    - cur_data_file (pd.DataFrame): Input dataset in the Hearing Aid Frame.

    Returns
    -------
    - pd.DataFrame: Dataset in the Ear Sensor Frame with optional gravity alignment. The returned DataFrame contains 'left_data' and 'right_data' keys.

    Attributes
    -----------
    - USE_GRAVITY_ALIGNMENT (bool): Flag to determine if gravity alignment should be applied.
    - window_length_s (float): Window length for identifying static samples.
    - static_signal_th (float): Threshold for static signal detection.
    - metric (str): Metric used for evaluating static nature, e.g., "median".

    Notes
    -----
    - The function first extracts left and right sensor data and their respective sampling rates.
    - It then renames columns to represent acceleration and gyroscope data.
    - The data is converted from g units to m/s^2.
    - Rotation matrices for D11 sensors are fetched and used to rotate the dataset from HAF to ESF.
    - If gravity alignment is enabled, the dataset is further aligned to gravity using the `new_align_dataset_to_gravity` function.

    """
    window_length_s = 0.7
    static_signal_th = 2.5
    metric = "median"

    # Extract the left and right sensor data and their respective sampling rates
    left_sensor_data_org = cur_data_file["sensor_data"]["left_sensor"][ALL_COLUMNS]
    right_sensor_data_org = cur_data_file["sensor_data"]["right_sensor"][ALL_COLUMNS]
    left_sampling_rate = cur_data_file["sampling_rate_hz"]["left_sensor"]
    right_sampling_rate = cur_data_file["sampling_rate_hz"]["right_sensor"]

    left_sensor_data_HAF = left_sensor_data_org.rename(
        columns={"x": "acc_x", "y": "acc_y", "z": "acc_z", "hiGyrX": "gyr_x", "hiGyrY": "gyr_y", "hiGyrZ": "gyr_z"}
    )
    right_sensor_data_HAF = right_sensor_data_org.rename(
        columns={"x": "acc_x", "y": "acc_y", "z": "acc_z", "hiGyrX": "gyr_x", "hiGyrY": "gyr_y", "hiGyrZ": "gyr_z"}
    )

    # g to m/s2
    g_to_ms2 = GRAV_VEC[-1]

    left_sensor_data_HAF[SF_ACC] = left_sensor_data_HAF[SF_ACC] * g_to_ms2
    right_sensor_data_HAF[SF_ACC] = right_sensor_data_HAF[SF_ACC] * g_to_ms2

    # Rotation from the gravity aligned HAF to ESF
    # Gets Rotation Matrix for D11 Sensor
    rotations = {"left_sensor": _get_rot_matrix_default()["left"], "right_sensor": _get_rot_matrix_default()["right"]}
    # defines dataset for rotation operation
    dataset = {"left_sensor": left_sensor_data_HAF, "right_sensor": right_sensor_data_HAF}
    # rotates specified datasets
    rotated_data_HAF_ESF = rotate_dataset(dataset, rotation=rotations, check_gyr=True)

    left_sensor_data_ESF = rotated_data_HAF_ESF["left_sensor"]
    right_sensor_data_ESF = rotated_data_HAF_ESF["right_sensor"]

    # Gravity Alignment if used else previous data is used
    if run_config.use_gravity_alignment:
        final_left_sensor_data_ESF = new_align_dataset_to_gravity(
            left_sensor_data_ESF, left_sampling_rate, window_length_s, static_signal_th, metric, run_config, True, GRAV_VEC
        )
        final_right_sensor_data_ESF = new_align_dataset_to_gravity(
            right_sensor_data_ESF, right_sampling_rate, window_length_s, static_signal_th, metric, run_config, True, GRAV_VEC
        )
    else:
        final_left_sensor_data_ESF = left_sensor_data_ESF
        final_right_sensor_data_ESF = right_sensor_data_ESF
    return {"left_sensor": final_left_sensor_data_ESF, "right_sensor": final_right_sensor_data_ESF}


def convert_sensor_to_body_frame(cur_data_file: pd.DataFrame, run_config=Config) -> dict[str, pd.DataFrame]:
    """Convert left and right sensor data into body frame.

    :param cur_data_file: input dataframe with left and right sensor data
    :return: dict with keys 'left_data' and 'right_data' containing pd.Dataframe
    of left and right data converted into the body frame

    UPDATE: also converts from Hearing Aid Frame (HAF) to Ear Sensor Frame (ESF) after that a possible gravity alignment of the
    Coordinate System is applied (convert_HAF_to_ESF_optional_gravity_alignment)

    after that conversion from ESF to Ear Body Frame (EBF) as before (convert_left_foot_to_fbf)
    """
    left_data = cur_data_file["sensor_data"]["left_sensor"][ALL_COLUMNS]
    left_data = left_data.rename(
        columns={"x": "acc_x", "y": "acc_y", "z": "acc_z", "hiGyrX": "gyr_x", "hiGyrY": "gyr_y", "hiGyrZ": "gyr_z"}
    )
    right_data = cur_data_file["sensor_data"]["right_sensor"][ALL_COLUMNS]
    right_data = right_data.rename(
        columns={"x": "acc_x", "y": "acc_y", "z": "acc_z", "hiGyrX": "gyr_x", "hiGyrY": "gyr_y", "hiGyrZ": "gyr_z"}
    )

    aligned_data = convert_HAF_to_ESF_optional_gravity_align(cur_data_file, run_config)

    convert_left = convert_left_foot_to_fbf(aligned_data["left_sensor"])
    convert_right = convert_right_foot_to_fbf(aligned_data["right_sensor"])

    convert_left = convert_left.rename(
        columns={
            "acc_pa": "x",
            "acc_ml": "y",
            "acc_si": "z",
            "gyr_pa": "hiGyrX",
            "gyr_ml": "hiGyrY",
            "gyr_si": "hiGyrZ",
        }
    )
    convert_right = convert_right.rename(
        columns={
            "acc_pa": "x",
            "acc_ml": "y",
            "acc_si": "z",
            "gyr_pa": "hiGyrX",
            "gyr_ml": "hiGyrY",
            "gyr_si": "hiGyrZ",
        }
    )
    return {"left_sensor": convert_left, "right_sensor": convert_right}


def load_data(participant: int, data_path: str, run_config: Config) -> dict:
    """Read the corresponding record data set and labels. Returns a pandas dataframes with sensor data and labels.

    :param participant: the ID of the selected participant
    :param data_path: data folder path of the current participant
    :param run_config: the current run config
    :return: dict containing 'id' of the participant and 'data', 'data' contains pandas dataframe with windowed
    'sensor_data' and 'label'
    :raises NotADirectoryError
    :raises FileNotFoundError
    """
    if os.path.isdir(data_path) is False:
        raise NotADirectoryError(f"Folder path {data_path} is not valid.")

    selected_coordinates = run_config.selected_coords
    # get data and label files and sort them according to activity block (AB)
    data_files = [f for f in os.listdir(data_path) if re.match(r"2020.*[0-9].*\.pkl", f)]
    data_files.sort(key=str.split)
    label_files = [f for f in os.listdir(data_path) if re.match(r"ID.*[0-9].*\.csv", f)]
    label_files.sort(key=str.split)
    if (len(label_files) == 0 or len(data_files) == 0) or (len(label_files) != len(data_files)):
        raise FileNotFoundError("No files or data and label files do not match.")

    participant_windows = []
    activity_block = 1
    activity_block_keys = []

    for i in range(len(label_files)):
        label_file = pd.read_csv(
            os.path.join(data_path, label_files[i]),
            usecols=[0, 2, 3, 4],
            names=["sensor", "start", "end", "description"],
            header=0,
        )
        label_frame = label_file[label_file["sensor"] == "left_sensor"][["start", "end", "description"]]
        cur_data_file = pd.read_pickle(os.path.join(data_path, data_files[i]))
        if run_config.body_frame_coords:
            converted_data = convert_sensor_to_body_frame(cur_data_file, run_config)
            cur_sensor_data_left = converted_data["left_sensor"][selected_coordinates]
            cur_sensor_data_right = converted_data["right_sensor"][selected_coordinates]
        else:
            cur_sensor_data_left = cur_data_file["sensor_data"]["left_sensor"][selected_coordinates]
            cur_sensor_data_right = cur_data_file["sensor_data"]["right_sensor"][selected_coordinates]
        activity_block_windows = []
        previous_start = -1
        for label_start_idx, labeled_window in label_frame.iterrows():
            label_start = labeled_window["start"]
            label_end = labeled_window["end"]
            start_cur_window = label_start - (label_start % run_config.step_size)
            # check if the previous window already started at the same index -> this can happen if two labels overlap
            if previous_start == start_cur_window:
                start_cur_window += run_config.step_size
            for start_idx_window in range(start_cur_window, label_end, run_config.step_size):
                cur_label = _get_label_of_window(
                    label_start_idx, start_idx_window, start_idx_window + run_config.window_length, label_frame
                )
                if cur_label is None:
                    previous_start = start_idx_window
                    continue
                if (start_idx_window + run_config.window_length) > len(cur_sensor_data_left) or (
                    start_idx_window + run_config.window_length
                ) > len(cur_sensor_data_right):
                    continue
                window_data_left = cur_sensor_data_left[
                    start_idx_window : start_idx_window + run_config.window_length : run_config.frequency_step
                ]
                window_data_right = cur_sensor_data_right[
                    start_idx_window : start_idx_window + run_config.window_length : run_config.frequency_step
                ]
                window_df_left = pd.DataFrame(
                    data=[[window_data_left.to_dict(), cur_label, "left_sensor"]],
                    columns=["sensor_data", "label", "sensor"],
                )
                activity_block_windows.append(window_df_left)
                if label_files[i] not in DEFECT_FILES:
                    window_df_right = pd.DataFrame(
                        data=[[window_data_right.to_dict(), cur_label, "right_sensor"]],
                        columns=["sensor_data", "label", "sensor"],
                    )
                    activity_block_windows.append(window_df_right)
                previous_start = start_idx_window
        activity_block_keys.append("AB" + str(activity_block))
        activity_block += 1
        if activity_block_windows:
            participant_windows.append(pd.concat(activity_block_windows, ignore_index=True))
    return {"id": participant, "data": pd.concat(participant_windows, keys=activity_block_keys)}


def get_data_all_participants(run_config: Config) -> list[dict]:
    """Wrapper function which loads the loaded activity windows,
    either directly from storage (if default settings) or by creating it from the raw data

    :param run_config: the current run config
    :return: list of dicts containing 'id' of the participant and 'data', 'data' contains pandas dataframe with windowed
    'sensor_data' and 'label'
    """
    data_participants = []
    default_settings = (
        run_config.window_length_in_ms == 3000
        and run_config.step_size_in_ms == 1500
        and run_config.hz == 200
        and len(run_config.selected_coords) == 3
        and run_config.body_frame_coords is True
    )
    # Default is True -> preprocessed windowed data used

    default_settings = default_settings and os.path.exists(
        os.path.join(
            run_config.data_base_path,
            "HEPdata",
            "data_fd",
            "ID" + str(1).zfill(2),
            "IMU",
            "Labeled",
            "ID" + str(1).zfill(2) + "_labeled_3s_windows_200hz.pkl",
        )
    )

    for participant in range(1, PARTICIPANTS + 1):
        participant_data_path = os.path.join(
            run_config.data_base_path, "HEPdata", "data_fd", "ID" + str(participant).zfill(2), "IMU", "Labeled"
        )

        if default_settings:
            windowed_pkl_file = os.path.join(
                participant_data_path, "ID" + str(participant).zfill(2) + "_labeled_3s_windows_200hz.pkl"
            )

            data = pd.read_pickle(windowed_pkl_file)
            cur_participant = {"id": participant, "data": data}
        else:
            cur_participant = load_data(participant, participant_data_path, run_config)
        data_participants.append(cur_participant)

    return data_participants
