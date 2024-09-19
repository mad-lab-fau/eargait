"""Test the Sequence detector class."""
import pickle

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from pathlib import Path
from eargait.preprocessing import align_gravity_and_convert_ear_to_ebf
from eargait.utils.helper_gaitmap import ValidationError, is_sensor_data
from signialib import Session

from eargait.gait_sequence_detection.gait_sequence_detector import GaitSequenceDetection

MAX_WINDOWS_FOR_SNAPSHOT = 50  # Restricts the Snapshots taken to 50 windows to limit json file size


@pytest.fixture(scope="module")
def setup_paths():
    base_path = Path(__file__).resolve().parent.parent
    yaml_path = base_path.joinpath(
        "eargait", "gait_sequence_detection", "pretrained_models", "50hz_grav_align", "version_0", "hparams.yaml"
    )
    with open(yaml_path, "r") as stream:
        hyperparams = yaml.safe_load(stream)

    sample_rate = hyperparams.get("hz", 50)
    data_path_pkl = base_path.joinpath("tests", "test_data", "data_.pkl")
    data_path_mat = base_path.joinpath("tests", "test_data", "subject01")

    return data_path_pkl, data_path_mat, sample_rate


def print_dict_structure(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}: {type(value).__name__}")
            print_dict_structure(value, indent + 1)
        elif isinstance(value, pd.DataFrame):
            print("  " * indent + f"{key}: DataFrame with shape {value.shape}")
            print("  " * (indent + 1) + "Columns:", list(value.columns))
        else:
            print("  " * indent + f"{key}: {type(value).__name__}")


@pytest.fixture(scope="module")
def load_data(setup_paths):
    data_path_pkl, data_path_mat, sample_rate = setup_paths
    with open(data_path_pkl, "rb") as f:
        data_pkl = pickle.load(f)

    print("Structure of data_pkl:")
    print_dict_structure(data_pkl)

    session = Session.from_folder_path(data_path_mat)
    print(session.info)
    align_calibrate_sess = session.align_calib_resample(resample_rate_hz=50, skip_calibration=True)
    data_mat = align_gravity_and_convert_ear_to_ebf(align_calibrate_sess)

    print("Structure of data_mat:")
    print_dict_structure(data_mat)

    gsd = GaitSequenceDetection(sample_rate=sample_rate)

    return data_pkl, data_mat, gsd


def prepare_data(data):
    data = data.rename(
        columns={
            "x": "acc_x",
            "y": "acc_y",
            "z": "acc_z",
            "hiGyrX": "gyr_x",
            "hiGyrY": "gyr_y",
            "hiGyrZ": "gyr_z",
            "acc_pa": "acc_x",
            "acc_ml": "acc_y",
            "acc_si": "acc_z",
            "gyr_pa": "gyr_x",
            "gyr_ml": "gyr_y",
            "gyr_si": "gyr_z",
        }
    )
    return data[["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]]


def test_dataset_loading(load_data, snapshot):
    data_pkl, data_mat, gsd = load_data

    data_pkl_left = prepare_data(pd.DataFrame(data_pkl["sensor_data"]["left_sensor"]))
    data_pkl_right = prepare_data(pd.DataFrame(data_pkl["sensor_data"]["right_sensor"]))
    data_mat_left = prepare_data(pd.DataFrame(data_mat["left_sensor"]))
    data_mat_right = prepare_data(pd.DataFrame(data_mat["right_sensor"]))

    datasets = {
        "data_pkl_left": data_pkl_left,
        "data_pkl_right": data_pkl_right,
        "data_mat_left": data_mat_left,
        "data_mat_right": data_mat_right,
    }

    combined_results = []

    for name, data in datasets.items():
        try:
            dataset_type = is_sensor_data(data)
            assert dataset_type in ["single", "multi"]
            combined_results.append({"dataset": name, "type": "initial", "result": dataset_type})

            gsd.detect(data, activity="walking")
            dataset_type_detect = "single" if isinstance(gsd.sequence_list_, pd.DataFrame) else "multi"
            assert dataset_type_detect in ["single", "multi"]
            combined_results.append({"dataset": name, "type": "detect", "result": dataset_type_detect})
        except ValidationError as e:
            pytest.fail(f"Validation error in {name}: {e}")

    # Convert combined results to DataFrame for snapshot
    combined_results_df = pd.DataFrame(combined_results)
    snapshot.assert_match(combined_results_df, "combined_dataset_loading_results")


@pytest.mark.parametrize("invalid_activity", ["invalid_activity", "running_fast", "unknown"])
def test_activity_validation(load_data, invalid_activity, snapshot):
    _, data_mat, gsd = load_data
    data_mat_left = prepare_data(pd.DataFrame(data_mat["left_sensor"]))

    with pytest.raises(ValueError) as excinfo:
        gsd.detect(data_mat_left, activity=invalid_activity)

    # Capture the error message and additional context in the snapshot
    error_message = str(excinfo.value)
    expected_error_part = "Invalid List of activities or single activity"

    # Check that the expected part of the error message is in the actual error message
    assert expected_error_part in error_message

    # Convert to DataFrame for snapshot comparison
    snapshot_data = pd.DataFrame([{"invalid_activity": invalid_activity, "error_message": error_message}])

    # Snapshot the DataFrame
    snapshot.assert_match(snapshot_data, "invalid_activity_error")


def test_strictness_and_min_length(load_data, snapshot):
    _, data_mat, gsd = load_data

    data_mat_left = prepare_data(pd.DataFrame(data_mat["left_sensor"]))

    invalid_strictness = -1
    gsd.strictness = invalid_strictness
    with pytest.raises(AssertionError):
        gsd.detect(data_mat_left, activity="walking")
    snapshot.assert_match(pd.DataFrame([{"strictness": gsd.strictness}]), "strictness_error")

    invalid_min_length = 0
    gsd.minimum_seq_length = invalid_min_length
    with pytest.raises(AssertionError):
        gsd.detect(data_mat_left, activity="walking")
    snapshot.assert_match(pd.DataFrame([{"minimum_seq_length": gsd.minimum_seq_length}]), "minimum_length_error")


def test_multi_sensor_handling(load_data, snapshot):
    _, data_mat, gsd = load_data
    gsd.strictness = 0
    gsd.minimum_seq_length = 1
    data_mat_multi = {
        "left_sensor": prepare_data(pd.DataFrame(data_mat["left_sensor"])),
        "right_sensor": prepare_data(pd.DataFrame(data_mat["right_sensor"])),
    }

    gsd.detect(data_mat_multi, activity="walking")
    multi_result_type = "multi" if isinstance(gsd.sequence_list_, dict) else "single"
    assert isinstance(gsd.sequence_list_, dict), "Expected multi-sensor data to result in a dictionary of results"

    data_mat_single = prepare_data(pd.DataFrame(data_mat["left_sensor"]))

    gsd.detect(data_mat_single, activity="walking")
    single_result_type = "single" if isinstance(gsd.sequence_list_, pd.DataFrame) else "multi"
    assert isinstance(gsd.sequence_list_, pd.DataFrame), "Expected single-sensor data to result in DataFrame of results"

    output = pd.DataFrame(
        [
            {"sensor_type": "multi", "result_type": multi_result_type},
            {"sensor_type": "single", "result_type": single_result_type},
        ]
    )

    snapshot.assert_match(output, "sensor_handling_results")


def test_detect_single(load_data, snapshot):
    _, data_mat, gsd = load_data

    # Prepare data: trimming to just acceleration columns
    data_mat_left = prepare_data(pd.DataFrame(data_mat["left_sensor"]))
    data_mat_left = data_mat_left[["acc_x", "acc_y", "acc_z"]]

    # Ensure the model is loaded before testing
    gsd.model_path = gsd._get_model()
    gsd._load_trained_model()

    # Print the model path to verify which model is being used
    print(f"Model path: {gsd.model_path}")

    # Set the activity to a valid one
    gsd.activity = ["walking"]

    # Perform detection
    sequence_mat = gsd._detect_single(data_mat_left)
    assert isinstance(sequence_mat, pd.DataFrame), "Expected result type pd.DataFrame"

    # Convert the shape tuple to a dictionary for snapshot comparison
    shape_dict = {"rows": data_mat_left.shape[0], "columns": data_mat_left.shape[1]}
    snapshot.assert_match(pd.DataFrame([shape_dict]), "data_mat_shape")

    windows_mat = [d for k, d in data_mat_left.groupby(data_mat_left.index // gsd._window_length_samples)]
    windows_mat = [window for window in windows_mat if window.shape == windows_mat[0].shape]
    snapshot.assert_match(pd.DataFrame([{"windows_mat_length": len(windows_mat)}]), "windows_mat_length")

    tensor_windows_mat = torch.tensor(np.stack(windows_mat), dtype=torch.float32)
    if tensor_windows_mat.shape[0] > MAX_WINDOWS_FOR_SNAPSHOT:
        tensor_windows_mat = tensor_windows_mat[:MAX_WINDOWS_FOR_SNAPSHOT]

    # Fix applied here
    tensor_windows_mat_df = pd.DataFrame(tensor_windows_mat.tolist()).reset_index(drop=True)
    tensor_windows_mat_df.columns = tensor_windows_mat_df.columns.astype(str)  # Convert columns to string type

    # Ensure the index is also a string type to match the snapshot format
    tensor_windows_mat_df.index = tensor_windows_mat_df.index.astype(str)

    snapshot.assert_match(tensor_windows_mat_df, "tensor_windows_mat")

    tensor_windows_mat = gsd._standardize_data(tensor_windows_mat)
    if tensor_windows_mat.shape[0] > MAX_WINDOWS_FOR_SNAPSHOT:
        tensor_windows_mat = tensor_windows_mat[:MAX_WINDOWS_FOR_SNAPSHOT]
    standardized_tensor_windows_mat_df = pd.DataFrame(tensor_windows_mat.tolist())
    standardized_tensor_windows_mat_df.columns = standardized_tensor_windows_mat_df.columns.astype(str)
    standardized_tensor_windows_mat_df.index = standardized_tensor_windows_mat_df.index.astype(str)
    snapshot.assert_match(standardized_tensor_windows_mat_df, "standardized_tensor_windows_mat")

    tensor_windows_mat = torch.transpose(tensor_windows_mat, 1, 2)
    if tensor_windows_mat.shape[0] > MAX_WINDOWS_FOR_SNAPSHOT:
        tensor_windows_mat = tensor_windows_mat[:MAX_WINDOWS_FOR_SNAPSHOT]
    transposed_tensor_windows_mat_df = pd.DataFrame(tensor_windows_mat.tolist())
    transposed_tensor_windows_mat_df.columns = transposed_tensor_windows_mat_df.columns.astype(str)
    transposed_tensor_windows_mat_df.index = transposed_tensor_windows_mat_df.index.astype(str)
    snapshot.assert_match(transposed_tensor_windows_mat_df, "transposed_tensor_windows_mat")

    tensor_windows_mat = tensor_windows_mat[:, None, :]
    _, output_mat = gsd._trained_model(tensor_windows_mat)
    output_mat_df = pd.DataFrame(output_mat.tolist())
    output_mat_df.columns = output_mat_df.columns.astype(str)
    output_mat_df.index = output_mat_df.index.astype(str)
    snapshot.assert_match(output_mat_df, "output_mat")

    _, predicted_mat = torch.max(output_mat, 1)
    predicted_mat_df = pd.DataFrame(predicted_mat.tolist())
    predicted_mat_df.columns = predicted_mat_df.columns.astype(str)
    predicted_mat_df.index = predicted_mat_df.index.astype(str)
    snapshot.assert_match(predicted_mat_df, "predicted_mat")

    predictions_df_mat = pd.DataFrame([gsd.labels[i] for i in predicted_mat], columns=["activity"])
    predictions_df_mat["start"] = predictions_df_mat.index * gsd._window_length_samples
    predictions_df_mat["end"] = (predictions_df_mat.index * gsd._window_length_samples) + gsd._window_length_samples
    predictions_df_mat.index = predictions_df_mat.index.astype(str)
    snapshot.assert_match(predictions_df_mat, "predictions_df_mat")

    is_doing_x_mat = predictions_df_mat["activity"].isin(gsd.activity)
    activity_df_mat = predictions_df_mat[is_doing_x_mat].reset_index(drop=True)
    activity_df_mat.index = activity_df_mat.index.astype(str)
    snapshot.assert_match(activity_df_mat, "activity_df_mat")

    difference_mat = activity_df_mat["start"].sub(activity_df_mat["end"].shift())
    sequence_mat = activity_df_mat.groupby(difference_mat.gt(0).cumsum()).agg({"start": "min", "end": "max"})
    sequence_mat.index = sequence_mat.index.astype(str)
    snapshot.assert_match(sequence_mat, "sequence_mat")
    snapshot.assert_match(sequence_mat, "final_sequence_mat")


def test_ensure_strictness(load_data, snapshot):
    _, data_mat, gsd = load_data

    data_mat_left = prepare_data(pd.DataFrame(data_mat["left_sensor"]))
    data_mat_left = data_mat_left[["acc_x", "acc_y", "acc_z"]]

    gsd.model_path = gsd._get_model()
    gsd._load_trained_model()

    gsd.activity = ["walking"]

    gsd.strictness = 0
    seq_mat = gsd._detect_single(data_mat_left)
    original_seq_mat = seq_mat.copy()

    gsd.strictness = 2
    strict_seq_mat = gsd._ensure_strictness(seq_mat)

    original_length_mat = len(original_seq_mat)
    new_length_mat = len(strict_seq_mat)

    assert new_length_mat <= original_length_mat, "Strictness should reduce or maintain the number of sequences"
    assert all(strict_seq_mat["end"] - strict_seq_mat["start"] > 0), "All sequences should have positive length"
    assert all(strict_seq_mat["start"].diff().dropna() >= 0), "Start times should be non-decreasing"
    assert all(strict_seq_mat["end"].diff().dropna() >= 0), "End times should be non-decreasing"
    assert all(
        strict_seq_mat["end"] - strict_seq_mat["start"] >= gsd._window_length_samples
    ), "Sequences should be at least one window length"

    snapshot.assert_match(original_seq_mat, "original_seq_mat")
    snapshot.assert_match(strict_seq_mat, "strict_seq_mat")
    snapshot.assert_match(
        pd.DataFrame([{"original_length": original_length_mat, "new_length": new_length_mat}]), "length_comparison"
    )


def test_ensure_minimum_length(load_data, snapshot):
    _, data_mat, gsd = load_data

    data_mat_left = prepare_data(pd.DataFrame(data_mat["left_sensor"]))
    data_mat_left = data_mat_left[["acc_x", "acc_y", "acc_z"]]

    gsd.model_path = gsd._get_model()
    gsd._load_trained_model()

    gsd.activity = ["walking"]

    gsd.minimum_seq_length = 1
    seq_mat = gsd._detect_single(data_mat_left)
    original_seq_mat = seq_mat.copy()

    gsd.minimum_seq_length = 2
    min_length_seq_mat = gsd._ensure_minimum_length(seq_mat)

    original_length_mat = len(original_seq_mat)
    new_length_mat = len(min_length_seq_mat)

    assert new_length_mat <= original_length_mat, "Minimum length should reduce or maintain the number of sequences"
    assert all(
        min_length_seq_mat["end"] - min_length_seq_mat["start"] >= 2 * gsd._window_length_samples
    ), "All sequences should meet the minimum length"

    snapshot.assert_match(original_seq_mat, "original_seq_mat")
    snapshot.assert_match(min_length_seq_mat, "min_length_seq_mat")
    snapshot.assert_match(
        pd.DataFrame([{"original_length": original_length_mat, "new_length": new_length_mat}]), "length_comparison"
    )


def test_model_loading(load_data, snapshot):
    _, data_mat, gsd = load_data

    model_path = gsd._get_model()
    assert model_path is not None, "Model path is None"

    gsd.model_path = model_path
    gsd._load_trained_model()

    assert gsd._trained_model is not None, "Trained model is None"

    yaml_path = gsd.model_path.joinpath("hparams.yaml")
    with open(yaml_path, "r") as stream:
        hyperparams = yaml.safe_load(stream)
    sample_rate_from_yaml = hyperparams.get("hz", 50)

    assert gsd.sample_rate == sample_rate_from_yaml, (
        f"Sample rate mismatch: expected {sample_rate_from_yaml}, " f"got {gsd.sample_rate}"
    )

    snapshot.assert_match(str(model_path), "model_path")


def test_load_trained_model(load_data, snapshot):
    _, data_mat, gsd = load_data

    gsd.model_path = gsd._get_model()
    gsd._load_trained_model()

    assert gsd._trained_model is not None, "Trained model is None"
    assert gsd._trained_model.freeze, "Model is not frozen"

    yaml_path = gsd.model_path.joinpath("hparams.yaml")
    with open(yaml_path, "rb") as stream:
        hyperparams = yaml.safe_load(stream)

    flat_hyperparams = {
        f"{key}_{sub_key}" if isinstance(value, dict) else key: sub_value if isinstance(value, dict) else value
        for key, value in hyperparams.items()
        for sub_key, sub_value in (value.items() if isinstance(value, dict) else [(None, value)])
    }

    hyperparams_df = pd.DataFrame(list(flat_hyperparams.items()), columns=["key", "value"])
    snapshot.assert_match(hyperparams_df, "hyperparams")

    checkpoint_path = list(gsd.model_path.joinpath("checkpoints").glob("*.ckpt"))
    assert len(checkpoint_path) == 1, "Expected exactly one checkpoint file"
    snapshot.assert_match(str(checkpoint_path[0]), "checkpoint_path")


if __name__ == "__main__":
    pytest.main()
