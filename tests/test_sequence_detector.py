"""Test the Sequence detector class."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from signialib import Session

from eargait.gait_sequence_detection.gait_sequence_detector import GaitSequenceDetection
from eargait.preprocessing import align_gravity_and_convert_ear_to_ebf
from eargait.utils.helper_gaitmap import ValidationError, is_sensor_data


class TestGaitSequenceDetector:
    """Test Class for the Gait Sequence Detector:"""

    MAX_WINDOWS_FOR_SNAPSHOT = 50  # Restricts the Snapshots taken to 50 windows to limit json file size

    @classmethod
    def setup_class(cls):
        """Set up class-wide stuff."""
        base_path = Path(__file__).resolve().parent.parent
        yaml_path = base_path.joinpath(
            "eargait", "gait_sequence_detection", "pretrained_models", "50hz_grav_align", "version_0", "hparams.yaml"
        )
        with open(yaml_path, "r") as stream:
            cls.hyperparams = yaml.safe_load(stream)
        cls.sample_rate = cls.hyperparams.get("hz", 50)

        cls.data_path_mat = base_path.joinpath("tests", "test_data", "subject01")

        cls.session = Session.from_folder_path(cls.data_path_mat)
        align_calibrate_sess = cls.session.align_calib_resample(resample_rate_hz=50, skip_calibration=True)
        cls.data_mat = align_gravity_and_convert_ear_to_ebf(align_calibrate_sess)

        cls.gsd = GaitSequenceDetection(sample_rate=cls.sample_rate)

    def prepare_data(self, data):
        return data.rename(
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

    def test_dataset_loading(self, snapshot):
        data_mat_left = self.prepare_data(pd.DataFrame(self.data_mat["left_sensor"]))
        data_mat_right = self.prepare_data(pd.DataFrame(self.data_mat["right_sensor"]))

        datasets = {"data_mat_left": data_mat_left, "data_mat_right": data_mat_right}

        combined_results = []

        for name, data in datasets.items():
            try:
                dataset_type = is_sensor_data(data)
                assert dataset_type in ["single", "multi"]
                combined_results.append({"dataset": name, "type": "initial", "result": dataset_type})

                self.gsd.detect(data, activity="walking")

                dataset_type_detect = "single" if isinstance(self.gsd.sequence_list_, pd.DataFrame) else "multi"
                assert dataset_type_detect in ["single", "multi"]
                combined_results.append({"dataset": name, "type": "detect", "result": dataset_type_detect})

            except ValidationError as e:
                pytest.fail(f"Validation error in {name}: {e}")

        combined_results_df = pd.DataFrame(combined_results)
        snapshot.assert_match(combined_results_df, "combined_dataset_loading_results")

    @pytest.mark.parametrize("invalid_activity", ["invalid_activity", "running_fast", "unknown"])
    def test_activity_validation(self, invalid_activity, snapshot):
        data_mat_left = self.prepare_data(pd.DataFrame(self.data_mat["left_sensor"]))

        with pytest.raises(ValueError) as excinfo:
            self.gsd.detect(data_mat_left, activity=invalid_activity)

        error_message = str(excinfo.value)
        expected_error_part = "Invalid List of activities or single activity"

        assert expected_error_part in error_message

        snapshot_data = pd.DataFrame([{"invalid_activity": invalid_activity, "error_message": error_message}])
        snapshot.assert_match(snapshot_data, "invalid_activity_error")

    def test_strictness_and_min_length(self, snapshot):
        data_mat_left = self.prepare_data(pd.DataFrame(self.data_mat["left_sensor"]))

        invalid_strictness = -1
        self.gsd.strictness = invalid_strictness
        with pytest.raises(AssertionError):
            self.gsd.detect(data_mat_left, activity="walking")
        snapshot.assert_match(pd.DataFrame([{"strictness": self.gsd.strictness}]), "strictness_error")

        invalid_min_length = 0
        self.gsd.minimum_seq_length = invalid_min_length
        with pytest.raises(AssertionError):
            self.gsd.detect(data_mat_left, activity="walking")
        snapshot.assert_match(
            pd.DataFrame([{"minimum_seq_length": self.gsd.minimum_seq_length}]), "minimum_length_error"
        )

    def test_multi_sensor_handling(self, snapshot):
        self.gsd.strictness = 0
        self.gsd.minimum_seq_length = 1
        data_mat_multi = {
            "left_sensor": self.prepare_data(pd.DataFrame(self.data_mat["left_sensor"])),
            "right_sensor": self.prepare_data(pd.DataFrame(self.data_mat["right_sensor"])),
        }

        self.gsd.detect(data_mat_multi, activity="walking")
        multi_result_type = "multi" if isinstance(self.gsd.sequence_list_, dict) else "single"
        assert isinstance(
            self.gsd.sequence_list_, dict
        ), "Expected multi-sensor data to result in a dictionary of results"

        data_mat_single = self.prepare_data(pd.DataFrame(self.data_mat["left_sensor"]))

        self.gsd.detect(data_mat_single, activity="walking")
        single_result_type = "single" if isinstance(self.gsd.sequence_list_, pd.DataFrame) else "multi"
        assert isinstance(
            self.gsd.sequence_list_, pd.DataFrame
        ), "Expected single-sensor data to result in DataFrame of results"

        output = pd.DataFrame(
            [
                {"sensor_type": "multi", "result_type": multi_result_type},
                {"sensor_type": "single", "result_type": single_result_type},
            ]
        )

        snapshot.assert_match(output, "sensor_handling_results")

    def test_detect_single(self, snapshot):
        data_mat_left = self.prepare_data(pd.DataFrame(self.data_mat["left_sensor"]))
        data_mat_left = data_mat_left[["acc_x", "acc_y", "acc_z"]]

        self.gsd.model_path = self.gsd._get_model()
        self.gsd._load_trained_model()

        # print(f"Model path: {self.gsd.model_path}")

        self.gsd.activity = ["walking"]

        sequence_mat = self.gsd._detect_single(data_mat_left)
        assert isinstance(sequence_mat, pd.DataFrame), "Expected result type pd.DataFrame"

        shape_dict = {"rows": data_mat_left.shape[0], "columns": data_mat_left.shape[1]}
        snapshot.assert_match(pd.DataFrame([shape_dict]), "data_mat_shape")

        windows_mat = [d for k, d in data_mat_left.groupby(data_mat_left.index // self.gsd._window_length_samples)]
        windows_mat = [window for window in windows_mat if window.shape == windows_mat[0].shape]
        snapshot.assert_match(pd.DataFrame([{"windows_mat_length": len(windows_mat)}]), "windows_mat_length")

        tensor_windows_mat = torch.tensor(np.stack(windows_mat), dtype=torch.float32)
        if tensor_windows_mat.shape[0] > self.MAX_WINDOWS_FOR_SNAPSHOT:
            tensor_windows_mat = tensor_windows_mat[: self.MAX_WINDOWS_FOR_SNAPSHOT]

        tensor_windows_mat_df = pd.DataFrame(tensor_windows_mat.tolist()).reset_index(drop=True)
        tensor_windows_mat_df.columns = tensor_windows_mat_df.columns.astype(str)

        tensor_windows_mat_df.index = tensor_windows_mat_df.index.astype(str)

        snapshot.assert_match(tensor_windows_mat_df, "tensor_windows_mat")

        tensor_windows_mat = self.gsd._standardize_data(tensor_windows_mat)
        if tensor_windows_mat.shape[0] > self.MAX_WINDOWS_FOR_SNAPSHOT:
            tensor_windows_mat = tensor_windows_mat[: self.MAX_WINDOWS_FOR_SNAPSHOT]
        standardized_tensor_windows_mat_df = pd.DataFrame(tensor_windows_mat.tolist())
        standardized_tensor_windows_mat_df.columns = standardized_tensor_windows_mat_df.columns.astype(str)
        standardized_tensor_windows_mat_df.index = standardized_tensor_windows_mat_df.index.astype(str)
        snapshot.assert_match(standardized_tensor_windows_mat_df, "standardized_tensor_windows_mat")

        tensor_windows_mat = torch.transpose(tensor_windows_mat, 1, 2)
        if tensor_windows_mat.shape[0] > self.MAX_WINDOWS_FOR_SNAPSHOT:
            tensor_windows_mat = tensor_windows_mat[: self.MAX_WINDOWS_FOR_SNAPSHOT]
        transposed_tensor_windows_mat_df = pd.DataFrame(tensor_windows_mat.tolist())
        transposed_tensor_windows_mat_df.columns = transposed_tensor_windows_mat_df.columns.astype(str)
        transposed_tensor_windows_mat_df.index = transposed_tensor_windows_mat_df.index.astype(str)
        snapshot.assert_match(transposed_tensor_windows_mat_df, "transposed_tensor_windows_mat")

        tensor_windows_mat = tensor_windows_mat[:, None, :]
        _, output_mat = self.gsd._trained_model(tensor_windows_mat)

        _, predicted_mat = torch.max(output_mat, 1)
        predicted_mat_df = pd.DataFrame(predicted_mat.tolist())
        predicted_mat_df.columns = predicted_mat_df.columns.astype(str)
        predicted_mat_df.index = predicted_mat_df.index.astype(str)

        predictions_df_mat = pd.DataFrame([self.gsd.labels[i] for i in predicted_mat], columns=["activity"])
        predictions_df_mat["start"] = predictions_df_mat.index * self.gsd._window_length_samples
        predictions_df_mat["end"] = (
            predictions_df_mat.index * self.gsd._window_length_samples
        ) + self.gsd._window_length_samples
        predictions_df_mat.index = predictions_df_mat.index.astype(str)
        snapshot.assert_match(predictions_df_mat, "predictions_df_mat")

        is_doing_x_mat = predictions_df_mat["activity"].isin(self.gsd.activity)
        activity_df_mat = predictions_df_mat[is_doing_x_mat].reset_index(drop=True)
        activity_df_mat.index = activity_df_mat.index.astype(str)
        snapshot.assert_match(activity_df_mat, "activity_df_mat")

        difference_mat = activity_df_mat["start"].sub(activity_df_mat["end"].shift())
        sequence_mat = activity_df_mat.groupby(difference_mat.gt(0).cumsum()).agg({"start": "min", "end": "max"})
        sequence_mat.index = sequence_mat.index.astype(str)
        snapshot.assert_match(sequence_mat, "sequence_mat")
        snapshot.assert_match(sequence_mat, "final_sequence_mat")

    def test_ensure_strictness(self, snapshot):
        data_mat_left = self.prepare_data(pd.DataFrame(self.data_mat["left_sensor"]))
        data_mat_left = data_mat_left[["acc_x", "acc_y", "acc_z"]]

        self.gsd.model_path = self.gsd._get_model()
        self.gsd._load_trained_model()
        self.gsd.activity = ["walking"]

        self.gsd.strictness = 0
        seq_mat = self.gsd._detect_single(data_mat_left)
        original_seq_mat = seq_mat.copy()

        self.gsd.strictness = 2
        strict_seq_mat = self.gsd._ensure_strictness(seq_mat)

        original_length_mat = len(original_seq_mat)
        new_length_mat = len(strict_seq_mat)

        assert new_length_mat <= original_length_mat, "Strictness should reduce or maintain the number of sequences"
        assert all(strict_seq_mat["end"] - strict_seq_mat["start"] > 0), "All sequences should have positive length"
        assert all(strict_seq_mat["start"].diff().dropna() >= 0), "Start times should be non-decreasing"
        assert all(strict_seq_mat["end"].diff().dropna() >= 0), "End times should be non-decreasing"
        assert all(
            strict_seq_mat["end"] - strict_seq_mat["start"] >= self.gsd._window_length_samples
        ), "Sequences should be at least one window length"

        snapshot.assert_match(original_seq_mat, "original_seq_mat")
        snapshot.assert_match(strict_seq_mat, "strict_seq_mat")
        snapshot.assert_match(
            pd.DataFrame([{"original_length": original_length_mat, "new_length": new_length_mat}]), "length_comparison"
        )

    def test_ensure_minimum_length(self, snapshot):
        data_mat_left = self.prepare_data(pd.DataFrame(self.data_mat["left_sensor"]))
        data_mat_left = data_mat_left[["acc_x", "acc_y", "acc_z"]]

        self.gsd.model_path = self.gsd._get_model()
        self.gsd._load_trained_model()

        self.gsd.activity = ["walking"]

        self.gsd.minimum_seq_length = 1
        seq_mat = self.gsd._detect_single(data_mat_left)
        original_seq_mat = seq_mat.copy()

        self.gsd.minimum_seq_length = 2
        min_length_seq_mat = self.gsd._ensure_minimum_length(seq_mat)

        original_length_mat = len(original_seq_mat)
        new_length_mat = len(min_length_seq_mat)

        assert new_length_mat <= original_length_mat, "Minimum length should reduce or maintain the number of sequences"
        assert all(
            min_length_seq_mat["end"] - min_length_seq_mat["start"] >= 2 * self.gsd._window_length_samples
        ), "All sequences should meet the minimum length"

        snapshot.assert_match(original_seq_mat, "original_seq_mat")
        snapshot.assert_match(min_length_seq_mat, "min_length_seq_mat")
        snapshot.assert_match(
            pd.DataFrame([{"original_length": original_length_mat, "new_length": new_length_mat}]), "length_comparison"
        )

    def test_model_loading(self, snapshot):
        model_path = self.gsd._get_model()
        assert model_path is not None, "Model path is None"

        self.gsd.model_path = model_path
        self.gsd._load_trained_model()

        assert self.gsd._trained_model is not None, "Trained model is None"

        yaml_path = self.gsd.model_path.joinpath("hparams.yaml")
        with open(yaml_path, "r") as stream:
            hyperparams = yaml.safe_load(stream)
        sample_rate_from_yaml = hyperparams.get("hz", 50)

        assert self.gsd.sample_rate == sample_rate_from_yaml, (
            f"Sample rate mismatch: expected {sample_rate_from_yaml}, " f"got {self.gsd.sample_rate}"
        )
        snapshot.assert_match(pd.DataFrame([{"path": list(Path(model_path).parts[-5::])}]), "model_path")

    def test_load_trained_model(self, snapshot):
        self.gsd.model_path = self.gsd._get_model()
        self.gsd._load_trained_model()

        assert self.gsd._trained_model is not None, "Trained model is None"
        assert self.gsd._trained_model.freeze, "Model is not frozen"

        yaml_path = self.gsd.model_path.joinpath("hparams.yaml")
        with open(yaml_path, "rb") as stream:
            hyperparams = yaml.safe_load(stream)

        flat_hyperparams = {
            f"{key}_{sub_key}" if isinstance(value, dict) else key: sub_value if isinstance(value, dict) else value
            for key, value in hyperparams.items()
            for sub_key, sub_value in (value.items() if isinstance(value, dict) else [(None, value)])
        }

        hyperparams_df = pd.DataFrame(list(flat_hyperparams.items()), columns=["key", "value"])
        snapshot.assert_match(hyperparams_df, "hyperparams")

        checkpoint_path = list(self.gsd.model_path.joinpath("checkpoints").glob("*.ckpt"))
        assert len(checkpoint_path) == 1, "Expected exactly one checkpoint file"
        snapshot.assert_match(pd.DataFrame([{"path": list(Path(checkpoint_path[0]).parts[-7::])}]), "checkpoint_path")
