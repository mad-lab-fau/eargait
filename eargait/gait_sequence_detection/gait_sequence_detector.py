"""A gait sequence detection algorithm for ear-worn IMUs."""
from pathlib import Path
from typing import Dict, Hashable, List, TypeVar, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tpcp import Algorithm

from eargait.gait_sequence_detection.har_predictor import HARPredictor
from eargait.utils.consts import LABELS
from eargait.utils.helper_gaitmap import SensorData, is_sensor_data
from eargait.utils.helpers import get_standardized_data

Self = TypeVar("Self", bound="GaitSequenceDetection")
repo_root = Path(__file__).resolve().parent.parent.parent


class GaitSequenceDetection(Algorithm):
    """Find gait events in the IMU raw signal using a pre-trained DL HAR model.

    The model was trained on a fixed window length of 3s.
    A pre-trained model exists for 50Hz and 200hz.

    Parameters
    ----------
    sample_rate
        The sample rate of the data
    strictness
        Determines the size of the gap (in number of windows) at which two consecutive sequences are linked
        together to a single sequence.
    minimum_seq_length
        Determines the minimum length of a sequence (in windows). Needs to be >= 1.


    Attributes
    ----------
    sequence_list_
        The result of the `detect` method holding all gait sequences with their start and end samples. Formatted
        as pandas DataFrame or for MultiSensorData dicts with pandas DataFrames


    Other Parameters
    ----------------
    data
        The data passed to the `detect` method.
    activity
        The sampling rate of the data

    Examples
    --------
    Find sequences of gait in sensor signal

    >>> gsd = GaitSequenceDetection()
    >>> gsd = gsd.detect(data, "walking")
    >>> gsd.sequence_list_

    """

    sample_rate: int
    strictness: int
    minimum_seq_length: int

    model_path: Path
    labels = LABELS
    _trained_model: None
    _window_length_samples: int

    sequence_list_: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    activity_df: pd.DataFrame = None

    data: SensorData
    activity: str

    WINDOW_LENGTH = 3

    def __init__(
        self,
        sample_rate: int = 50,
        strictness: int = 0,
        minimum_seq_length: int = 1,
        criteria_order: str = "strictness_first",
    ):
        self.sample_rate = sample_rate
        self.strictness = strictness
        self.minimum_seq_length = minimum_seq_length
        self.criteria_order = criteria_order

        # Defaults, possibly overrriden after loading of hyperparamterfile of used Model
        self.selected_coords = ["x", "y", "z"]  # Default coordinates
        self.window_length_in_ms = 3000  # Default window length
        self.step_size_in_ms = 1500  # Default step size
        self.body_frame_coords = False

        self.model_path = self._get_model()
        self._load_trained_model()

        self._window_length_samples = sample_rate * self.WINDOW_LENGTH
        self.activity_df = pd.DataFrame()
        self.data = pd.DataFrame()
        self.activity = ""
        super().__init__()

    def detect(self, data: SensorData, activity: Union[str, List[str]] = "walking") -> Self:
        """Find gait sequences or activity sequence in data.

        Parameters
        ----------
        data
            The data set holding the imu raw data
        activity
            The activity of interest.

        To analyze a group of activities e.g. gait related activities specify a list of those activities.
        now use the function call (Example):
            detect(data, activity=["walking", "jogging", "stairs up", "stairs down]).
        This enables the detection of various movement patterns in one analysis.

        Returns
        -------
        self
            The class instance with all result attributes populated

        Notes
        -----
        Data is a pd.Dataframe that required to following columns: ["acc_x", "acc_y", "acc_z"],
        Where x is the vertical acceleration, y is the ML acceleration (left-right), and z is the forward acceleration.

        """
        self.data = data

        if isinstance(activity, str):
            activity = [activity]
        self.activity = activity

        assert self.strictness >= 0
        assert self.minimum_seq_length >= 1
        if not set(self.activity).issubset(self.labels):
            raise ValueError(
                "\n Invalid List of activities or single activity. Please enter a string or list of strings of "
                "the following activities: "
                "jogging, biking, walking, sitting, lying, jumping, stairs up, stairs down, stand or transition. \n"
            )
        # load model
        self.model_path = self._get_model()
        self._load_trained_model()

        dataset_type = is_sensor_data(data, check_acc=True, check_gyr=False)
        # (dataset_type)
        if dataset_type == "single":
            results = self._detect_single(data)
        else:
            results: Dict[Hashable, Dict[str, pd.DataFrame]] = {}
            for sensor in self.data:
                results[sensor] = self._detect_single(data[sensor])
        self.sequence_list_ = results
        return self

    def plot(self, csv_activity_table=None):
        """Plot accelerometer data and detected sequences."""
        assert self.sequence_list_ is not None
        dataset_type = is_sensor_data(self.data)

        if dataset_type == "single":
            _, ax = plt.subplots()
            self._plot_single(self.data, self.sequence_list_, ax, csv_activity_table)
        else:
            _, axes = plt.subplots(2, 1, sharex=True, sharey=True)
            for idx, sensor in enumerate(self.data):
                self._plot_single(self.data[sensor], self.sequence_list_[sensor], axes[idx], csv_activity_table)

        plt.show()

    def _detect_single(self, data):
        assert is_sensor_data(data, check_gyr=False) == "single"

        if "gyr_x" in data or "gyr_pa" in data:
            drop_columns = ["gyr_x", "gyr_y", "gyr_z"] if "gyr_x" in data else ["gyr_pa", "gyr_ml", "gyr_si"]
            data = data.drop(drop_columns, axis=1)
        if not (
            data.columns.isin(["acc_x", "acc_y", "acc_z"]).any()
            or data.columns.isin(["acc_pa", "acc_ml", "acc_si"]).any()
        ):
            raise KeyError("Columns needed: ['acc_x', 'acc_y', 'acc_z'] or ['acc_pa', 'acc_ml', 'acc_si']")

        # windows
        windows = [d for k, d in data.groupby(data.index // self._window_length_samples)]

        # Remove the last list element, if the length is different from the others -> otherwise tensor cannot be created
        if len(windows[-1]) != len(windows[-2]):
            windows.pop(-1)

        # Create a tensor from the list of dataframes
        tensor_windows = torch.tensor(np.stack(windows), dtype=torch.float32)
        # Standardize data
        tensor_windows = self._standardize_data(tensor_windows)

        # 3. Predictions
        tensor_windows = torch.transpose(tensor_windows, 1, 2)
        # self.tensor_windows = tensor_windows
        tensor_windows = tensor_windows[:, None, :]
        _, output = self._trained_model(tensor_windows)
        _, predicted = torch.max(output, 1)
        # 4. Numbers to class label
        predictions_df = pd.DataFrame([self.labels[i] for i in predicted], columns=["activity"])
        predictions_df["start"] = predictions_df.index * self._window_length_samples
        predictions_df["end"] = (predictions_df.index * self._window_length_samples) + self._window_length_samples
        # self.predictions_df = predictions_df

        # 5. Throw away all classes except requested activity
        is_doing_x = predictions_df["activity"].isin(self.activity)
        activity_df = predictions_df[is_doing_x]
        activity_df = activity_df.reset_index(drop=True)
        self.activity_df = activity_df

        # 6. Connect consecutive windows & create table with start/stop
        difference = activity_df["start"].sub(activity_df["end"].shift())
        sequence = activity_df.groupby(difference.gt(0).cumsum()).agg({"start": "min", "end": "max"})
        # self.sequence = sequence

        if self.criteria_order == "strictness_first":
            if self.strictness != 0:
                sequence = self._ensure_strictness(sequence)
            if self.minimum_seq_length != 1:
                sequence = self._ensure_minimum_length(sequence)
        elif self.criteria_order == "min_length_first":
            if self.minimum_seq_length != 1:
                sequence = self._ensure_minimum_length(sequence)
            if self.strictness != 0:
                sequence = self._ensure_strictness(sequence)

        return sequence

    def _ensure_strictness(self, seq: pd.DataFrame):
        assert (np.arange(len(seq)) != seq.index).sum() == 0
        drop_indices = []
        # Start with the first sequence
        current_index = 0

        for i in range(1, len(seq)):

            if (seq.loc[i, "start"] - seq.loc[current_index, "end"]) <= self._window_length_samples * self.strictness:

                seq.loc[current_index, "end"] = seq.loc[i, "end"]

                drop_indices.append(i)
            else:

                current_index = i

        seq = seq.drop(index=drop_indices)

        return seq.reset_index(drop=True)

    def _ensure_minimum_length(self, seq: pd.DataFrame):
        bool_array = (seq.end - seq.start) >= (self.minimum_seq_length * self._window_length_samples)
        return seq[bool_array].reset_index(drop=True)

    def _get_model(self):
        use_default_model = True  # Default is model with gravity alignment & Conversion to Body Frame
        if self.sample_rate == 50:
            if use_default_model:
                model_path = repo_root.joinpath(
                    "eargait", "gait_sequence_detection", "pretrained_models", "50hz_grav_align", "version_0"
                )
                self.use_gravity_aligned_model = True
            else:
                raise ValueError("No 50hz model available without gravity alignment and conversion to Body Frame.")
        elif self.sample_rate == 200:
            if use_default_model:
                model_path = repo_root.joinpath(
                    "eargait",
                    "gait_sequence_detection",
                    "pretrained_models",
                    "200hz_grav_align",
                    "default87",
                    "version_0",
                )
                self.use_gravity_aligned_model = True
            else:
                model_path = repo_root.joinpath(
                    "eargait",
                    "gait_sequence_detection",
                    "pretrained_models",
                    "200hz_grav_align_off",
                    "default",
                    "version_0",
                )
                self.use_gravity_aligned_model = False
        else:
            raise ValueError("Unsupported sample rate. Please use either 50, or 200.")
        return model_path

    def _standardize_data(self, tensor):
        data_array = tensor.numpy()
        if self.sample_rate == 50:
            scaler_path = repo_root.joinpath(
                "eargait", "gait_sequence_detection", "pretrained_models", "50hz_gravity_aligned_scaler.save"
            )
        elif self.sample_rate == 200:
            scaler_path = repo_root.joinpath(
                "eargait", "gait_sequence_detection", "pretrained_models", "200hz_gravity_aligned_scaler.save"
            )

        else:
            raise ValueError("Unsupported sample rate. Please use either 50 or 200.")
        if Path(scaler_path).exists():
            scalars = joblib.load(scaler_path)
        else:
            raise FileNotFoundError(
                f"Scaler not found at {scaler_path}. Please ensure the pre-trained scaler is available."
            )

        standardized_data = get_standardized_data(scalars, data_array)
        tensor_standardized_data = torch.from_numpy(standardized_data)
        return tensor_standardized_data

    def _load_trained_model(self):
        # Load hyperparams of the trained model
        yaml_path = self.model_path.joinpath("hparams.yaml")
        with open(yaml_path, "rb") as stream:
            hyperparams = yaml.safe_load(stream)

        self.sample_rate = hyperparams.get(
            "hz", self.sample_rate
        )  # either hz value in models .yaml file or default = 50
        self.selected_coords = hyperparams.get("selected_coords", self.selected_coords)
        self.window_length_in_ms = hyperparams.get("window_length_in_ms", self.window_length_in_ms)
        self.step_size_in_ms = hyperparams.get("step_size_in_ms", self.step_size_in_ms)
        self.body_frame_coords = hyperparams.get("body_frame_coords", self.body_frame_coords)

        input_channels = hyperparams["input_channels"]
        checkpoint_path = list(self.model_path.joinpath("checkpoints").glob("*.ckpt"))
        if len(checkpoint_path) != 1:
            raise FileNotFoundError(
                f"Expected one checkpoint, but found {len(checkpoint_path)} at {self.model_path}/checkpoints."
            )

        # Get the trained model
        trained_model = HARPredictor.load_from_checkpoint(
            checkpoint_path[0],
            input_channels=input_channels,
            num_classes=len(LABELS),
        )
        self._trained_model = trained_model
        self._trained_model.freeze()
        return self

    def _plot_single(self, data, seq_list, ax, csv_activity_table=None):
        if "acc_x" in data:
            ax.plot(data[["acc_x", "acc_y", "acc_z"]], label=["acc_x", "acc_y", "acc_z"])
        else:
            ax.plot(data[["acc_pa", "acc_ml", "acc_si"]], label=["acc_pa", "acc_ml", "acc_si"])

        for _, row in seq_list.iterrows():
            ax.axvspan(row.start, row.end, color="blue", alpha=0.5, label="Predicted")

        if csv_activity_table is not None:
            for _, row in csv_activity_table.iterrows():
                ax.axvspan(row.start, row.end, color="red", alpha=0.5, label="Ground Truth")

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
