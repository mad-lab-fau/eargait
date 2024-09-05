"""Author: Maximilian Vogel. & Lukas Jahnel."""

import yaml
import pathlib as Path

from eargait.gait_sequence_detection.temp_data_preprocessing.data_loading import get_data_all_participants
from eargait.gait_sequence_detection.temp_data_preprocessing.prepare_data import prepare_participant_data
from eargait.gait_sequence_detection.temp_utils_.helper import get_class_distribution

from eargait.gait_sequence_detection.temp_utils_.oversample import get_weighted_samples, oversample
from eargait.gait_sequence_detection.temp_utils_.standardize import get_standardized_data, get_standardized_scalars


def prepare_dl_model_data(run_config) -> dict:
    """Goal of this modified function is to extract a not saved scalar.
    Prepares and standardizes data for training deep learning models, addressing class imbalance.
    This function transforms participant data into a format suitable for deep learning models. It handles class
    imbalances using specified methods and standardizes the data. Additionally, it extracts scalars for standardization,
    which can be used for consistent data processing.

    Parameters
    ----------
    - data_participants: A list of dictionaries, each containing data for a participant.
    - train_participants_idx: Indices of the participants designated for the training dataset.
    - validation_participants_idx: Indices of the participants designated for the validation dataset.
    - test_participants_idx: Indices of the participants designated for the tests dataset.
    - run_config: Configuration settings for the data preparation process.
    - batch_size: The batch size to be used in the data loaders.

    Returns
    -------
    A dictionary containing:
    - 'weighted_probabilities': Probabilities for each class, used to handle class imbalance.
    - 'scaler': A StandardScaler instance fitted to the training data, useful for data normalization.

    The function processes the data by filtering it based on participant indices, applying oversampling or other
    imbalance handling methods if needed, and normalizing the input data. It ensures that the data fed into the
    model is consistent, standardized, and balanced in terms of class representation.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    print("Starting reconstruction of Scaler")
    if run_config.hz == 50:
        yaml_path = repo_root / "eargait/gait_sequence_detection/pretrained_models/50hz_grav_align/version_0/hparams.yaml"
    elif run_config.hz == 200:
        yaml_path = repo_root / "eargait/gait_sequence_detection/pretrained_models/200hz_grav_align/default87/version_0/hparams.yaml"
    else:
        raise ValueError("Unsupported sample rate. Please use either 50 or 200.")

    # Load YAML File
    with open(yaml_path, "r") as file:
        yaml_data = yaml.safe_load(file)

    if "imbalance_method" in yaml_data:
        run_config.imbalance_method = yaml_data["imbalance_method"]
        print(f"Imbalance Method imported from yaml: {run_config.imbalance_method}")
    else:
        print("Imbalance method not found in YAML. Using default method.")

    # Extract participant IDs directly without adjusting indices
    train_ids = yaml_data["train_participants"]

    data_participants = get_data_all_participants(run_config)
    # transform data to numpy
    train_data = prepare_participant_data(data_participants, train_ids, run_config)

    # select chosen imbalance method
    imbalance_method = run_config.imbalance_method
    weighted_random_sampler = None
    weighted_probabilities = None
    # 1. weighted cross entropy loss
    if imbalance_method == "WCE":
        weighted_probabilities = [(1 / count) * 1000 for count in get_class_distribution(train_data["labels"]).values()]
    # 2. pytorch WeightedRandomSampler (so far worse performance than without)
    elif imbalance_method == "WRS":
        weighted_random_sampler = get_weighted_samples(train_data["labels"])
    # 3. oversample with RandomOverSampler ("ros")
    elif imbalance_method == "ROS":
        train_data = oversample(train_data, run_config.selected_coords, run_config.window_length_in_samples, "ros")
    # 3. oversample with SMOTE "smote"
    elif imbalance_method == "SMOTE":
        train_data = oversample(train_data, run_config.selected_coords, run_config.window_length_in_samples, "smote")

    # normalize input data:
    scalars = get_standardized_scalars(train_data["data"])
    train_data["data"] = get_standardized_data(scalars, train_data["data"])

    return {"weighted_probabilities": weighted_probabilities, "scaler": scalars}
