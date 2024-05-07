"""Class spatial parameter using a Random Forest model.

The final 'production' model was created by following the guidlines in
https://tpcp.readthedocs.io/en/latest/guides/algorithm_evaluation.html.
Grid Search was performed using a 5 fold grouped cross validation, final model was refitted on all data.
"""
import pickle
from pathlib import Path
from typing import Dict, Optional, TypeVar, Union

import pandas as pd
from sklearn.pipeline import Pipeline

from eargait.spatial_params.feature_extractor import FeatureExtractor, FeatureExtractorDemographics
from eargait.spatial_params.spatial_params_base import SpatialParamsBase
from eargait.utils.helper_datatype import EventList
from eargait.utils.helper_gaitmap import SensorData

HERE = Path(__file__).parent

Self = TypeVar("Self", bound="SpatialParamsRandomForest")


class SpatialParamsRandomForest(SpatialParamsBase):
    """Spatial Parameter Estimation Class using a pretrained Random Forest."""

    sample_rate_hz: int
    grav_alignment_method: Optional[str]  # trim or static
    model_path: Path

    model: Pipeline
    extractor: FeatureExtractor

    step_length_: pd.DataFrame

    def __init__(
        self,
        sample_rate_hz,
        grav_alignment_method="static",
        model_path=None,
    ):
        self.sample_rate_hz = sample_rate_hz
        self.grav_alignment_method = grav_alignment_method
        self.model_path = model_path
        self.model = None
        self.extractor = None
        super().__init__()

    def estimate(self, data: SensorData, event_list: EventList) -> Union[Dict, pd.DataFrame]:
        self._check_subjects_characteristics()
        self._load_model()
        self._load_feature_extractor()

        features_rf = self.extractor.get_features(data, event_list)
        # features_rf = features.drop(features.columns.difference(self.features_model), 1)
        features_rf_not_nan = features_rf.dropna(axis=0)
        step_length = self.model.predict(features_rf_not_nan)
        step_length_series = pd.Series(name="step_length", index=event_list.index, dtype="float64")
        step_length_series.loc[step_length_series.index.isin(features_rf_not_nan.index)] = step_length
        step_length_series = step_length_series.shift(1)
        stride_length_series = step_length_series[1::] + step_length_series[0:-1]
        stride_length_series.name = "stride_length"
        spatial = pd.concat([step_length_series, stride_length_series], axis=1)
        spatial = spatial.assign(side=event_list.side)
        self.step_length_ = spatial
        return spatial

    def _check_subjects_characteristics(self):
        # only necessary for random forest with demographics
        pass

    def _load_model(self):
        if not self.model_path:
            # find model path

            if self.grav_alignment_method == "trim":
                self.model_path = HERE.joinpath(
                    "trained_models",
                    "ml_randomforest",
                    "2023_12_rf_" + str(self.sample_rate_hz) + "hz_regressor_trimGravityAlignment.pkl",
                )
            elif self.grav_alignment_method == "static":
                self.model_path = HERE.joinpath(
                    "trained_models",
                    "ml_randomforest",
                    "2024_05_rf_" + str(self.sample_rate_hz) + "hz_regressor_staticGravityAlignment.pkl",
                )
            else:
                raise ValueError(
                    f"grav_alignment_method must be in ['trim', 'static']. Is: {self.grav_alignment_method}"
                )

        if not self.model_path.is_file():
            potential_models = [
                int(x.stem.split("hz")[0].split("_")[-1])
                for x in HERE.joinpath("trained_models", "ml_randomforest").iterdir()
                if (x.is_file() and "hz" in str(x))
            ]
            raise ValueError(
                f"No model available for sample rate {self.sample_rate_hz}Hz. "
                f"Model are given for: {set(potential_models)} Hz"
            )

        with open(self.model_path, "rb") as handler:
            self.model = pickle.load(handler)

    def _load_feature_extractor(self):
        self.extractor = FeatureExtractor(self.sample_rate_hz)


class SpatialParamsRandomForestDemographics(SpatialParamsRandomForest):
    """Spatial Parameter Estimation Class using a pretrained Random Forest including demographic features."""

    age: int  # years
    gender: str  # in ['m', 'f', 'w']
    height: float  # cm
    weight: int  # in kg

    extractor: FeatureExtractorDemographics

    def __init__(
        self,
        sample_rate_hz,
        age,
        gender,
        height,
        weight,
        model_path=None,
    ):
        self.age = age
        self.height = height
        self.weight = weight
        self.gender = gender
        super().__init__(sample_rate_hz=sample_rate_hz, model_path=model_path)

    def _check_subjects_characteristics(self):
        if self.gender and self.gender not in ["m", "f", "w"]:
            raise ValueError("Gender must be in ['m', 'f', 'w']")
        if self.age and 18 > self.age < 110:
            raise ValueError(f"Age must be in [18, 110]. Is: {self.age}")
        if self.height and 140 > self.height < 215:
            raise ValueError(f"Height must be in [140, 215]. Is: {self.height}")
        if self.weight and 45 > self.weight < 200:
            raise ValueError(f"Height must be in [45, 200]. Is: {self.weight}")

    def _load_model(self):
        if not self.model_path:
            # find model path

            if self.grav_alignment_method == "trim":
                self.model_path = HERE.joinpath(
                    "trained_models",
                    "ml_randomforest",
                    "2023_12_rf_" + str(self.sample_rate_hz) + "hz_regressor_withDemo_trimGravityAlignment.pkl",
                )
            elif self.grav_alignment_method == "static":
                self.model_path = HERE.joinpath(
                    "trained_models",
                    "ml_randomforest",
                    "2023_05_rf_" + str(self.sample_rate_hz) + "hz_regressor_withDemo_staticGravityAlignment.pkl",
                )
            else:
                raise ValueError(
                    f"grav_alignment_method must be in ['trim', 'static']. Is: {self.grav_alignment_method}"
                )

            self.model_path = HERE.joinpath(
                "trained_models",
                "ml_randomforest",
                "2023_05_rf_" + str(self.sample_rate_hz) + "hz_regressor_withDemo.pkl",
            )
        if not self.model_path.is_file():
            potential_models = [
                int(x.stem.split("hz")[0].split("_")[-1]) for x in HERE.iterdir() if (x.is_file() and "hz" in str(x))
            ]
            raise ValueError(
                f"No model available for sample rate {self.sample_rate_hz}Hz. Model are given for: {potential_models}"
            )

        with open(self.model_path, "rb") as handler:
            self.model = pickle.load(handler)

    def _load_feature_extractor(self):
        self.extractor = FeatureExtractorDemographics(
            self.sample_rate_hz, self.height, self.gender, self.age, self.weight
        )
