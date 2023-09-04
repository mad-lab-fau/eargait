"""Eargait: The Gait and Movement Analysis Package unsing Ear-Worn IMU sensors."""
from eargait.spatial_params.spatial_params_cnn import SpatialParamsCNN
from eargait.spatial_params.spatial_params_example_class import SpatialParamsExample
from eargait.spatial_params.spatial_params_inverted_pendulum import SpatialParamsInvertedPendulum
from eargait.spatial_params.spatial_params_rf import SpatialParamsRandomForest, SpatialParamsRandomForestDemographics

__all__ = [
    "SpatialParamsExample",
    "SpatialParamsCNN",
    "SpatialParamsInvertedPendulum",
    "SpatialParamsRandomForest",
    "SpatialParamsRandomForestDemographics",
]
