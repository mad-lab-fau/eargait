"""Base class for spatial parameter methods."""


class SpatialParamsBase:
    """Base Class for Spatial Parameter Estimation."""

    def __init__(self):
        pass

    def estimate(self, data, event_list):
        raise NotImplementedError("No valid spatial parameter method was provided.")
