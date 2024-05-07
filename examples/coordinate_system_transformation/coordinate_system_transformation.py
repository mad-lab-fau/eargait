r"""
.. _example_coordinate_system_transformation:

To Do To Do
===========

This example illustrates how ...
"""

# %%
# Getting some example data
# -------------------------
#
# For this we take some example data that contains regular walking movements.

from eargait import EarGait
from eargait.event_detection import DiaoAdaptedEventDetection
from eargait.spatial_params import SpatialParamsRandomForest

# %%
# Loading the data
# ----------------
#
# Calibrated + aligned to gravity + body frame
from eargait.utils.example_data import get_example_data

# Todo
data, target_sample_rate = get_example_data()
data
