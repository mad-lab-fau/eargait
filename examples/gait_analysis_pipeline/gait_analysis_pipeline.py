r"""
.. _example_gait_analyis_pipeline:

Gait Analysis Pipeline
======================

This example illustrates how the gait analysis pipeline by the :class:`~eargait.eargait.EarGait`
can be applied to ear-worn accerlation data.

The used gait event detection method is based on the work of Diao et al. [1]_ with a few adaptations as described in :class:`~eargait.event_detection.DiaoAdaptedEventDetection`

.. [1] Diao, Y., Ma, Y., Xu, D., Chen, W., & Wang, Y. (2020). A novel gait parameter estimation method for healthy
    adults and postoperative patients with an ear-worn sensor. Physiological measurement, 41(5), 05NT01
"""

# %%
# Getting some example data
# -------------------------
#
# For this we take some example data that contains regular walking movements.

from eargait import EarGait
from eargait.event_detection import DiaoAdaptedEventDetection
from eargait.spatial_params import SpatialParamsExample

# %%
# Loading the data
# ----------------
#
# Calibrated + alinged to gravity + body frame
from eargait.utils.example_data import get_example_data, get_example_data_path

data, target_sample_rate = get_example_data()
data

# %%
target_sample_rate

# %%
# Initializing event detection algorithm
# --------------------------------------
#
# Recommended parameters:
# apply filter = True <br />
# sampling_rate_hz needs to correspond to target_sample_rate_hz <br />
# window_length should be equal to sampling_rate_hz

event_detection_algorithm = DiaoAdaptedEventDetection(
    sample_rate_hz=target_sample_rate, window_length=target_sample_rate
)

# %%
# Initializing spatial parameter estimation method
# ------------------------------------------------
# Note: SpatialParamsExample is an placeholder class.
# Needs to be implemented by user if spatial parameters want to be estimated.
spatial_method = SpatialParamsExample(target_sample_rate)

# %%
# Initializing Gait Analysis Pipeline
# -----------------------------------
#
# Recommended parameters:
# sampling_rate_hz needs to correspond to target_sample_rate_hz
ear_gait = EarGait(
    sample_rate_hz=target_sample_rate,
    event_detection_method=event_detection_algorithm,
    spatial_params_method=spatial_method,
    bool_use_event_list_consistent=True,
)

# %%
# Detect gait events of gait sequence
# -----------------------------------
#
ear_gait.detect(data)
gait_events = ear_gait.event_list

# %%
# Get all gait parameters
# -----------------------
#
gait_params = ear_gait.get_gait_parameters()
gait_params

# %%
# Get temporal gait parameters of gait sequence
# ---------------------------------------------
#
temporal_params = ear_gait.temporal_params
temporal_params

# %%
# Get average temporal gait parameters
# ------------------------------------
#
average_temporal_params = ear_gait.average_temporal_params
average_temporal_params

# %%
# Get spatial parameter for walking bout
# --------------------------------------
#
spatial_params = ear_gait.spatial_params
spatial_params


# %%
# Get average spatial parameter over walking bout
# -----------------------------------------------
#
spatial_params_average = ear_gait.average_spatial_params
spatial_params_average


# %%
# Get cadence (num steps/duration)
# --------------------------------
#
cadence = ear_gait.cadence
cadence

# %%
# Get cadence based on the dominant frequency
# -------------------------------------------
#
cadence_dominant_freq = ear_gait.cadence_dominant_freq
cadence_dominant_freq

# %%
# Get asymmetry, symetry index or variability
# -------------------------------------------
#
symmetry_index = ear_gait.get_symmetry_index()
symmetry_index
# same for ear_gait.get_variability(), ear_gait.get_asymmetry()


# %%
# Plotting gait events
# --------------------
#
ear_gait.plot()
k = 1
