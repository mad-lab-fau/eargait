r"""
.. _example_spatial_parameter_estimation:

Spatial Parameter Estimation
============================

This example illustrates how Random Forest by the :class:`~eargait.spatial_params.SpatialParamsRandomForest`
can be used to estimate spatial gait parameters, including step_length and stride_length.

The gait event detection algorithm :class:`~eargait.event_detection.DiaoAdaptedEventDetection` is applied to obtain gait events.
"""

# %%
# Getting some example data
# -------------------------
#
# For this we take some example data that contains regular walking movements.
import pandas as pd

from eargait import EarGait
from eargait.event_detection import DiaoAdaptedEventDetection
from eargait.preprocessing import align_gravity_and_convert_ear_to_ebf, convert_ear_to_ebf, load
from eargait.spatial_params import SpatialParamsRandomForest
from eargait.utils.example_data import get_mat_example_data_path

# data directory
data_path = get_mat_example_data_path()

# %%
# Loading the data
# ----------------
#
# A data session refers to a recording by signia hearing aids. A session can consist of a single `*.mat` file or two `*.mat` file, for left and right ear, respectively.
# The session is loaded using the local path data_path` of the directory, in which the matlab file(s) are stored.
# For more options regardind loading the data see :ref:`example_load_data`. :ref:`example_gait_event_detection`
target_sample_rate = 50
session = load(data_path, target_sample_rate_hz=target_sample_rate, skip_calibration=False)
session.info

# %%
# Gravity alignment and data transformation into body frame
# ---------------------------------------------------------
#
# Align session to gravity and transform coordinate system into body frame

ear_data = align_gravity_and_convert_ear_to_ebf(session)

# Alternatively, you can skip the gravity alignment by using the following function:  convert_ear_to_ebf
# ear_data = convert_ear_to_ebf(session)

# %%
# Extract walking interval
# ------------------------
#
# Note: Here prior knowledge about walking sequence within the given data session are applied.
# load csv file containing walking bouts
rescale_factor = 200 / target_sample_rate
walking_bout_list = pd.read_csv(data_path.parent.joinpath("walking_bout_indices.csv"))
interval = (int(walking_bout_list["start"][0] / rescale_factor), int(walking_bout_list["stop"][0] / rescale_factor))

# alternative if walking bout indices are already known, e.g.
# interval = (14317, 17637)    --> indices need to be replaced based on data

ear_data_short = {}
for side in ear_data.keys():
    ear_data_short[side] = ear_data[side][interval[0] : interval[1]]

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
# Two Alternatives
# 1. spatial_method = SpatialParamsRandomForestDemographics(target_sample_rate, age, gender, height, weight)
# 2. spatial_method = SpatialParamsCNN(target_sample_rate)

spatial_method = SpatialParamsRandomForest(target_sample_rate)


# %%
# Initializing Gait Analysis Pipeline
# --------------------------------------
#
# Recommended parameters:
# sampling_rate_hz needs to correspond to target_sample_rate_hz <br />
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
ear_gait.detect(ear_data_short)
gait_events = ear_gait.event_list


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
