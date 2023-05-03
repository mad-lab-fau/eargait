r"""
.. _example_gait_analyis_pipeline_signia_data:

Gait Analysis Pipeline for Signia Hearing Aid Data
==================================================

This example illustrates how the gait analysis pipeline by the :class:`~eargait.eargait.EarGait`
can be applied to data recorded with Signia Hearing Aids.

The used gait event detection method is based on the work of Diao et al. [1]_ with a few adaptations as described in :class:`~eargait.event_detection.DiaoAdaptedEventDetection`

.. [1] Diao, Y., Ma, Y., Xu, D., Chen, W., & Wang, Y. (2020). A novel gait parameter estimation method for healthy
    adults and postoperative patients with an ear-worn sensor. Physiological measurement, 41(5), 05NT01
"""

# %%
# Getting some example data
# -------------------------
#
# For this we take some example data that contains regular walking movements.
import pandas as pd

from eargait import EarGait
from eargait.event_detection import DiaoAdaptedEventDetection
from eargait.preprocessing import align_gravity_and_convert_ear_to_ebf, load
from eargait.spatial_params import SpatialParamsExample
from eargait.utils.example_data import get_mat_example_data_path

# path to data file (.txt or .mat) or data directory (only for .mat)
data_path = get_mat_example_data_path()

# %%
# Loading the data
# ----------------
#
# A data session refers to a recording by signia hearing aids.
# A session can consist of a single `*.txt` or `*.mat` file, or two `*.mat` files, for left and right ear, respectively.
# The session is loaded using the local path `data_path` of the directory, in which the matlab/txt file(s) are stored.
# For more options regarding loading the data see :ref:`example_load_data`.
target_sample_rate = 50
session = load(data_path, target_sample_rate_hz=target_sample_rate, skip_calibration=True)
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
# Load csv file containing walking bouts.
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
# `apply filter = True`,
# `sampling_rate_hz` needs to correspond to target_sample_rate_hz,
# * window_length` should be equal to sampling_rate_hz.

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
# `sampling_rate_hz`needs to correspond to `target_sample_rate_hz`.
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
