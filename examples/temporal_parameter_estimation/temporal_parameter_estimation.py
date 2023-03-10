r"""
.. _example_temporal_parameter_estimation:

Temporal Parameter Estimation
=============================

This example illustrates how temporal gait parameters can be calculated using the  :class:`~eargait.eargait.EarGait`.
The gait event detection method :class:`~eargait.event_detection.DiaoAdaptedEventDetection` is used for estimating gait events.

"""

# %%
# Getting some example data
# -------------------------
#
# For this we take some example data that contains regular walking movements.
import pandas as pd

from eargait import EarGait
from eargait.event_detection import DiaoAdaptedEventDetection
from eargait.preprocessing import aling_gravity_and_convert_ear_to_ebf, convert_ear_to_ebf, load
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
session = load(data_path, target_sample_rate_hz=target_sample_rate, skip_calibration=True)
session.info

# %%
# Gravity alignment and data transformation into body frame
# ---------------------------------------------------------
#
# Align session to gravity and transform coordinate system into body frame

ear_data = aling_gravity_and_convert_ear_to_ebf(session)

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
# Initializing Gait Analysis Pipeline
# --------------------------------------
#
# Recommended parameters:
# sampling_rate_hz needs to correspond to target_sample_rate_hz <br />
ear_gait = EarGait(target_sample_rate, event_detection_algorithm, None, True)

# %%
# Detect gait events of gait sequence
# -----------------------------------
#
ear_gait.detect(ear_data_short)
gait_events = ear_gait.event_list

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
