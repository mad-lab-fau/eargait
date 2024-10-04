r"""
.. _example_gait_sequence_detector:

Gait Sequence Detection for Session-Compatible Datasets
=======================================================

This example illustrates how the gait sequence detection pipeline can be applied to Session-compatible
datasets recorded by Signia Hearing Aids.

"""
# %%
# Getting example data
# --------------------
#
# Get example data, load it, align with gravity alignment and transformation into body frame.
# For mor information on data loading see :ref:`example_load_data`.

from eargait.preprocessing import align_gravity_and_convert_ear_to_ebf, load
from eargait.utils import TrimMeanGravityAlignment
from eargait.utils.example_data import get_mat_example_data_path, load_groundtruth

data_path = get_mat_example_data_path()
target_sample_rate = 50
session = load(data_path, target_sample_rate_hz=target_sample_rate, skip_calibration=True)
trim_method = TrimMeanGravityAlignment(sampling_rate_hz=target_sample_rate)
ear_data = align_gravity_and_convert_ear_to_ebf(session, trim_method)


# %%
# Initialize Gait Sequence Detection
# ----------------------------------
#
# We instantiate the `GaitSequenceDetection` class with the desired parameters.
# strictness=0 and minimum_sequence_length=1 are seen as standard.
# strictness is defined as >=0 while minimum_seq_length as >=1 definitions of these parameters in the Gsd class.
from eargait.gait_sequence_detection.gait_sequence_detector import GaitSequenceDetection

gsd = GaitSequenceDetection(sample_rate=target_sample_rate, strictness=0, minimum_seq_length=1)

# %%
# Detect Gait Sequences
# ---------------------
#
# Apply the detection algorithm to the processed data.
# The `activity` parameter specifies the type of activity to detect, e.g., "walking".
# This uses a pretrained model for gait detection trained on 50hz data, so make sure to resample your data
# according to this in "align_calibrate_sess"

gsd.detect(ear_data, activity="walking")
sequence_list = gsd.sequence_list_
print("Timeframes of specified activity", sequence_list)


# %%
# Plot Detected Sequences
# -----------------------
#
# Visualize the detected sequences by overlaying them on the accelerometer data.

gsd.plot()


# %%
# Advanced Usage and Customization
# --------------------------------
#
# The pipeline allows customization of parameters like `strictness` and `minimum_seq_length` to fine-tune the detection
# process based on the specific requirements of your dataset.

gsd = GaitSequenceDetection(sample_rate=target_sample_rate, strictness=0, minimum_seq_length=1)

# %%
# Handling Multiple Activities
# ----------------------------
#
# The `GaitSequenceDetection` class supports detecting multiple activities simultaneously
# by passing a list of activities to the `detect` method.

gsd.detect(ear_data, activity=["walking", "jogging", "stairs up", "stairs down"])


# %%
# Further usefully analysis: Overlay Ground truth sequences and detected sequences
# --------------------------------------------------------------------------------
#
# To compare the detected sequences with eventually present Ground truth data we first need to make sure
# the Ground truth is present in the same sampling rate.

from eargait.utils.overlapping_regions import categorize_intervals, plot_categorized_intervals

csv_path_groundtruth = data_path.parent.joinpath("walking_bout_indices.csv")
csv_activity_table = load_groundtruth(csv_path_groundtruth, target_sample_rate=target_sample_rate)
csv_activity_table = csv_activity_table[csv_activity_table["speed"] == data_path.stem]
csv_activity_table = csv_activity_table.rename(columns={"stop": "end"})

# Plotting this overlays Ground truth and detected activity sequences.
gsd.plot(csv_activity_table)
