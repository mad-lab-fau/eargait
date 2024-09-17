r"""
.. _example_gait_sequence_detector:

Gait Sequence Detection for Session-Compatible Datasets
=======================================================

This example illustrates how the gait sequence detection pipeline can be applied to Session-compatible
datasets recorded by Signia Hearing Aids.

"""
# %%
# Getting example data
# -------------------------
#
# First, we import the necessary modules and load example data.
import pandas as pd
from pathlib import Path
from eargait.gait_sequence_detection.gait_sequence_detector import GaitSequenceDetection
from eargait.utils.example_data import get_mat_example_data_path

# Path to data file (.mat) or .mat data directory
# Session should also work with txt file? so also gait detection?
repo_root = Path(__file__).resolve().parent.parent.parent
print("Repo Root",repo_root)

data_path = get_mat_example_data_path()
csv_path = repo_root / "example_data/mat_files/walking_bout_indices.csv"

# %%
# Loading data
# ----------------
#
# A data session refers to a recording (by Signia Hearing Aids).
# A session can consist of a single `*.txt` or `*.mat` file, or two `*.mat` files, for left and right ear, respectively.
# The session is loaded using the local path `data_path` of the directory, in which the Matlab/txt file(s) are stored.
from signialib import Session

session = Session.from_folder_path(data_path)
# Its recommended to NOT use skip_calibration = True, but use an up-to-date calibration file for the used Sensor.
align_calibrate_sess = session.align_calib_resample(resample_rate_hz=50, skip_calibration=True)
session.info

# %%
# Gravity alignment and transformation into body frame
# ---------------------------------------------------------
#
# Align session to gravity and transform the coordinate system into a body frame.
# Two methods are provided: `StaticWindowGravityAlignment` and `TrimMeanGravityAlignment`.
# Skipping this leads to unusable data on which the gait detection is applied.

from eargait.utils import StaticWindowGravityAlignment, TrimMeanGravityAlignment
from eargait.preprocessing import align_gravity_and_convert_ear_to_ebf

gravity_method = "static"
static_method = StaticWindowGravityAlignment(sampling_rate_hz=50)
trim_method = TrimMeanGravityAlignment(sampling_rate_hz=50)

if gravity_method == "static":
    ear_data = align_gravity_and_convert_ear_to_ebf(align_calibrate_sess, static_method)
else:
    ear_data = align_gravity_and_convert_ear_to_ebf(align_calibrate_sess, trim_method)

# %%
# Initialize Gait Sequence Detection
# ----------------------------------
#
# We instantiate the `GaitSequenceDetection` class with the desired parameters.
# strictness=0 and minimum_sequence_length=1 are seen as standard.
# strictness is defined as >=0 while minimum_seq_length as >=1 definitions of these parameters in the Gsd class.

gsd = GaitSequenceDetection(sample_rate=50, strictness=0, minimum_seq_length=1)

# %%
# Detect Gait Sequences
# ---------------------
#
# Apply the detection algorithm to the processed data.
# The `activity` parameter specifies the type of activity to detect, e.g., "walking".
# This uses a pretrained model for gait detection trained on 50hz data, so make sure to resample your data
# according to this in "align_calibrate_sess"

gsd.detect(ear_data, activity="walking")

# %%
# Display Detected Sequences
# --------------------------
#
# Print the timeframes where the specified activity occurs in the data.

print("Timeframes of specified activity", gsd.sequence_list_)

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

gsd = GaitSequenceDetection(sample_rate=50, strictness=0, minimum_seq_length=1)

# Handling Multiple Activities
# ----------------------------
#
# The `GaitSequenceDetection` class supports detecting multiple activities simultaneously
# by passing a list of activities to the `detect` method.

gsd.detect(ear_data, activity=["walking", "jogging", "stairs up", "stairs down"])

# this is also plottable_
gsd.plot()



########################################################################################################################
# AB HIER ALLES WEG FÜR DAS MINIMAL BSP
# %%
# Further usefully analysis: Overlay Ground truth sequences and detected sequences
# --------------------------------------------------------------------------------
#
# To compare the detected sequences with eventually present Ground truth data we first need to make sure
# the Ground truth is present in the same sampling rate.


def downsample_ground_truth(csv_path, target_sample_rate):
    csv_table = pd.read_csv(csv_path)
    downsampling_factor = 200 / target_sample_rate
    csv_table["start"] = (csv_table["start"] / downsampling_factor).astype(int)
    csv_table["stop"] = (csv_table["stop"] / downsampling_factor).astype(int)
    return csv_table


tempo = get_mat_example_data_path().stem
csv_activity_table = downsample_ground_truth(csv_path, target_sample_rate=50)
csv_activity_table = csv_activity_table[csv_activity_table["speed"] == tempo]
csv_activity_table = csv_activity_table.rename(columns={"stop": "end"})

# Plotting this overlays Ground truth and detected activity sequences.
gsd.plot(csv_activity_table)


# %%
# Percentual representation of overlap:
# -------------------------------------
#
# To have a one number expression of how good the detection worked we can display the percentual overlap of predicted
# and Ground truth sequences as the expresssion of true positive percentage of correctly identified walking sequences.

from eargait.utils.overlapping_regions import categorize_intervals


def calculate_tp_percentage(detected_sequences, ground_truth_sequences):
    categorized_intervals = categorize_intervals(detected_sequences, ground_truth_sequences)

    ground_truth_duration = sum(ground_truth_sequences["end"] - ground_truth_sequences["start"])
    true_positive_duration = sum(
        categorized_intervals.tp_intervals["end"] - categorized_intervals.tp_intervals["start"]
    )
    tp_percentage_gt = (true_positive_duration / ground_truth_duration) * 100 if ground_truth_duration > 0 else 0
    return tp_percentage_gt


detected_sequences = gsd.sequence_list_["left_sensor"][["start", "end"]]
ground_truth_sequences = csv_activity_table[["start", "end"]]
tp_percentage = calculate_tp_percentage(detected_sequences, ground_truth_sequences)

print(f"True Positive Percentage: {tp_percentage:.2f}%")
########################################################################################################################
# Strictness and min_length Parameter
# ----------------------------
#
# strictness
# Determines the size of the gap (in number of windows) at which two consecutive sequences are linked
# together to a single sequence.
# minimum_seq_length
# Determines the minimum length of a sequence (in windows). Needs to be >= 1.
sequence = pd.DataFrame({
        'start': [2100, 2550, 3750, 4750, 6000, 7300, 8250, 9300, 10200],
        'end': [2550, 3600, 4450, 5850, 6900, 7950, 9000, 9450, 10350]
    })
print("Original Seqeuence:", sequence)
sample_rate = 50
strictness = 2
minimum_seq_length = 2
gsd = GaitSequenceDetection(sample_rate=sample_rate, strictness=strictness, minimum_seq_length=minimum_seq_length)
sequence = gsd._ensure_strictness(sequence)
print("Seqeunce after Strictness criterion:", sequence)
sequence = gsd._ensure_minimum_length(sequence)
print("Sequence after min_length criterion:", sequence)
