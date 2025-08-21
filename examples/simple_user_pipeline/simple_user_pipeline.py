r"""
.. _example_simple_user_pipeline:

User Pipeline for easy usage of GSD and Eargait combination.
=======================================================

This example illustrates a simple approach on the usage of User friendly Interface which combines GSD and Eargait in
the background.

"""
# %%
# Load and preprocess example data
# --------------------------------
#
# Loading example data, align sensor data using gravity alignment and convert to EBF for the sake of this example.
# Make sure you preprocess your data to only having one-sided Sensor data.

from eargait.preprocessing import align_gravity_and_convert_ear_to_ebf, load
from eargait.utils import TrimMeanGravityAlignment
from eargait.utils.example_data import get_mat_example_data_path
# Load data
sample_rate = 50
data_path = get_mat_example_data_path()
session = load(data_path, target_sample_rate_hz=sample_rate, skip_calibration=True)

# Preprocess data
trim_method = TrimMeanGravityAlignment(sampling_rate_hz=sample_rate)
ear_data = align_gravity_and_convert_ear_to_ebf(session, trim_method)
ear_data = ear_data["left_sensor"]

# %%
# Initialize Gait Sequence Analyzer Pipeline
# ------------------------------------------
#
# Instantiate the pipeline with adjustable parameters for sequence detection and analysis.

from eargait.gait_sequence_analyzer_pipeline import GaitSequenceAnalyzerPipeline

pipeline = GaitSequenceAnalyzerPipeline(sample_rate=sample_rate, strictness=0, min_seq_length=1)

# %%
# Automatic Gait Sequence Detection and Analysis
# ----------------------------------------------
#
# Automatically detect gait sequences and compute gait parameters.

pipeline.compute(ear_data, activity="walking")
# Display automatic detection results
print("Automatically detected sequences and parameters:")
print(pipeline.auto_sequence_list_)
print(pipeline.auto_average_params_)

# %%
# Manual Gait Sequence Analysis
# -----------------------------
#
# We also provide the option to have prelabeled sequences instead of relying on the GSD.
# Provide manually labeled sequences as pd.DataFrame as following::

import pandas as pd

manual_sequences = pd.DataFrame(
    {
        "start": [1050, 1800, 2700, 3750, 4650, 5550, 6450, 7350],
        "end": [1230, 2250, 3300, 4350, 5250, 6150, 7050, 7800],
    }
)

pipeline.compute_predef_seq(ear_data, manual_sequences)

# Display manual analysis results
print("Manually provided sequences and parameters:")
print(pipeline.manual_sequence_list_)
print(pipeline.manual_average_params_)