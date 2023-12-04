"""Constants."""

#: The default names of the Gyroscope columns in the sensor frame
SF_GYR = ["gyr_x", "gyr_y", "gyr_z"]
#: The default names of the Accelerometer columns in the sensor frame
SF_ACC = ["acc_x", "acc_y", "acc_z"]
#: The default names of all columns in the sensor frame
SF_COLS = [*SF_ACC, *SF_GYR]

#: The default names of the Gyroscope columns in the body frame
BF_GYR = ["gyr_pa", "gyr_ml", "gyr_si"]
#: The default names of the Accelerometer columns in the body frame
BF_ACC = ["acc_pa", "acc_ml", "acc_si"]
#: The default names of all columns in the body frame
BF_COLS = [*BF_ACC, *BF_GYR]

#: Minimum values of gravity aligned data in body frame
MIN_VALUES_GRAV_ALIGNED_BF_ACC = {"acc_pa": -3.5, "acc_ml": -1.5, "acc_si": -10.5}
#: Maximum values of gravity aligned data in body frame
MAX_VALUES_GRAV_ALIGNED_BF_ACC = {"acc_pa": 4.0, "acc_ml": 1.5, "acc_si": -7.5}
# Values for gravity aligned singal in BF based on dataset distribution.
# Dataset: Seifer et al., "EarGait: estimation of temporal gait parameters from hearing aid
# integrated inertial sensors." Sensors 23(14), 2023. https://doi.org/10.3390/s23146565."

#: Minimum reasonable values of acceleration data in sensor frame
# MIN_VALUES_SF_ACC = {"acc_x": -3.5, "acc_y": -9.81, "acc_z": 2.0}

#: Maximum reasonable values of acceleration data in sensor frame
# MAX_VALUES_SF_ACC = {"acc_x": -3.5, "acc_y": -1.5, "acc_z": -10.5}


#: Minimum reasonable values of acceleration data in sensor body frame (no gravity alignment).
# MIN_VALUES_BF_ACC = {"acc_pa": -3.5, "acc_ml": -1.5, "acc_si": -10.5}

#: Maximum reasonable values of acceleration data in sensor frame
# MAX_VALUES_BF_ACC = {"acc_pa": -3.5, "acc_ml": -1.5, "acc_si": -10.5}
