"""Helper functions for loading data from Signia Hearing Aids."""

import warnings
from pathlib import Path

import numpy as np
from signialib import SIGNIA_CAL_PATH, Session


def load(path: str, target_sample_rate_hz: int = 0, skip_calibration: bool = False, calib_path: str = None):
    """Load motion data recorded using signia hearing aids.

    Parameters
    ----------
    path:
        Local path of folder containing the recording.

    target_sample_rate_hz:
        The target sampling rate of the data. The target sampling rate only has to be provided, if data
        should be resampled to a lower sampling rate.
        Default 0 means, no resampling.

    skip_calibration:
        If True, (Ferraris) calibration is skipped. It's strongly recommended to use calibration.
        Calibration file for each sensor is stored in signia lib.

    calib_path:
        Folder containing the calibration files for given IMU sensors.
        If None, the default path by signialib SIGNIA_CAL_PATH is used. Can only be set if ´skip_calibration´ is False.

    Returns
    -------
    session
        Signia Session

    Notes
    -----
    This function can only be applied to dataset recorded using the Signia Hearing Aids.

    """
    path = Path(path)
    if path.is_file():
        session = Session.from_file_path(path)
    else:
        session = Session.from_folder_path(path)
    if target_sample_rate_hz != 0:
        _check_valid_resample_rate(session.info.sampling_rate_hz[0], target_sample_rate_hz)
    else:
        target_sample_rate_hz = session.info.sampling_rate_hz[0]

    if skip_calibration:
        if calib_path is not None:
            raise ValueError("Variable `calib_path` can only be set if `skip_calibration=False`.")
        warnings.warn("Calibration was skipped. Calibration is strongly recommended.")
        if len(session.datasets) == 2:
            session = session.align_to_syncregion()
        session = session.resample(target_sample_rate_hz)
    else:
        if calib_path is None:
            calib_path = SIGNIA_CAL_PATH
        session = session.align_calib_resample(resample_rate_hz=target_sample_rate_hz, calib_path=calib_path)
    return session


def _check_valid_resample_rate(orig_sample_rate_hz, target_rate_hz):
    if np.mod(orig_sample_rate_hz, target_rate_hz) != 0:
        raise ValueError(
            "Please choose valid target sample rate. Has to be divivend of original sample rate {}Hz".format(  # noqa
                int(orig_sample_rate_hz)
            )
        )
    return np.mod(orig_sample_rate_hz, target_rate_hz)
