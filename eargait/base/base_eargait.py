"""Base class for all event detection algorithms based on ear worn motion sensors."""
import warnings
from typing import TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import norm

from eargait.utils.helper_gaitmap import BF_ACC, BF_GYR, is_sensor_data

Self = TypeVar("Self", bound="BaseEarGait")


class BaseEarGait:
    """Base Class for EarGait."""

    def __init__(self):
        self._temporal_params_memory = None
        self._spatial_params_memory = None
        self._spatiotemporal_params_memory = None

    def _plot(self):
        assert self.data is not None
        dataset_type = is_sensor_data(self.data, frame="body")
        if dataset_type == "single":
            self._plot_single(self.data)
        else:
            for sensor in self.data.keys():
                self._plot_single(self.data[sensor])

    def _has_event_list(self: Self):
        if not hasattr(self, "event_list"):
            warnings.warn("No event_list exists. Please run self.detect().")
            return False
        return True

    @staticmethod
    def _plot_single(data):
        fau_blaues = ["#003865", "#00b1eb", "#98a4ae"]
        _, axes = plt.subplots(2, 1)
        for col, c in zip(BF_ACC, fau_blaues):
            axes[0].plot(data[col], label=col, c=c)
        axes[0].legend()
        for col, c in zip(BF_GYR, fau_blaues):
            axes[1].plot(data[col], label=col, c=c)
        axes[1].legend()

    @staticmethod
    def _single_step_counter(event_list):
        return int(event_list.ic.shape[0])

    @staticmethod
    def _single_cadence(event_list, no_steps, sample_rate) -> int:
        if no_steps == 0:
            return 0
        ics = event_list.ic.to_numpy()
        duration = (ics[-1] - ics[0]) / sample_rate
        return 60 / duration * (no_steps - 1)  # steps/min

    @staticmethod
    def _single_cadence_dominant_freq(filtered_data, sample_rate_hz):
        signal_1d = norm(filtered_data[BF_ACC], axis=1)
        signal_1d = np.reshape(signal_1d, (1, signal_1d.shape[0]))
        locomotion_band = (0.3, 4)
        lower_bound = int(np.floor(sample_rate_hz / locomotion_band[1]))
        # print("lower bound:", lower_bound)
        # (sampling_rate / locomotion_band_lower = upper bound of autocorrelation)
        upper_bound = int(np.ceil(sample_rate_hz / locomotion_band[0]))
        # print("upper bound:", upper_bound)
        # autocorr from 0-upper motion band
        auto_corr = np.empty((signal_1d.shape[0], upper_bound + 1))
        for tau in range(upper_bound + 1):
            tmax = signal_1d.shape[1] - tau
            umax = signal_1d.shape[1] + tau
            auto_corr[:, tau] = (signal_1d[:, :tmax] * signal_1d[:, tau:umax]).sum(axis=1)
        # calculate dominant frequency in Hz
        dominant_frequency = (
            1 / (np.argmax(auto_corr[:, lower_bound:], axis=-1) + lower_bound).astype(float) * sample_rate_hz
        )
        return dominant_frequency[0] * 60  # steps/min

    @staticmethod
    def _single_consistent_event_list(event_list):
        tmp = event_list.copy()
        tmp["diff"] = 1
        tmp.loc[tmp.side == "contralateral", "diff"] = -1
        tmp["diff"] = tmp["diff"].diff()
        r = np.abs(np.diff(np.sign(tmp["diff"]))) == 2
        r[0] = True
        # counter = sum(~(r))
        r = np.append(r, True)
        event_list.loc[~r, ["ic", "tc"]] = np.nan
        return event_list

    @staticmethod
    def _single_symmetry_index(spatiotemp_params):
        ipsi = spatiotemp_params.loc[spatiotemp_params.side == "ipsilateral"].mean(numeric_only=True)
        contra = spatiotemp_params.loc[spatiotemp_params.side == "contralateral"].mean(numeric_only=True)
        si = 2 * np.abs((ipsi - contra) / (ipsi + contra)) * 100
        si = si.add_suffix("_si")
        return si

    @staticmethod
    def _single_variability(spatiotemp_params):
        varia = spatiotemp_params.loc["std"].add_suffix("_std")
        varia = pd.concat([varia, (spatiotemp_params.loc["std"] / spatiotemp_params.loc["mean"]).add_suffix("_cv")])
        return varia

    @staticmethod
    def _single_asymmetry(spatiotemp_params, average_spatiotemp_params):
        ipsi = spatiotemp_params.loc[spatiotemp_params.side == "ipsilateral"].mean(numeric_only=True)
        contra = spatiotemp_params.loc[spatiotemp_params.side == "contralateral"].mean(numeric_only=True)
        diff = np.abs(ipsi - contra)
        diff_percent = diff / average_spatiotemp_params.loc["mean"]
        diff = diff.add_suffix("_asymmetry")
        diff_percent = diff_percent.add_suffix("_asymmetry_percent")
        diff = pd.concat([diff, diff_percent])
        return diff

    def _has_subject_data(self):
        if any([self.age, self.height, self.gender]):
            raise ValueError(
                "Subjects characteristic are not given: age, gender height. Please set by using self.set_subject_data"
            )

    @staticmethod
    def _estimate_gait_velocity(spatial: pd.DataFrame, temporal: pd.DataFrame) -> pd.DataFrame:
        spatial.insert(2, "gait_velocity", spatial.step_length / temporal.step_time)
        return spatial

    def _clear_all_variable(self):
        self._temporal_params_memory = None
        self._spatial_params_memory = None
        self._spatiotemporal_params_memory = None
