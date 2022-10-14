"""Mixed Diao and Jarchi (IC detection diao, side assignment jarchi). Not recommendet to use."""
from typing import Dict, Optional, Tuple, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyts.decomposition import SingularSpectrumAnalysis
from scipy.signal import argrelextrema, find_peaks, firwin, kaiserord, lfilter

from eargait.event_detection.base_event_detection import BaseEventDetection
from eargait.utils.consts import BF_ACC
from eargait.utils.helper_gaitmap import SensorData

Self = TypeVar("Self", bound="MixedEventDetection")

# pylint: skip-file


class MixedEventDetection(BaseEventDetection):
    """Find gait events in the IMU raw signal based on signal characteristics.

    Parameters
    ----------
    window_length


    Attributes
    ----------
    To Do

    Other Parameters
    ----------------
    To Do

    Examples
    --------
    Get gait events from single sensor signal

    >>> event_detection = MixedEventDetection()
    >>> event_detection.detect(data=data, sampling_rate_hz=200.0)
            ic      tc       side
    s_id
    0      651.0    584.0    ipsilateral
    1      839.0    802.0    contralateral
    2      1089.0   1023.0   ipsilateral
    ...

    """

    window_length: int
    segmented_event_list: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]

    data: SensorData
    sampling_rate_hz: float
    ssa: SingularSpectrumAnalysis
    # stride_list: pd.DataFrame

    # filter parameter
    filter: bool
    filter_order_N: int
    filter_taps: np.ndarray
    filtered_data: SensorData

    def __init__(
        self,
        window_length: int = 200,
        sampling_rate_hz: Optional[float] = 200,
        filter: bool = False,
    ):
        super(MixedEventDetection, self).__init__(sampling_rate_hz)
        raise Warning("Mixed event detection method is not fully implemented and hence not recommended to use.")
        self.window_length = window_length
        self.filter = filter
        self.ssa = SingularSpectrumAnalysis(
            window_size=window_length, groups=[[0], [1], np.arange(2, window_length, 1)]
        )

        if filter:
            # initialize FIR filter
            nyq_rate = self.sampling_rate_hz / 2.0
            width = 2.0 / nyq_rate
            ripple_db = 60.0
            self.filter_order_N, beta = kaiserord(ripple_db, width)
            cutoff_hz = 5.0
            self.filter_taps = firwin(self.filter_order_N, cutoff_hz / nyq_rate, window=("kaiser", beta))

    def _detect_single_dataset(self, data) -> Dict[str, pd.DataFrame]:
        """Detect gait events for a single sensor data set and put into correct output stride list."""

        if self.filter:
            data = self._filter_data(data)

        acc = data[BF_ACC]

        # find events
        event_detection_func = self._select_all_event_detection_method()
        ic, tc = event_detection_func(acc, self.ssa, self.sampling_rate_hz)
        segmented_event_list = self._get_segmented_event_list(ic, tc)
        segmented_event_list = pd.DataFrame(segmented_event_list).set_index("s_id")
        return {"segmented_event_list": segmented_event_list}

    def _find_all_events(
        self, acc: pd.DataFrame, ssa: SingularSpectrumAnalysis, sampling_rate_hz: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Find events in provided data by looping over single strides."""

        acc_si = acc["acc_si"].to_numpy().reshape(1, -1)
        acc_ssa_si = ssa.fit_transform(acc_si)
        acc_ml = acc["acc_ml"].to_numpy().reshape(1, -1) * -1  # changed
        acc_ssa_ml = ssa.fit_transform(acc_ml)

        ic, ic_sides = self._detect_ic(acc_ssa_si[1], acc_ssa_ml[2] + acc_ssa_ml[1], sampling_rate_hz)
        tc, tc_sides = self._detect_tc(acc_ssa_ml[2] + acc_ssa_ml[1], ic, ic_sides)

        ic += acc.index[0]
        tc += acc.index[0]

        return (
            pd.DataFrame.from_dict({"ic": ic, "side": ic_sides}),
            pd.DataFrame.from_dict({"tc": tc, "side": tc_sides}),
        )

    @staticmethod
    def _detect_ic(
        acc_si_dominant: np.ndarray, acc_ml_wo_trend: np.ndarray, sampling_rate_hz: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        # find minimum on SI axis with removed trend. Peaks corresponds to IC
        peaks_ic, prop = find_peaks(acc_si_dominant * -1, height=0, distance=sampling_rate_hz * 0.2)
        # determine ipsilateral and contralateral IC
        ic_side = []
        for i in range(len(peaks_ic) - 2):
            mean1 = np.mean(acc_ml_wo_trend[peaks_ic[i] : peaks_ic[i + 1]])
            mean2 = np.mean(acc_ml_wo_trend[peaks_ic[i + 1] : peaks_ic[i + 2]])
            if mean1 > mean2:
                ic_side.append("ipsilateral")
            else:
                ic_side.append("contralateral")
        ic_side = (
            ic_side + ["ipsilateral", "contralateral"]
            if ic_side[-1] == "contralateral"
            else ic_side + ["contralateral", "ipsilateral"]
        )
        return peaks_ic, np.array(ic_side)

    @staticmethod
    def _detect_tc(acc_ml_wo_trend: np.ndarray, ic: np.ndarray, sides: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        mins = argrelextrema(acc_ml_wo_trend, np.less, order=2)[0]  # changes np.greater used to be np.less
        contra = ic[sides == "contralateral"]

        contra = contra[contra < mins[-1]]
        tc_ipsi = [int(mins[mins > x][0]) for x in contra]

        maxs = argrelextrema(acc_ml_wo_trend, np.greater, order=2)[0]  # changes  np.less used to be np.greater

        ipsi = ic[sides == "ipsilateral"]
        ipsi = ipsi[ipsi < maxs[-1]]
        tc_contra = [int(maxs[maxs > x][0]) for x in ipsi]

        tc_ipsi = list(set(tc_ipsi))
        tc_contra = list(set(tc_contra))

        # combine
        toe_off = np.array(tc_ipsi + tc_contra).astype(int)
        toe_off_side = np.array(["ipsilateral" for x in tc_ipsi] + ["contralateral" for x in tc_contra])

        # sort
        toe_off_side = toe_off_side[toe_off.argsort()]
        toe_off = toe_off[toe_off.argsort()]
        return toe_off, toe_off_side

    def _filter_data(self, data):
        delay = int(0.5 * (self.filter_order_N - 1))
        filtered_data = data.apply(self._apply_filter_on_column, axis=0).to_numpy()[delay:]
        index = data.index.to_numpy()[0 : data.shape[0] - delay]
        filtered_data = pd.DataFrame(filtered_data, index=index, columns=data.columns)
        return filtered_data

    def _apply_filter_on_column(self, x):
        return lfilter(self.filter_taps, 1.0, x)

    @staticmethod
    def plot_tmp(acc, acc_si_wo, acc_ml_wo):
        fig, axes = plt.subplots(1, 1)
        axes.plot(acc["acc_si"].to_numpy(), label="acc_si", c="red")
        axes.plot(acc["acc_ml"].to_numpy(), label="acc_ml", c="b")
        axes.plot(acc_si_wo[1], label="acc_si_wo_trend", c="red", ls="--")
        axes.plot(acc_ml_wo[1], label="acc_ml_wo_trend", c="b", ls="--")
        plt.legend()
        plt.show()

    def plot(self: Self) -> Self:
        fau_colors = {
            "fau": "#003865",  # dunkelblau
            "phil": "#c99313",  # gelb
            "wiso": "#8d1429",  # rot
            "med": "#00b1eb",  # hellblau
            "nat": "#009b77",  # grün
            "tech": "#98a4ae",  # grau
        }
        fau_blaues = [fau_colors["fau"], fau_colors["med"], fau_colors["tech"]]
        if not self._has_event_list():
            return None
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 6))

        for i, sensor_pos in enumerate(self.segmented_event_list_.keys()):
            for axis, style, color in zip(["acc_si", "acc_ml", "acc_pa"], ["-", "--", "-."], fau_blaues):
                axes[i].plot(self.data[sensor_pos][axis], label=axis, color=color, ls=style)

            if self.filter:
                data = self._filter_data(self.data[sensor_pos])
                t = data["acc_si"].index.to_numpy()
                for axis, ls in zip(data.columns, ["--", "-", "-."]):
                    if axis == "acc_pa":
                        continue
                    ssa = self.ssa.fit_transform(data[axis].to_numpy().reshape(1, -1))
                    t = t[0 : len(ssa[0])]
                    if axis == "acc_si":
                        ssa = ssa[1]
                        label = axis + "_dom_os"
                    else:
                        ssa = ssa[1] + ssa[2]
                        label = axis + "_wo_trend"

                    axes[i].plot(t, ssa, label=label, color="red", alpha=1.0, ls=ls)

            axes[i].set_title(sensor_pos)
            strides = self.segmented_event_list_[sensor_pos]
            data = self.data[sensor_pos]

            for side, color in zip(["ipsilateral", "contralateral"], ["orange", "green"]):
                for event, marker in zip(["ic", "tc"], ["o", "v"]):
                    tmp = strides.loc[strides.side == side, event].dropna()
                    axes[i].plot(tmp, data.loc[tmp]["acc_si"], marker, c=color, label=side + " " + event, ms=6, lw=1)
                    if event == "ic":
                        for ev in tmp:
                            axes[i].axvline(ev, c=color, lw=1, alpha=0.5)

        axes[0].legend(fontsize=12, bbox_to_anchor=(1.0, 1.0), loc="upper left")
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.138)
        plt.show()
        return self
