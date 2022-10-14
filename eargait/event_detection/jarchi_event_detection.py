"""Event detection algorithm by Jarchi. Not recommended to use."""
import warnings
from typing import Dict, Tuple, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyts.decomposition import SingularSpectrumAnalysis
from scipy.signal import argrelextrema, find_peaks, firwin, kaiserord, lfilter  # noqa

from eargait.event_detection.base_event_detection import BaseEventDetection
from eargait.utils.consts import BF_ACC
from eargait.utils.helper_gaitmap import SensorData

Self = TypeVar("Self", bound="JarchiEventDetection")

# pylint: skip-file


class JarchiEventDetection(BaseEventDetection):
    """Find gait events in the IMU raw signal based on signal characteristics.

    RamppEventDetection uses signal processing .... [cite]

    Parameters
    ----------
    window_length
        The window length for the SSA.

    Attributes
    ----------
    event_list_ : ?? A stride list or dictionary with such values
        The result of the `detect` method holding all temporal gait events and start / end of all strides.
        This version of the results has the same stride borders than the input `stride_list` and has additional columns
        for all the detected events.
        Strides for which no valid events could be found are removed.

    walking_speed_

    Other Parameters
    ----------------
    data
        The data passed to the `detect` method.
    sampling_rate_hz
        The sampling rate of the data


    Examples
    --------
    Get gait events from single sensor signal

    >>> event_detection = JarchiEventDetection()
    >>> event_detection.detect(data=data, sampling_rate_hz=200.0)
            ic      tc       side
    s_id
    0      651.0    584.0    ipsilateral
    1      839.0    802.0    contralateral
    2      1089.0   1023.0   ipsilateral
    ...

    Notes
    -----
    terminal contact (`tc`), originally called toe-off (TO):

    initial contact (`ic`), originally called heel strike (HS):

    .. [1] Jarchi et al...

    """

    data: SensorData
    sampling_rate_hz: float

    window_length: int
    ssa: SingularSpectrumAnalysis
    t1: int

    event_list_: Union[pd.DataFrame, Dict[str, pd.DataFrame]]

    def __init__(
        self,
        window_length: int = 200,
        sampling_rate_hz: int = 200,
    ):
        # raise ValueError("Jarchi event detection method is not fully implemented and hence not recommended to use.")
        message = "Jarchi event detection method is not fully implemented and hence not recommended to use."
        warnings.warn(message)

        super(JarchiEventDetection, self).__init__()  # noqa
        self.window_length = window_length
        self.sampling_rate_hz = sampling_rate_hz
        self.ssa = SingularSpectrumAnalysis(
            window_size=window_length, groups=[[0], [1, 2], np.arange(3, window_length, 1)]
        )
        self.t1 = int(0.05 * sampling_rate_hz)
        assert self.t1 != 0  # based on approximation using an image in the original paper,
        # because not explicitly stated in the original paper

    def _detect_single_dataset(self, data) -> Dict[str, pd.DataFrame]:
        """Detect gait events for a single sensor data set and put into correct output stride list."""

        acc = data[BF_ACC]

        # find events
        event_detection_func = self._select_all_event_detection_method()
        ic, tc = event_detection_func(acc)
        event_list_ = self._get_event_list(ic, tc)
        event_list_ = pd.DataFrame(event_list_).set_index("s_id")
        return {"event_list_": event_list_}

    def _find_all_events(self, acc: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Find events in provided data by looping over single strides."""
        # from eargait.utils.helpers import butter_lowpass_filter
        # for col in acc.columns:
        #    acc[col] = butter_lowpass_filter(acc[col], 20, self.sampling_rate_hz/2, 4)
        acc_ssa = self._fit_transform_ssa(acc)
        ic, ic_sides = self._detect_ic(acc, acc_ssa)
        acc_ml_wo_trend = acc_ssa["acc_ml"][1] + acc_ssa["acc_ml"][2] + np.mean(acc["acc_ml"])
        tc, tc_sides = self._detect_tc(acc_ml_wo_trend, ic, ic_sides)

        ic += acc.index[0]
        tc += acc.index[0]

        return (
            pd.DataFrame.from_dict({"ic": ic, "side": ic_sides}),
            pd.DataFrame.from_dict({"tc": tc, "side": tc_sides}),
        )

    def _fit_transform_ssa(self, acc: pd.DataFrame) -> Dict:
        signal_compents = {}
        for axis in acc.columns:
            signal_compents[axis] = self.ssa.fit_transform(acc[axis].to_numpy().reshape(1, -1))
        return signal_compents

    def _detect_ic(self, acc: pd.DataFrame, acc_ssa: dict):
        ap_dominant = acc_ssa["acc_pa"][1]
        mins, _ = find_peaks(ap_dominant * -1, height=0, distance=self.sampling_rate_hz * 0.2)

        acc_wo_trend = {}
        for axes in acc_ssa.keys():
            acc_wo_trend[axes] = acc_ssa[axes][1] + acc_ssa[axes][2] + np.mean(acc[axes])

        # Note: Modification original paper.
        #       acc_si * -1 because defined as pointing upwards in paper, here: pointing downwards
        ic_events = np.array(
            [self._find_min_in_interval(acc_wo_trend["acc_pa"], acc_wo_trend["acc_si"] * -1, x, self.t1) for x in mins]
        )
        ic_side = []
        for i in range(len(ic_events) - 2):
            mean1 = np.mean(acc_wo_trend["acc_ml"][ic_events[i] : ic_events[i + 1]])
            mean2 = np.mean(acc_wo_trend["acc_ml"][ic_events[i + 1] : ic_events[i + 2]])

            # Note: Modification original paper.
            #       Flipped smaller sign because otherwise ipsi and contralateral are swapped
            if mean1 < mean2:
                ic_side.append("ipsilateral")
            else:
                ic_side.append("contralateral")
        ic_side = (
            ic_side + ["ipsilateral", "contralateral"]
            if ic_side[-1] == "contralateral"
            else ic_side + ["contralateral", "ipsilateral"]
        )
        return ic_events, np.array(ic_side)

    @staticmethod
    def _find_min_in_interval(acc_pa, acc_si, peak, t):
        if peak < t:
            t = peak
        elif (peak + t) > len(acc_pa):
            t = len(acc_pa) - peak
        interval = acc_pa[peak - t : peak + t] * (acc_si[peak - t : peak + t])
        return np.argmin(interval) + peak - t

    def _detect_tc(self, acc_ml, ic, ic_side):
        # contra TC: find local max BEFORE contra IC --> flip to local min
        # ipsi TC: find first local minimum AFFTER contra IC --> flip to local max
        # On ml axis without trend corrected with mean
        maximas, _ = find_peaks(acc_ml, width=self.sampling_rate_hz * 0.1)
        minimas, _ = find_peaks(acc_ml * -1, width=self.sampling_rate_hz * 0.1)
        tc_contra = []
        tc_ipsi = []
        for i in ic[ic_side == "contralateral"]:
            potenial_contra = maximas[(i - maximas) > 0]
            if potenial_contra.shape[0] > 0:
                tc_contra.append(potenial_contra[-1])
            else:
                tc_contra.append(np.nan)
            potential_ipsi = minimas[(minimas - i) > 0]
            if potential_ipsi.shape[0] > 0:
                tc_ipsi.append(potential_ipsi[0])
            else:
                tc_ipsi.append(np.nan)
        tc_contra = pd.DataFrame(data=tc_contra, columns=["tc"])
        tc_contra = tc_contra.assign(side="contralateral")
        tc_ipsi = pd.DataFrame(data=tc_ipsi, columns=["tc"])
        tc_ipsi = tc_ipsi.assign(side="ipsilateral")
        tc = pd.concat([tc_contra, tc_ipsi])
        tc = tc.sort_values(by="tc")
        return tc["tc"].to_numpy(), tc["side"].to_numpy()

    def plot(self: Self) -> Self:
        if not self._has_event_list():
            return None
        fig, axes = plt.subplots(2, 1, sharex=True)

        for i, sensor_pos in enumerate(self.event_list_.keys()):
            axes[i].plot(self.data[sensor_pos]["acc_si"], label="acc_si", color="b")
            axes[i].plot(self.data[sensor_pos]["acc_pa"], label="acc_pa", color="b", alpha=0.3, ls="--")
            axes[i].plot(self.data[sensor_pos]["acc_ml"], label="acc_ml", color="b", alpha=0.3, ls="-.")

            axes[i].set_title(sensor_pos)
            events = self.event_list_[sensor_pos]
            data = self.data[sensor_pos]

            for side, color in zip(["ipsilateral", "contralateral"], ["orange", "green"]):
                for event, marker in zip(["ic", "tc"], ["o", "v"]):
                    if event == "tc":
                        continue
                    tmp = events.loc[events.side == side, event].dropna()
                    axes[i].plot(tmp, data.loc[tmp]["acc_si"], marker, c=color, label=side + " " + event, ms=6, lw=1)
                    if event == "ic":
                        for ev in tmp:
                            axes[i].axvline(ev, c=color, lw=1, alpha=0.5)
        axes[0].legend(fontsize=12, bbox_to_anchor=(1.0, 0.0), loc="upper left")
        # plt.tight_layout()
        plt.show()
        return self
