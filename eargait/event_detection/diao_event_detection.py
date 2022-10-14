"""The event detection algorithm by Diao et al, 2020 [1]."""
from typing import Dict, Tuple, TypeVar

import numpy as np
import pandas as pd
from pyts.decomposition import SingularSpectrumAnalysis
from scipy.signal import argrelextrema, find_peaks, firwin, kaiserord, lfilter

from eargait.event_detection.base_event_detection import BaseEventDetection
from eargait.utils.consts import BF_ACC
from eargait.utils.helper_gaitmap import SensorData

Self = TypeVar("Self", bound="DiaoEventDetection")


class DiaoEventDetection(BaseEventDetection):
    """Find gait events in the IMU raw signal based on signal characteristics.

    DiaoEventDetection uses a Singuar Spectrum Analysis to decompose acceleration data.
    The peaks in the dominant ozcillation of the SI axis are consindered to be the initial contact (IC).
    Ipsi and contralateral sides are assigned by looking at the adjecent sample in the ML axis.
    Terminate contacts (TC) are determined based the next minima or maxima in the ML axis.
    More information on the algorithm can be found in the paper by Diao et al. [1].

    Parameters
    ----------
    data
        The data passed to the `detect` method.
    sampling_rate_hz
        The sampling rate of the data
    ssa
        Singular Spectrum Analysis (SSA)
    window_length
        The window length for the SSA.


    Attributes
    ----------
    event_list_
        Event list or dictionary with such values.
        The result of the `detect` method holding all temporal gait events. A stride is defined
        such that terminate contact is follwed by an initial contact of the same foot.
        Gait sequences with no steady alternating sequence of ipsi- and contralateral strides are removed.
    nonconsistent_event_list_
        Event list with non consistent gait events. Non-consistent gait event are stride that do not follow the
        stride defintion as described in 'event_list_', e.g. initial contact is before the terminate contact.


    Examples
    --------
    Get gait events from single sensor signal

    >>> event_detection = DiaoEventDetection()
    >>> event_detection.detect(data=data, sample_rate_hz=200.0)
            ic      tc       side
    s_id
    0      651.0    584.0    ipsilateral
    1      839.0    802.0    contralateral
    2      1089.0   1023.0   ipsilateral
    ...

    Notes
    -----
    terminal contact (`tc`), often also referred to as toe off:
        At `tc` the movement of the ankle joint changes from a plantar flexion to a dorsal extension in the sagittal
        plane..

    initial contact (`ic`), often also referred to as heel strike:
        At `ic` the foot decelerates rapidly when the foot hits the ground.

    ipsilateral and contralateral:
        In contrast to foot worn sensors, ear worn sensors contain gait events from left and right foot.
        A stride of the foot situated at same body side as the ear worn sensor is referred to as ipsilateral stride.
        A stride of the foot situated at the oppisite body side if referred to as contralateral stride.
        A gait sequence is a steady alternating sequence of ipsi- and contralateral strides.


    [1] Diao, Y., Ma, Y., Xu, D., Chen, W., & Wang, Y. (2020). A novel gait parameter estimation method for healthy
    adults and postoperative patients with an ear-worn sensor. Physiological measurement, 41(5), 05NT01.

    """

    window_length: int
    ssa: SingularSpectrumAnalysis

    # filter parameter
    filter_order_n: int
    filter_taps: np.ndarray
    filtered_data: SensorData

    def __init__(
        self,
        sample_rate_hz: int,
        window_length: int = None,
    ):
        self.window_length = window_length
        self.ssa = None
        super().__init__(sample_rate_hz=sample_rate_hz)

    def _detect_single_dataset(self, data) -> Dict[str, pd.DataFrame]:
        """Detect gait events for a single sensor data set and put into correct output stride list."""
        self._initialize_filter()

        if self.window_length is None:
            self.window_length = self.sample_rate_hz
        assert self.window_length == self.sample_rate_hz

        if self.ssa is None:
            self._initialize_ssa()

        data = self._filter_data(data)

        acc = data[BF_ACC]
        assert acc.shape[0] > 3 * self.sample_rate_hz, "Walking Bout length must be greater than 3 seconds."

        # find events
        event_detection_func = self._select_all_event_detection_method()
        ic, tc = event_detection_func(acc, self.ssa, self.sample_rate_hz)
        event_list = self._get_event_list(ic, tc)
        event_list = pd.DataFrame(event_list).set_index("s_id")
        return {"event_list_": event_list}

    def _find_all_events(
        self, acc: pd.DataFrame, ssa: SingularSpectrumAnalysis, sample_rate_hz: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Find events in provided data by looping over single strides."""
        acc_si = acc["acc_si"].to_numpy().reshape(1, -1)
        acc_ssa_si = ssa.fit_transform(acc_si)
        acc_ml = acc["acc_ml"].to_numpy().reshape(1, -1) * -1  # changed
        acc_ssa_ml = ssa.fit_transform(acc_ml)

        ic, ic_sides = self._detect_ic(acc_ssa_si[1], acc_ssa_ml[1] + acc_ssa_ml[2], sample_rate_hz)
        tc, tc_sides = self._detect_tc(acc_ssa_ml[1] + acc_ssa_ml[2], ic, ic_sides)

        ic += acc.index[0]
        tc += acc.index[0]

        return (
            pd.DataFrame.from_dict({"ic": ic, "side": ic_sides}),
            pd.DataFrame.from_dict({"tc": tc, "side": tc_sides}),
        )

    @staticmethod
    def _detect_ic(
        acc_si_dominant: np.ndarray, acc_ml_wo_trend: np.ndarray, sample_rate_hz: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        # find minimum on SI axis with removed trend. Peaks corresponds to IC
        peaks_ic, _ = find_peaks(acc_si_dominant * -1, height=0.2, distance=sample_rate_hz * 0.2)
        # determine ipsilateral and contralateral IC
        sides = np.array(["contralateral" for i in range(peaks_ic.shape[0])])

        sides[(acc_ml_wo_trend[peaks_ic + 1] - acc_ml_wo_trend[peaks_ic]) > 0] = "ipsilateral"
        # flipped compared to Diao, because acc_ml was flipped in previous function
        return peaks_ic, sides

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
        delay = int(0.5 * (self.filter_order_n - 1))
        filtered_data = data.apply(self._apply_filter_on_column, axis=0).to_numpy()[delay:]
        index = data.index.to_numpy()[0 : data.shape[0] - delay]
        filtered_data = pd.DataFrame(filtered_data, index=index, columns=data.columns)
        return filtered_data

    def _apply_filter_on_column(self, x):
        return lfilter(self.filter_taps, 1.0, x)

    def _initialize_filter(self):
        # initialize FIR filter
        nyq_rate = self.sample_rate_hz / 2.0
        width = 2.0 / nyq_rate
        ripple_db = 60.0
        self.filter_order_n, beta = kaiserord(ripple_db, width)
        cutoff_hz = 5.0
        self.filter_taps = firwin(self.filter_order_n, cutoff_hz / nyq_rate, window=("kaiser", beta))

    def _initialize_ssa(self) -> Self:
        self.ssa = SingularSpectrumAnalysis(
            window_size=int(self.window_length), groups=[[0], [1], np.arange(2, self.window_length, 1)]
        )
