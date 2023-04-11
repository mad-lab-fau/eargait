"""The event detection algorithm by Diao et al, 2020 [1] with adaptations."""
import warnings
from typing import Tuple, TypeVar

import numpy as np
import pandas as pd
from pyts.decomposition import SingularSpectrumAnalysis

from eargait.event_detection.diao_event_detection import DiaoEventDetection

Self = TypeVar("Self", bound="DiaoAdaptedEventDetection")


class DiaoAdaptedEventDetection(DiaoEventDetection):
    """Find gait events in the IMU raw signal based on adapted Diao algorithm.

    DiaoEventDetection uses a Singuar Spectrum Analysis to decompose acceleration data.
    The peaks in the dominant ozcillation of the SI axis are consindered to be the initial contact (IC).
    Ipsi and contralateral sides are assigned by looking at the adjecent sample in the ML axis.
    Terminate contacts (TC) are determined based the next minima or maxima in the ML axis.
    More information on the algorithm can be found in the paper by Diao et al. [1].

    To Do: report adaptations

    Parameters
    ----------
    data
        The IMU data passed to the `detect` method.
    sampling_rate_hz
        The sampling rate of the data
    ssa
        Singular Spectrum Analysis (SSA).
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

    window_length

    Examples
    --------
    Get gait events from single sensor signal

    >>> event_detection = DiaoAdaptedEventDetection()
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
        An event of the foot situated at same body side as the ear worn sensor is referred to as ipsilateral event.
        An event of the foot situated at the oppisite body side if referred to as contralateral event.
        A gait sequence is a steady alternating sequence of ipsi- and contralateral events.


    [1] Diao, Y., Ma, Y., Xu, D., Chen, W., & Wang, Y. (2020). A novel gait parameter estimation method for healthy
    adults and postoperative patients with an ear-worn sensor. Physiological measurement, 41(5), 05NT01.

    """

    def __init__(
        self,
        sample_rate_hz: int,
        window_length: int = None,
    ):
        super().__init__(sample_rate_hz=sample_rate_hz, window_length=window_length)

    def _initialize_ssa(self) -> Self:
        self.ssa = SingularSpectrumAnalysis(
            window_size=self.window_length, groups=[[0], [1], [2], np.arange(3, self.window_length, 1)]
        )

    def _find_all_events(
        self, acc: pd.DataFrame, ssa: SingularSpectrumAnalysis, sample_rate_hz: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Find events in provided data by looping over single strides."""
        acc_si = acc["acc_si"].to_numpy().reshape(1, -1)
        acc_ssa_si = ssa.fit_transform(acc_si)
        acc_ml = acc["acc_ml"].to_numpy().reshape(1, -1) * -1  # changed
        acc_ssa_ml = ssa.fit_transform(acc_ml)

        ic, ic_sides = self._detect_ic(acc_ssa_si[1], acc_ssa_ml[1], sample_rate_hz)
        tc, tc_sides = self._detect_tc(acc_ssa_ml[2] + acc_ssa_ml[1] + acc_ssa_ml[3], ic, ic_sides)

        ic += acc.index[0]
        tc += acc.index[0]

        step_time_ration = acc.shape[0] / sample_rate_hz / ic.shape[0]
        if 0.3 > step_time_ration > 0.8:
            # print(np.abs(acc.shape[0]/(sample_rate_hz*ic.shape[0])-0.5))
            msg = (
                f"Walking bout length and number of ICs are unrealistic: "
                f"{acc.shape[0] / sample_rate_hz:.2f}s and {ic.shape[0]} steps"
            )
            warnings.warn(msg, UserWarning)

        return (
            pd.DataFrame.from_dict({"ic": ic, "side": ic_sides}),
            pd.DataFrame.from_dict({"tc": tc, "side": tc_sides}),
        )
