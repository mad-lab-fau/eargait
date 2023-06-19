"""Base class for all event detection algorithms based on ear worn motion sensors."""
import warnings
from typing import Callable, Dict, Optional, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eargait.utils.helper_gaitmap import BaseEventDetection as BaseEventDetectionGaitmap
from eargait.utils.helper_gaitmap import (
    Hashable,
    SensorData,
    get_multi_sensor_names,
    invert_result_dictionary,
    is_sensor_data,
    set_params_from_dict,
)

Self = TypeVar("Self", bound="BaseEventDetection")


class BaseEventDetection(BaseEventDetectionGaitmap):
    """Base class for gait event detection methods.

    Parameters
    ----------
    data
        The IMU data passed to the `detect` method.
    sample_rate_hz
        The sample rate of the data


    Attributes
    ----------
    event_list_
        Event list or dictionary with such values.
        The result of the `detect` method holding all temporal gait events. A stride is defined
        such that terminate contact is follwed by an initial contact of the same foot.
    event_list_consistent_
        Event list with consistent gait events, hence, an alternating sequence of ipsi- and contralateral events.
        Non consistent (non-alternating) step are set to NaN.

    Examples
    --------
    Get gait events from single sensor signal

    >>> event_detection = BaseEventDetection()
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

    """

    data: SensorData = None
    sample_rate_hz: float

    event_list_: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    event_list_consistent_: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]

    bool_use_event_list_consistent: bool

    def __init__(
        self,
        sample_rate_hz: float,
        bool_use_event_list_consistent: bool = False,
    ):
        self.sample_rate_hz = sample_rate_hz
        self.bool_use_event_list_consistent = bool_use_event_list_consistent

    def detect(self: Self, data: SensorData) -> Self:  # noqa
        """Find gait events in data.

        Parameters
        ----------
        data
            The data set holding the imu raw data

        Returns
        -------
        self
            The class instance with all result attributes populated

        """
        dataset_type = is_sensor_data(data, frame="body", check_gyr=False)

        self.data = data

        if dataset_type == "single":
            results = self._detect_single_dataset(self.data)  # noqa
        else:
            results_dict: Dict[Hashable, Dict[str, pd.DataFrame]] = {}
            for sensor in get_multi_sensor_names(data):
                results_dict[sensor] = self._detect_single_dataset(self.data[sensor])  # noqa
            results = invert_result_dictionary(results_dict)

        set_params_from_dict(self, results, result_formatting=True)
        return self

    @property
    def event_list_consistent_(self: Self):
        """Consistent based on alternating ipsi- and contralateral steps, non-consistent steps are set to NaN."""
        assert self._has_event_list()
        dataset_type = is_sensor_data(self.data, frame="body", check_gyr=False)
        if dataset_type == "single":
            event_list_consistent_ = self._single_consistent_event_list(self.event_list_)
        else:
            event_list_consistent_: Dict[Hashable, Dict[str, pd.DataFrame]] = {}
            for sensor in get_multi_sensor_names(self.event_list_):
                event_list_consistent_[sensor] = self._single_consistent_event_list(self.event_list_[sensor])
        return event_list_consistent_

    def plot(self: Self, plot_ssa: bool = False):
        """Plot gait data and events."""
        assert self._has_event_list()
        if isinstance(self.event_list_, dict) == 1:
            for side in self.event_list_.keys():
                self._plot(sensor_pos=side, plot_ssa=plot_ssa)
        else:
            self._plot(sensor_pos=None, plot_ssa=plot_ssa)

    def _select_all_event_detection_method(self: Self) -> Callable:  # noqa: no-self-use
        """Select the function to calculate the all events.

        This is separate method to make it easy to overwrite by a subclass.
        """
        return self._find_all_events

    def _has_event_list(self: Self):
        if not hasattr(self, "event_list_"):
            warnings.warn("No event_list exists. Please run self.detect().")
            return False
        return True

    @staticmethod
    def _get_event_list(ic: pd.DataFrame, tc: pd.DataFrame) -> pd.DataFrame:
        """Create an event list based on a list of ic's and tc's.

        To each IC a suitable TC is assigned taking the sides into consideration.
        A stride is defined such that the TC is followed by an IC.

        Parameters
        ----------
        ic
            Dataframe containing all IC events and the corresponding side

        tc
            Dataframe containing all TC events and the corresponding side

        Returns
        -------
        event_list
            Dataframe containing gait events of IMU signal

        """
        event_list = ic.copy()
        event_list.ic = event_list.ic.astype(float)
        event_list.insert(1, "tc", np.nan)
        for side in event_list.side.unique():
            ic_side = ic.loc[ic.side == side].ic.to_numpy()
            tc_side = tc.loc[tc.side == side].tc.to_numpy()
            for i in range(ic_side.shape[0] - 1):
                tc_idx = tc_side[(ic_side[i] < tc_side) * (ic_side[i + 1] > tc_side)]
                if tc_idx.shape[0] == 1:
                    event_list.loc[event_list.ic == ic_side[i + 1], "tc"] = tc_idx[-1]

            # handle first ic
            first_tc = tc_side[tc_side < ic_side[0]]
            if len(first_tc) > 0:
                event_list.loc[event_list.ic == ic_side[0], "tc"] = first_tc[-1]
        event_list.insert(0, "s_id", np.arange(0, event_list.shape[0], 1))
        return event_list

    def _detect_single_dataset(self, data) -> Dict[str, pd.DataFrame]:  # noqa
        """Gait event detection on single data set.

        Parameters
        ----------
        data
            Data of single dataset

        Returns
        -------
        event_list_ as dict in the form of: {"event_list_": pd.DataFrame}
        """
        raise NotImplementedError("Needs to be implemented by child class.")

    def _plot(self: Self, sensor_pos: str = None, plot_ssa: bool = False) -> Self:  # noqa
        fau_blaues = ["#003865", "#00b1eb", "#98a4ae", "#c99313", "#8d1429", "#009b77"]
        # dunkelblau, hellblau, grau, gelb,  rot, grün

        _, axes = plt.subplots(1, 1, sharex=True, figsize=(8, 4))

        if sensor_pos is None:
            data = self.data
        else:
            data = self.data[sensor_pos]
        filtered_data = self._filter_data(data)

        for axis, style, color in zip(["acc_si", "acc_ml", "acc_pa"], ["-", "--", "-."], fau_blaues):
            axes.plot(data[axis], label=axis, color=color, ls=style)
            # axes.plot(filtered_data[axis], color=color, ls=style, alpha=0.2)

        if plot_ssa:
            t = filtered_data["acc_si"].index.to_numpy()
            ssa_si = self.ssa.fit_transform(filtered_data["acc_si"].to_numpy().reshape(1, -1))
            ssa_ml = self.ssa.fit_transform(filtered_data["acc_ml"].to_numpy().reshape(1, -1))

            axes.plot(
                t,
                ssa_si[1] + filtered_data["acc_si"].mean(),
                label="acc_si_dominant_with_mean",
                color="red",
                alpha=0.7,
                ls="--",
            )
            axes.plot(
                t,
                ssa_ml[1] + filtered_data["acc_ml"].mean(),
                label="acc_ml_dominant_with_mean",
                color="red",
                alpha=0.7,
                ls="-.",
            )
            axes.plot(
                t,
                ssa_ml[1] + ssa_ml[2] + ssa_ml[3] + data["acc_ml"].mean(),
                label="acc_ml_rest",
                color="red",
                alpha=0.7,
                ls="--",
            )

        if sensor_pos is not None:
            axes.set_title(sensor_pos)
            events = self.event_list_[sensor_pos]
        else:
            events = self.event_list_

        for side, color in zip(["ipsilateral", "contralateral"], ["orange", "green"]):
            for event, marker in zip(["ic", "tc"], ["o", "v"]):
                tmp = events.loc[events.side == side, event].dropna()
                axes.plot(tmp, data.loc[tmp]["acc_si"], marker, c=color, label=side + " " + event, ms=6, lw=1)
        axes.legend(fontsize=12, loc="upper right")
        plt.show()
        return self

    @staticmethod
    def _single_consistent_event_list(event_list_):
        tmp = event_list_.copy()
        tmp["diff"] = 1
        tmp.loc[tmp.side == "contralateral", "diff"] = -1
        tmp["diff"] = tmp["diff"].diff()
        r = np.abs(np.diff(np.sign(tmp["diff"]))) == 2
        r[0] = True
        # counter = sum(~(r))
        r = np.append(r, True)
        event_list_.loc[~r, ["ic", "tc"]] = np.nan
        return event_list_
