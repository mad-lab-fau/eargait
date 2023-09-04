"""Class for all event detection algorithms based on ear worn motion sensors."""
from typing import Dict, TypeVar, Union

import pandas as pd

from eargait.base.base_eargait import BaseEarGait
from eargait.utils.gait_parameters import (
    combine_spatial_temporal,
    get_average_spatial_params,
    get_average_spatiotemp_params,
    get_average_temporal_params,
    get_temporal_params,
)
from eargait.utils.helper_gaitmap import Hashable, SensorData, get_multi_sensor_names, is_sensor_data, is_stride_list
from eargait.utils.helpers import butter_lowpass_filter

Self = TypeVar("Self", bound="EarGait")


class EarGait(BaseEarGait):
    """Base class for gait analysis using ear worn sensors.

    Parameters
    ----------
    data
        The IMU data passed to the `detect` method.
    sample_rate_hz
        The sample rate of the data


    Attributes
    ----------
    event_detection_method
        Method used to detect gait events, typically DiaoAdaptedEventDetection
    event_list
        Event list or dictionary with such values.
        The result of the `detect` method holding all temporal gait events. A stride is defined
        such that terminate contact is follwed by an initial contact of the same foot.
    event_list_consistent_
        Event list with consistent gait events, hence, an alternating sequence of ipsi- and contralateral strides.
        Non consistent (non-alternating) step are set to NaN.
    temporal_params
        Temporal parameter of the strides in a gait sequence given the event_list_.
    average_temporal_params
        Average temporal parameters of a gait sequence.
    spatial_params
        Spatial parameter of the strides in a gait sequence.
    average_spatial_params
        Average spatial parameters of a gait sequence.
    spatiotemporal_params
        Spatiotemporal parameter of the strides in a gait sequence.
    average_spatiotemporal_params
        Average spatiotemporal parameters of a gait sequence.
    step_counter
        Number of steps in a gait sequences. Calculated by the sum over the number if initial contacts in the
        event_list_.
    cadence
        Frequency of steps averaged over a gait sequence in steps/minute.


    Examples
    --------
    Gait Analysis for ear-worn sensors
    >>> gait_event_detection_method = DiaoAdaptedEventDetection(200,200)
    >>> eargait = EarGait(200, gait_event_detection_method, True)
    >>> eargait = detect(data)
    >>> average_temporal_params = eargait.average_temporal_params()

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

    """

    sample_rate_hz: float
    event_detection_method = None
    spatial_params_method = None
    bool_use_event_list_consistent: True

    data: SensorData = None
    event_list: Union[pd.DataFrame, Dict[str, pd.DataFrame]] = None

    _temporal_params_memory = None
    _spatial_params_memory = None
    _spatiotemporal_params_memory = None

    def __init__(
        self,
        sample_rate_hz: float,
        event_detection_method=None,
        spatial_params_method=None,
        bool_use_event_list_consistent: bool = False,
    ):
        super().__init__()
        self.sample_rate_hz = sample_rate_hz
        self.event_detection_method = event_detection_method
        if isinstance(spatial_params_method, bool):
            raise ValueError(
                "Input for spatial method can not be of type bool. "
                "Please provide spatial parameter estimation method or None."
            )
        self.spatial_params_method = spatial_params_method
        self.bool_use_event_list_consistent = bool_use_event_list_consistent

    def detect(self: Self, data: SensorData) -> Self:  # noqa
        """Find gait events in data within strides provided by stride_list.

        Parameters
        ----------
        data
            The data set holding the imu raw data

        sample_rate_hz
            The sample rate of the data

        Returns
        -------
        self
            The class instance with all result attributes populated

        """
        self._clear_all_variable()
        self.data = data
        if self.event_detection_method is None:
            raise ValueError(
                "No event detection method was specified. Please define a event detection method, "
                "e.g. DiaoAdaptedEventDetection"
            )
        self.event_detection_method.detect(self.data)
        if not self.bool_use_event_list_consistent:
            self.event_list = self.event_detection_method.event_list_
        else:
            self.event_list = self.event_detection_method.event_list_consistent_
        return self

    @property
    def temporal_params(self) -> pd.DataFrame:
        """Return temporal gait parameters."""
        if self._temporal_params_memory is None:
            assert self._has_event_list()
            temporal_params = get_temporal_params(self.event_list, self.sample_rate_hz)
            self._temporal_params_memory = temporal_params
        return self._temporal_params_memory

    @property
    def spatial_params(self) -> pd.DataFrame:
        """Return spatial gait parameters."""
        if self._spatial_params_memory is None:
            assert self._has_event_list()

            if is_stride_list(self.event_list) == "single":
                spatial_tmp = self.spatial_params_method.estimate(self.data, self.event_list)
                spatial_params = self._estimate_gait_velocity(spatial_tmp, self.temporal_params)
            else:
                spatial_params = {}
                for sensor, events in self.event_list.items():
                    spatial_tmp = self.spatial_params_method.estimate(data=self.data[sensor], event_list=events)
                    spatial_params[sensor] = self._estimate_gait_velocity(spatial_tmp, self.temporal_params[sensor])
            self._spatial_params_memory = spatial_params
        return self._spatial_params_memory

    @property
    def average_temporal_params(self) -> pd.DataFrame:
        """Return average temporal gait parameters."""
        average_temporal_params = get_average_temporal_params(self.temporal_params)
        return average_temporal_params

    @property
    def average_spatial_params(self) -> pd.DataFrame:
        """Return average spatial gait parameters."""
        average_spatial_params = get_average_spatial_params(self.spatial_params)
        return average_spatial_params

    @property
    def spatiotemporal_params(self) -> pd.DataFrame:
        """Return spatiotemporal gait parameters."""
        if self._spatiotemporal_params_memory is None:
            temp = self.temporal_params
            spat = self.spatial_params
            assert len(temp) == len(spat)
            spatiotemporal = combine_spatial_temporal(temp, spat)
            self._spatiotemporal_params_memory = spatiotemporal
        return self._spatiotemporal_params_memory

    @property
    def average_spatiotemporal_params(self) -> pd.DataFrame:
        """Return average spatiotemporal gait parameters."""
        average_spatiotemp = get_average_spatiotemp_params(self.spatiotemporal_params)
        return average_spatiotemp

    @property
    def number_of_steps(self: Self) -> int:
        """Get number of steps of a gait sequence."""
        assert self._has_event_list()
        event_list_type = is_stride_list(self.event_list)
        events_list = self.event_list
        if event_list_type == "single":
            results = self._single_step_counter(events_list)
        else:
            results: Dict[Hashable, Dict[str, pd.DataFrame]] = {}
            for sensor in get_multi_sensor_names(events_list):
                results[sensor] = self._single_step_counter(events_list[sensor])
        return results

    @property
    def cadence(self: Self) -> float:
        """Get the cadence of a gait sequence based on number of step divided by duration."""
        steps = self.number_of_steps
        if is_stride_list(self.event_list) == "single":
            results = self._single_cadence(self.event_list, steps, self.sample_rate_hz)
        else:
            results: Dict[Hashable, Dict[str, pd.DataFrame]] = {}
            for sensor in get_multi_sensor_names(self.event_list):
                results[sensor] = self._single_cadence(self.event_list[sensor], steps[sensor], self.sample_rate_hz)
        return results

    @property
    def cadence_dominant_freq(self: Self) -> float:
        """Get the cadence of a gait sequence based on dominant frequency of a walking bout."""
        if self.data is not None:
            dataset_type = is_sensor_data(self.data, frame="body", check_gyr=False)
            filtered_data = butter_lowpass_filter(self.data, 20, self.sample_rate_hz / 2, 4)
            if dataset_type == "single":
                results = self._single_cadence_dominant_freq(filtered_data, self.sample_rate_hz)
            else:
                results: Dict[Hashable, Dict[str, pd.DataFrame]] = {}
                for sensor in get_multi_sensor_names(self.data):
                    results[sensor] = self._single_cadence_dominant_freq(filtered_data[sensor], self.sample_rate_hz)
            return results
        return None

    def get_gait_parameters(self):
        """Get the average gait parameters of a gait sequence."""
        assert self._has_event_list()
        average_spatiotemp = self.average_spatiotemporal_params
        asy = self.get_asymmetry()
        si = self.get_symmetry_index()
        vari = self.get_variability()
        no_steps = self.number_of_steps
        cad = self.cadence
        cad_harm = self.cadence_dominant_freq

        if is_stride_list(self.event_list) == "single":
            gait_params = self.single_gait_parameters(average_spatiotemp, asy, si, vari, no_steps, cad, cad_harm)
        else:
            gait_params = {}
            for se in self.event_list.keys():
                df = self.single_gait_parameters(
                    average_spatiotemp[se], asy[se], si[se], vari[se], no_steps[se], cad[se], cad_harm[se]
                )
                gait_params[se] = df
        return gait_params

    @staticmethod
    def single_gait_parameters(average_spatiotemp, asy, si, vari, no_steps, cad, cad_harm):
        df = pd.Series(dtype="float64")
        df = pd.concat([df, average_spatiotemp.loc["mean"]])
        df = pd.concat([df, pd.Series(no_steps, index=["number_of_steps"])])
        df = pd.concat([df, pd.Series(cad, index=["cadence"])])
        df = pd.concat([df, pd.Series(cad_harm, index=["cadence_dom_freq"])])
        df = pd.concat([df, asy, si])
        df = pd.concat([df, vari])
        return df

    def plot(self: Self, plot_gait_events: bool = True, plot_ssa: bool = False):
        """Plot gait data and events."""
        if plot_gait_events:
            assert self._has_event_list()
            self.event_detection_method.plot(plot_ssa)
        else:
            self._plot()

    def get_asymmetry(self):
        """Get asymmetry for temporal and spatial parameters.

        Absolute and in percent.
        """
        if is_stride_list(self.event_list) == "single":
            asymmetry = self._single_asymmetry(self.spatiotemporal_params, self.average_spatiotemporal_params)
        else:
            asymmetry = {}
            for sensor, temps in self.spatiotemporal_params.items():
                asymmetry[sensor] = self._single_asymmetry(temps, self.average_spatiotemporal_params[sensor])
        return asymmetry

    def get_symmetry_index(self):
        """Get symmetry index for temporal and spatial parameters.

        Defined by:
        [Zhao et al., IMU-based Gait Analysis for Rehabilitation Assessment of Patients with Gait Disorders, IEEE 2017]

        """
        assert self._has_event_list()
        if is_stride_list(self.event_list) == "single":
            symmetry = self._single_symmetry_index(self.spatiotemporal_params)
        else:
            symmetry = {}
            for sensor, temps in self.spatiotemporal_params.items():
                symmetry[sensor] = self._single_symmetry_index(temps)
        return symmetry

    def get_variability(self):
        """Get variability for spatiotemporal parameters.

        STD: standard deviation
        CV: Coefficient Variation: ration STD/MEAN

        """
        assert self._has_event_list()

        if is_stride_list(self.event_list) == "single":
            variability = self._single_variability(self.average_spatiotemporal_params)
        else:
            variability = {}
            for sensor, temps in self.average_spatiotemporal_params.items():
                variability[sensor] = self._single_variability(temps)
        return variability
