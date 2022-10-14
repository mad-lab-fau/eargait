"""Set of helper functions to to deal with different data types."""
from typing import Dict, Hashable, Union

import pandas as pd
from typing_extensions import Literal

from eargait.utils.helper_gaitmap import ValidationError, is_stride_list

SingleSensorEventList = pd.DataFrame
MultiSensorEventList = Union[pd.DataFrame, Dict[Union[Hashable, str], SingleSensorEventList]]
EventList = Union[SingleSensorEventList, MultiSensorEventList]


def is_event_list(event_list: EventList) -> Literal["single", "multi"]:
    """Check if an object is a valid multi-sensor or single-sensor event list.

    This function utilizes the is_stride_list function by gaitmap.
    For more information, please refer to: eargait.utils.helper_gaitmap.is_stride_list

    Parameters
    ----------
    event_list
        The object that should be tested

    Returns
    -------
    dataset_type
        "single" in case of a single-sensor event list, "multi" in case of a multi-sensor event list.
        In case it is neither, a ValidationError is raised.

    """
    return is_stride_list(event_list)


def get_event_list_type(
    event_list: EventList,
) -> Literal["lower_body_sensor", "upper_body_sensor_bilateral", "upper_body_sensor_unilateral"]:
    """Check if an object is a valid multi-sensor or single-sensor event list.

    In case one of the two checks is successful, a string is returned, which type of dataset the input is.
    Otherwise a descriptive error is raised

    Parameters
    ----------
    event_list
        The object that should be tested

    Returns
    -------
    dataset_type
        "lower_body_sensor" in case of a event list by e.g. foot-mounted sensors,
        "upper_body_sensor_bilateral" in case of a event list by a sensor mounted to the upper body
        (e.g. ear, head, lower back) including events of both legs,
        "upper_body_sensor_unilateral" in case of a event list by a sensor mounted to the upper body
        (e.g. ear, head, lower back) including events of single leg.
        In case it is neither, a ValidationError is raised.

    """
    kind = is_event_list(event_list)
    check_bilateral(event_list)

    if kind == "single":
        event_list_type = _get_event_list_type_single_sensor_list(event_list)
    else:
        event_list_type_list = []
        for sensor_pos in event_list.keys():
            event_list_type_list.append(_get_event_list_type_single_sensor_list(event_list[sensor_pos]))
        if len(set(event_list_type_list)) > 1:
            raise ValidationError(
                "The passed object appears to be neither a multi-sensor event list. "
                "However, the lists appear to be of two different kinds:\n\n"
                f"{str(event_list_type_list)}\n\n"
            )
        event_list_type = event_list_type_list[0]
    return event_list_type


def _get_event_list_type_single_sensor_list(event_list: SingleSensorEventList):
    if "side" in event_list.columns:
        if event_list.side.nunique() == 1:
            return "upper_body_sensor_unilateral"
        if event_list.side.nunique() == 2:
            return "upper_body_sensor_bilateral"
        return None
    return "lower_body_sensor"


def check_bilateral(event_list: EventList):
    kind = is_event_list(event_list)

    if kind == "single":
        _check_bilateral_single(event_list)
    else:
        for events in event_list.values():
            _check_bilateral_single(events)


def _check_bilateral_single(event_list: EventList):
    if "side" in event_list.columns:
        side_set = set(event_list.side.unique())
        if not side_set.issubset(set(["ipsilateral", "contralateral"])):
            raise ValueError(
                "Side of bilateral sensor are not in [ipsilateral, contralaeral]", event_list.side.unique()
            )
