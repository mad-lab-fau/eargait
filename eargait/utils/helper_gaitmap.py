"""This is a collection of helpers that were copied from MaDLab internal python package gaitmap.
The original copyright belongs to MaD-DiGait group.
All original authors agreed to have the respective code snippets published in this way.
In case any problems are detected, these changes should be upstreamed to the respective internal libraries.
"""

import warnings
from typing import Any, Dict, Hashable, Iterable, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import pandas as pd
import tpcp
from scipy.spatial.transform import Rotation
from typing_extensions import Literal

from eargait.utils.consts import *

GRAV_VEC = np.array([0.0, 0.0, 9.81])
METRIC_FUNCTION_NAMES = Literal["maximum", "variance", "mean", "median"]

BaseType = TypeVar("BaseType", bound="_BaseSerializable")

SingleSensorData = pd.DataFrame
SingleSensorStrideList = pd.DataFrame

MultiSensorData = Union[pd.DataFrame, Dict[Union[Hashable, str], SingleSensorData]]
SensorData = Union[SingleSensorData, MultiSensorData]

MultiSensorStrideList = Dict[Union[Hashable, str], pd.DataFrame]
StrideList = Union[SingleSensorStrideList, MultiSensorStrideList]

_ALLOWED_FRAMES = ["any", "body", "sensor"]
_ALLOWED_FRAMES_TYPE = Literal["any", "body", "sensor"]

#: Sensor to body frame conversion for the left foot
FSF_FBF_CONVERSION_LEFT = {
    "acc_x": (1, "acc_pa"),
    "acc_y": (1, "acc_ml"),
    "acc_z": (-1, "acc_si"),
    "gyr_x": (-1, "gyr_pa"),
    "gyr_y": (-1, "gyr_ml"),
    "gyr_z": (-1, "gyr_si"),
}

#: Sensor to body frame conversion for the right foot
FSF_FBF_CONVERSION_RIGHT = {
    "acc_x": (1, "acc_pa"),
    "acc_y": (-1, "acc_ml"),
    "acc_z": (-1, "acc_si"),
    "gyr_x": (1, "gyr_pa"),
    "gyr_y": (-1, "gyr_ml"),
    "gyr_z": (1, "gyr_si"),
}


class ValidationError(Exception):
    """An error indicating that data-object does not comply with the guidelines."""


class _BaseSerializable(tpcp.BaseTpcpObject):
    @classmethod
    def _get_subclasses(cls: Type[BaseType]):
        for subclass in cls.__subclasses__():
            yield from subclass._get_subclasses()
            yield subclass

    @classmethod
    def _find_subclass(cls: Type[BaseType], name: str) -> Type[BaseType]:
        for subclass in _BaseSerializable._get_subclasses():
            if subclass.__name__ == name:
                return subclass
        raise ValueError("No algorithm class with name {} exists".format(name))


class BaseAlgorithm(tpcp.Algorithm, _BaseSerializable):
    """Base class for all algorithms.

    All type-specific algorithm classes should inherit from this class and need to

    1. overwrite `_action_method` with the name of the actual action method of this class type
    2. implement a stub for the action method

    Attributes
    ----------
    _action_method
        The name of the action method used by the Childclass

    """


class BaseEventDetection(BaseAlgorithm):
    """Base class for all event detection algorithms."""

    _action_methods = ("detect",)

    def detect(self: BaseType, data: SensorData, stride_list: StrideList, sampling_rate_hz: float) -> BaseType:
        """Find gait events in data within strides provided by roi_list."""
        raise NotImplementedError("Needs to be implemented by child class.")


def invert_result_dictionary(
    nested_dict: Dict[Hashable, Dict[TypeVar("_HashableVar", Hashable, str), Any]]
) -> Dict[TypeVar("_HashableVar", Hashable, str), Dict[Hashable, Any]]:
    """Invert result dictionaries that are obtained from multi sensor results.

    This method expects a two level dictionary and flips the levels.
    This means that if a value can be accessed as `nested_dict[k1][k2] = v` in the input, it can be accessed as
    `output_dict[k2][k1] = v`.

    Examples
    --------
    >>> in_dict = {"level_1_1": {"level_2_1": "val_2_1", "level_2_2": "val_2_2"}, "level_1_2": {"level_2_3": "val_2_3"}}
    >>> from pprint import pprint
    >>> pprint(invert_result_dictionary(in_dict))
    {'level_2_1': {'level_1_1': 'val_2_1'},
     'level_2_2': {'level_1_1': 'val_2_2'},
     'level_2_3': {'level_1_2': 'val_2_3'}}

    """
    out: Dict[TypeVar("_HashableVar", Hashable, str), Dict[Hashable, Any]] = dict()
    for ok, ov in nested_dict.items():
        for k, v in ov.items():
            nested = out.setdefault(k, dict())
            nested[ok] = v
    return out


def set_params_from_dict(obj: Any, param_dict: Dict[str, Any], result_formatting: bool = False):
    """Update object attributes from dictionary.

    The object will be updated inplace.

    Parameters
    ----------
    obj
        The gaitmap obj to update
    param_dict
        The dictionary of new values to set/update
    result_formatting
        If True all keys will get a trailing "_", if they don't have one already.
        This marks them as "results" based on the gaitmap guidelines.

    """
    for k, v in param_dict.items():
        if result_formatting is True:
            if not k.endswith("_"):
                k += "_"
        setattr(obj, k, v)


def get_multi_sensor_names(dataset: MultiSensorData):
    """Get the list of sensor names from a multi-sensor dataset.

    .. warning:
        This will not check, if the input is actually a multi-sensor dataset

    Notes
    -----
    The keys are not guaranteed to be ordered.

    """
    return _get_multi_sensor_data_names(dataset=dataset)


def _get_multi_sensor_data_names(dataset: Union[dict, pd.DataFrame]):
    if isinstance(dataset, pd.DataFrame):
        keys = dataset.columns.unique(level=0)
    else:
        # In case it is a dict
        keys = dataset.keys()

    return keys


def _assert_is_dtype(obj, dtype: Union[type, Tuple[type, ...]]):
    """Check if an object has a specific dtype."""
    if not isinstance(obj, dtype):
        raise ValidationError("The dataobject is expected to be one of ({},). But it is a {}".format(dtype, type(obj)))


def _assert_has_multindex_cols(df: pd.DataFrame, nlevels: int = 2, expected: bool = True):
    """Check if a pd.DataFrame has a multiindex as columns.

    Parameters
    ----------
    df
        The dataframe to check
    nlevels
        If MultiIndex is expected, how many level should the MultiIndex have
    expected
        If the df is expected to have a MultiIndex or not

    """
    has_multiindex = isinstance(df.columns, pd.MultiIndex)
    if has_multiindex is not expected:
        if expected is False:
            raise ValidationError(
                "The dataframe is expected to have a single level of columns. "
                "But it has a MultiIndex with {} levels.".format(df.columns.nlevels)
            )
        raise ValidationError(
            "The dataframe is expected to have a MultiIndex with {} levels as columns. "
            "It has just a single normal column level.".format(nlevels)
        )
    if has_multiindex is True:
        if not df.columns.nlevels == nlevels:
            raise ValidationError(
                "The dataframe is expected to have a MultiIndex with {} levels as columns. "
                "It has a MultiIndex with {} levels.".format(nlevels, df.columns.nlevels)
            )


def _assert_has_columns(df: pd.DataFrame, columns_sets):
    """Check if the dataframe has at least all columns sets.

    Examples
    --------
    >>> df = pd.DataFrame()
    >>> df.columns = ["col1", "col2"]
    >>> _assert_has_columns(df, [["other_col1", "other_col2"], ["col1", "col2"]])
    >>> # This raises no error, as df contains all columns of the second set

    """
    columns = df.columns
    result = False
    for col_set in columns_sets:
        result = result or all(v in columns for v in col_set)

    if result is False:
        if len(columns_sets) == 1:
            helper_str = "columns: {}".format(columns_sets[0])
        else:
            helper_str = "one of the following sets of columns: {}".format(columns_sets)
        raise ValidationError(
            "The dataframe is expected to have {}. Instead it has the following columns: {}".format(
                helper_str, list(df.columns)
            )
        )


def _get_expected_dataset_cols(
    frame: Literal["sensor", "body"], check_acc: bool = True, check_gyr: bool = True
) -> List:
    expected_cols = []
    if frame == "sensor":
        acc = SF_ACC
        gyr = SF_GYR
    elif frame == "body":
        acc = BF_ACC
        gyr = BF_GYR
    else:
        raise ValueError('`frame must be one of ["sensor", "body"]')
    if check_acc is True:
        expected_cols.extend(acc)
    if check_gyr is True:
        expected_cols.extend(gyr)
    return expected_cols


def is_single_sensor_data(
    data: SingleSensorData,
    check_acc: bool = True,
    check_gyr: bool = True,
    frame: _ALLOWED_FRAMES_TYPE = "any",
    raise_exception: bool = False,
) -> Optional[bool]:
    """Check if an object is a valid dataset following all conventions.

    A valid single sensor dataset is:

    - a :class:`pandas.DataFrame`
    - has only a single level of column indices that correspond to the sensor (or feature) axis that are available.

    A valid single sensor dataset in the body frame additionally:

    - contains all columns listed in :obj:`SF_COLS <gaitmap.utils.consts.SF_COLS>`

    A valid single sensor dataset in the sensor frame additionally:

    - contains all columns listed in :obj:`BF_COLS <gaitmap.utils.consts.BF_COLS>`

    Parameters
    ----------
    data
        Object that should be checked
    check_acc
        If the existence of the correct acc columns should be checked
    check_gyr
        If the existence of the correct gyr columns should be checked
    frame
        The frame the dataset is expected to be in.
        This changes which columns are checked for.
        In case of "any" a dataset is considered valid if it contains the correct columns for one of the two frames.
        If you just want to check the datatype and shape, but not for specific column values, set both `check_acc` and
        `check_gyro` to `False`.
    raise_exception
        If True an exception is raised if the object does not pass the validation.
        If False, the function will return simply True or False.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_multi_sensor_data: Explanation and checks for multi sensor datasets

    """
    if frame not in _ALLOWED_FRAMES:
        raise ValueError("The argument `frame` must be one of {}".format(_ALLOWED_FRAMES))
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_multindex_cols(data, expected=False)

        if frame == "any":
            _assert_has_columns(
                data,
                [
                    _get_expected_dataset_cols("sensor", check_acc=check_acc, check_gyr=check_gyr),
                    _get_expected_dataset_cols("body", check_acc=check_acc, check_gyr=check_gyr),
                ],
            )
        else:
            _assert_has_columns(data, [_get_expected_dataset_cols(frame, check_acc=check_acc, check_gyr=check_gyr)])

    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be SingleSensorData. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False
    return True


def _assert_multisensor_is_not_empty(obj: Union[pd.DataFrame, Dict]):
    sensors = _get_multi_sensor_data_names(obj)
    if len(sensors) == 0:
        raise ValidationError("The provided multi-sensor object does not contain any data/contains no sensors.")


def is_multi_sensor_data(
    data: MultiSensorData,
    check_acc: bool = True,
    check_gyr: bool = True,
    frame: _ALLOWED_FRAMES_TYPE = "any",
    raise_exception: bool = False,
) -> bool:
    """Check if an object is a valid multi-sensor data object.

    Valid multi sensor data is:

    - is either a :class:`pandas.DataFrame` with 2 level multi-index as columns or a dictionary of single sensor
      datasets (see :func:`~gaitmap.utils.dataset_helper.is_single_sensor_data`)

    In case the data is a :class:`pandas.DataFrame` with two levels, the first level is expected to be the names
    of the used sensors.
    In both cases (dataframe or dict), `dataset[<sensor_name>]` is expected to return a valid single sensor
    dataset.
    On each of the these single-sensor datasets, :func:`~gaitmap.utils.dataset_helper.is_single_sensor_data` is used
    with the same parameters that are used to call this function.

    Parameters
    ----------
    data
        Object that should be checked
    check_acc
        If the existence of the correct acc columns should be checked
    check_gyr
        If the existence of the correct gyr columns should be checked
    frame
        The frame the dataset is expected to be in.
        This changes which columns are checked for.
        In case of "any" a dataset is considered valid if it contains the correct columns for one of the two frames.
        If you just want to check the datatype and shape, but not for specific column values, set both `check_acc` and
        `check_gyro` to `False`.
    raise_exception
        If True an exception is raised if the object does not pass the validation.
        If False, the function will return simply True or False.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_single_sensor_data: Explanation and checks for single sensor data

    """
    try:
        _assert_is_dtype(data, (pd.DataFrame, dict))
        if isinstance(data, pd.DataFrame):
            _assert_has_multindex_cols(data, expected=True, nlevels=2)
        _assert_multisensor_is_not_empty(data)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be MultiSensorData. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False

    try:
        for k in get_multi_sensor_names(data):
            is_single_sensor_data(data[k], check_acc=check_acc, check_gyr=check_gyr, frame=frame, raise_exception=True)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object appears to be MultiSensorData, "
                'but for the sensor with the name "{}", the following validation error was raised:\n\n{}'.format(
                    k, str(e)
                )
            ) from e
        return False
    return True


def is_sensor_data(
    data: SensorData, check_acc: bool = True, check_gyr: bool = True, frame: _ALLOWED_FRAMES_TYPE = "any"
) -> Literal["single", "multi"]:
    """Check if an object is valid multi-sensor or single-sensor data.

    This function will try to check the input using :func:`~gaitmap.utils.dataset_helper.is_single_sensor_data` and
    :func:`~gaitmap.utils.dataset_helper.is_multi_sensor_data`.
    In case one of the two checks is successful, a string is returned, which type of dataset the input is.
    Otherwise a descriptive error is raised

    Parameters
    ----------
    data
        Object that should be checked
    check_acc
        If the existence of the correct acc columns should be checked
    check_gyr
        If the existence of the correct gyr columns should be checked
    frame
        The frame the dataset is expected to be in.
        This changes which columns are checked for.
        In case of "any" a dataset is considered valid if it contains the correct columns for one of the two frames.
        If you just want to check the datatype and shape, but not for specific column values, set both `check_acc` and
        `check_gyro` to `False`.

    Returns
    -------
    data_type
        "single" in case of single-sensor data, "multi" in case of multi-sensor data.
        In case it is neither, a ValidationError is raised.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_single_sensor_data: Explanation and checks for single sensor data
    gaitmap.utils.dataset_helper.is_multi_sensor_data: Explanation and checks for multi sensor data

    """
    try:
        is_single_sensor_data(data, check_acc=check_acc, check_gyr=check_gyr, frame=frame, raise_exception=True)
    except ValidationError as e:
        single_error = e
    else:
        return "single"

    try:
        is_multi_sensor_data(data, check_acc=check_acc, check_gyr=check_gyr, frame=frame, raise_exception=True)
    except ValidationError as e:
        multi_error = e
    else:
        return "multi"

    raise ValidationError(
        "The passed object appears to be neither single- or multi-sensor data. "
        "Below you can find the errors raised for both checks:\n\n"
        "Single-Sensor\n"
        "=============\n"
        f"{str(single_error)}\n\n"
        "Multi-Sensor\n"
        "=============\n"
        f"{str(multi_error)}"
    )


def is_multi_sensor_stride_list(
    stride_list: MultiSensorStrideList, stride_type: str, raise_exception: bool = False
) -> bool:
    """Check if an input is a multi-sensor stride list.

    A valid multi-sensor stride list is dictionary of single-sensor stride lists.

    This function :func:`~gaitmap.utils.dataset_helper.is_single_sensor_stride_list` for each of the contained stride
    lists.

    Parameters
    ----------
    stride_list
        The object that should be tested
    stride_type
        The expected stride type of this object.
        If this is "any" only the generally required columns are checked.
    raise_exception
        If True an exception is raised if the object does not pass the validation.
        If False, the function will return simply True or False.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_single_sensor_stride_list: Check for multi-sensor stride lists
    gaitmap.utils.stride_list_conversion.enforce_stride_list_consistency: Remove strides that do not have the correct
        event order

    """
    try:
        _assert_is_dtype(stride_list, dict)
        _assert_multisensor_is_not_empty(stride_list)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a MultiSensorStrideList. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False

    try:
        for k in stride_list.keys():
            is_single_sensor_stride_list(stride_list[k], stride_type=stride_type, raise_exception=True)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object appears to be a MultiSensorStrideList, "
                'but for the sensor with the name "{}", the following validation error was raised:\n\n{}'.format(
                    k, str(e)
                )
            ) from e
        return False
    return True


def _assert_has_index_columns(df: pd.DataFrame, index_cols: Iterable[Hashable]):
    ex_index_cols = list(index_cols)
    ac_index_cols = list(df.index.names)
    if ex_index_cols != ac_index_cols:
        raise ValidationError(
            "The dataframe is expected to have exactly the following index columns ({}), "
            "but it has {}".format(index_cols, df.index.name)
        )


def set_correct_index(
    df: pd.DataFrame, index_cols: Iterable[Hashable], drop_false_index_cols: bool = True
) -> pd.DataFrame:
    """Set the correct columns as index, or leave them if they are already in the index.

    Parameters
    ----------
    df
        The dataframe
    index_cols
        A list of names that correspond to the names of the multiindex level names (in order)
    drop_false_index_cols
        If True columns that are set as index in df, but shouldn't will be deleted.
        If False these columns will just be removed from the index and become regular df columns.

    Returns
    -------
    df
        A dataframe with the correct columns set as index

    """
    index_cols = list(index_cols)
    try:
        _assert_has_index_columns(df, index_cols)
        return df
    except ValidationError:
        pass

    # In case not all columns are in the the index, reset_the index and check the column names
    wrong_index = [i for i, n in enumerate(df.index.names) if n not in index_cols]
    all_wrong = len(wrong_index) == len(df.index.names)
    df_just_right_index = df.reset_index(level=wrong_index, drop=drop_false_index_cols)
    if not all_wrong:
        # In case correct indix cols are remaining make them to regular columns
        df_just_right_index = df_just_right_index.reset_index()

    try:
        _assert_has_columns(df_just_right_index, [index_cols])
    except ValidationError as e:
        raise ValidationError(
            "The dataframe is expected to have the following columns either in the index or as columns ({}), "
            "but it has {}".format(index_cols, df.columns)
        ) from e

    return df_just_right_index.set_index(index_cols)


def is_stride_list(stride_list: StrideList) -> Literal["single", "multi"]:
    """Check if an object is a valid multi-sensor or single-sensor stride list.

    This function will try to check the input using
    :func:`~gaitmap.utils.dataset_helper.is_single_sensor_stride_list` and
    :func:`~gaitmap.utils.dataset_helper.is_multi_sensor_stride_list`.
    In case one of the two checks is successful, a string is returned, which type of dataset the input is.
    Otherwise a descriptive error is raised

    Parameters
    ----------
    stride_list
        The object that should be tested
    stride_type
        The expected stride type of this object.
        If this is "any" only the generally required columns are checked.

    Returns
    -------
    dataset_type
        "single" in case of a single-sensor stride list, "multi" in case of a multi-sensor stride list.
        In case it is neither, a ValidationError is raised.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_single_sensor_stride_list: Explanation and checks for single sensor stride lists
    gaitmap.utils.dataset_helper.is_multi_sensor_stride_list: Explanation and checks for multi sensor stride lists

    """
    try:
        is_single_sensor_stride_list(stride_list, stride_type="event", raise_exception=True)
    except ValidationError as e:
        single_error = e
    else:
        return "single"

    try:
        is_multi_sensor_stride_list(stride_list, stride_type="event", raise_exception=True)
    except ValidationError as e:
        multi_error = e
    else:
        return "multi"

    raise ValidationError(
        "The passed object appears to be neither a single- or a multi-sensor stride list. "
        "Below you can find the errors raised for both checks:\n\n"
        "Single-Sensor\n"
        "=============\n"
        f"{str(single_error)}\n\n"
        "Multi-Sensor\n"
        "=============\n"
        f"{str(multi_error)}"
    )


def is_single_sensor_stride_list(
    stride_list: SingleSensorStrideList, stride_type: str, raise_exception: bool = False
) -> bool:
    """Check if an input is a single-sensor stride list.

    A valid stride list:

    - is a pandas Dataframe with at least the following columns: `["s_id", "start", "end"]`.
      The `s_id` column can also be part of the index.
    - has only a single level column index
    - the value of `s_id` is unique

    Note that this function does only check the structure and not the plausibility of the contained values.
    For this `~gaitmap.utils.stride_list_conversions.enforce_stride_list_consistency` can be used.

    However, depending on the type of stride list, further requirements need to be fulfilled:

    min_vel
        A min-vel stride list describes a stride list that defines a stride from one midstance (`min_vel`) to the next.
        This type of stride list can be performed for ZUPT based trajectory estimation.
        It is expected to additionally have the following columns describing relevant stride events: `["pre_ic", "ic",
        "min_vel", "tc"]`.
        See :mod:`~gaitmap.event_detection` for details.
        For this type of stride list it is further tested, that the "start" column is actual identical to the "min_vel"
        column.
    segmented
        A segmented stride list is a stride list in which every stride starts and ends between min_vel and tc.
        For this stride list, we expect that all relevant events within each stride are already detected.
        Hence, it is checked if columns with the name `["ic", "tc", "min_vel"]` exist.
        If you want to check the structure of a stride list right after the segmentation, where no events are detected
        yet use `"any"` as `stride_type`.
    ic
        A IC stride list is a stride list in which every stride starts and ends with a IC.
        Regarding columns, it has the same requirements as the "segmented" stride list.
        Additionally it is checked, if the "start" columns is actually identical to the "ic" column.

    Parameters
    ----------
    stride_list
        The object that should be tested
    stride_type
        The expected stride type of this object.
        If this is "any" only the generally required columns are checked.
    raise_exception
        If True an exception is raised if the object does not pass the validation.
        If False, the function will return simply True or False.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_multi_sensor_stride_list: Check for multi-sensor stride lists
    gaitmap.utils.stride_list_conversion.enforce_stride_list_consistency: Remove strides that do not have the correct
        event order

    """
    if stride_type == "event" or stride_type == "any":
        SL_COLS = ["ic", "tc"]
    elif stride_type == "stride":
        SL_COLS = ["start", "end", "ic", "tc"]
    else:
        raise ValueError("Stride type not of type 'event' or 'stride'.")

    try:
        _assert_is_dtype(stride_list, pd.DataFrame)
        _assert_has_multindex_cols(stride_list, expected=False)

        SL_INDEX = ["s_id"]
        stride_list = set_correct_index(stride_list, SL_INDEX)

        # Check if it has the correct columns
        all_columns = [*SL_COLS]
        _assert_has_columns(stride_list, [all_columns])

        start_event = {"min_vel": "min_vel", "ic": "ic"}
        # Check that the start time corresponds to the correct event
        if (
            start_event.get(stride_type, False)
            and len(stride_list) > 0
            and not np.array_equal(stride_list["start"].to_numpy(), stride_list[start_event[stride_type]].to_numpy())
        ):
            raise ValidationError(
                "For a {} stride list, the start column is expected to be identical to the {} column, "
                "but they are different.".format(stride_type, start_event[stride_type])
            )
        # Check that the stride ids are unique
        if not stride_list.index.nunique() == stride_list.index.size:
            raise ValidationError("The stride id of the stride list is expected to be unique.")

    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a SingleSensorStrideList. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False
    return True


def has_gyroscope_data(data: SensorData) -> bool:
    """Check if a dataset contains gyropcope data.

    Parameters
    ----------
    data: SensorData
        Dataset(s) to be checked.

    """
    type = is_sensor_data(data, check_gyr=False)
    if type == "single":
        return _has_gyroscope_data_single(data)
    else:
        res = {name: _has_gyroscope_data_single(data[name]) for name in get_multi_sensor_names(data)}
        return any(res.values())


def _has_gyroscope_data_single(data: SingleSensorData):
    return set(BF_GYR).issubset(data.columns) or set(SF_GYR).issubset(data.columns)


# ROTATIONs
def sliding_window_view(arr: np.ndarray, window_length: int, overlap: int, nan_padding: bool = False) -> np.ndarray:
    """Create a sliding window view of an input array with given window length and overlap.

    .. warning::
       This function will return by default a view onto your input array, modifying values in your result will directly
       affect your input data which might lead to unexpected behaviour! If padding is disabled (default) last window
       fraction of input may not be returned! However, if `nan_padding` is enabled, this will always return a copy
       instead of a view of your input data, independent if padding was actually performed or not!

    Parameters
    ----------
    arr : array with shape (n,) or (n, m)
        array on which sliding window action should be performed. Windowing
        will always be performed along axis 0.

    window_length : int
        length of desired window (must be smaller than array length n)

    overlap : int
        length of desired overlap (must be smaller than window_length)

    nan_padding: bool
        select if last window should be nan-padded or discarded if it not fits with input array length. If nan-padding
        is enabled the return array will always be a copy of the input array independent if padding was actually
        performed or not!

    Returns
    -------
    windowed view (or copy for nan_padding) of input array as specified, last window might be nan padded if necessary to
    match window size

    Examples
    --------
    >>> data = np.arange(0,10)
    >>> windowed_view = sliding_window_view(arr = data, window_length = 5, overlap = 3, nan_padding = True)
    >>> windowed_view
    array([[ 0.,  1.,  2.,  3.,  4.],
           [ 2.,  3.,  4.,  5.,  6.],
           [ 4.,  5.,  6.,  7.,  8.],
           [ 6.,  7.,  8.,  9., nan]])

    """
    if overlap >= window_length:
        raise ValueError("Invalid Input, overlap must be smaller than window length!")

    if window_length < 2:
        raise ValueError("Invalid Input, window_length must be larger than 1!")

    # calculate length of necessary np.nan-padding to make sure windows and overlaps exactly fits data length
    n_windows = np.ceil((len(arr) - window_length) / (window_length - overlap)).astype(int)
    pad_length = window_length + n_windows * (window_length - overlap) - len(arr)

    # had to handle 1D arrays separately
    if arr.ndim == 1:
        if nan_padding:
            # np.pad always returns a copy of the input array even if pad_length is 0!
            arr = np.pad(arr.astype(float), (0, pad_length), constant_values=np.nan)

        new_shape = (arr.size - window_length + 1, window_length)
    else:
        if nan_padding:
            # np.pad always returns a copy of the input array even if pad_length is 0!
            arr = np.pad(arr.astype(float), [(0, pad_length), (0, 0)], constant_values=np.nan)

        shape = (window_length, arr.shape[-1])
        n = np.array(arr.shape)
        o = n - shape + 1  # output shape
        new_shape = np.concatenate((o, shape), axis=0)

    # apply stride_tricks magic
    new_strides = np.concatenate((arr.strides, arr.strides), axis=0)
    view = np.lib.stride_tricks.as_strided(arr, new_shape, new_strides)[0 :: (window_length - overlap)]

    view = np.squeeze(view)  # get rid of single-dimensional entries from the shape of an array.

    return view


def _bool_fill(indices: np.ndarray, bool_values: np.ndarray, array: np.ndarray) -> np.ndarray:
    """Fill a preallocated array with bool_values.

    This method iterates over the indices and adds the values to the array at the given indices using a logical or.
    """
    for i in range(len(indices)):  # noqa: consider-using-enumerate
        index = indices[i]
        val = bool_values[i]
        index = index[~np.isnan(index)]
        # perform logical or operation to combine all overlapping window results
        array[index] = np.logical_or(array[index], val)
    return array


def normalize(v: np.ndarray) -> np.ndarray:
    """Simply normalize a vector.

    If a 2D array is provided, each row is considered a vector, which is normalized independently.
    In case an array has norm 0, np.nan is returned.

    Parameters
    ----------
    v : array with shape (3,) or (n, 3)
         vector or array of vectors

    Returns
    -------
    normalized vector or  array of normalized vectors

    Examples
    --------
    1D array

    >>> normalize(np.array([0, 0, 2]))
    array([0., 0., 1.])

    2D array

    >>> normalize(np.array([[2, 0, 0],[2, 0, 0]]))
    array([[1., 0., 0.],
           [1., 0., 0.]])

    0 Array:

    >>> normalize(np.array([0, 0, 0]))
    array([nan, nan, nan])

    """
    v = np.array(v)
    if len(v.shape) == 1:
        ax = 0
    else:
        ax = 1
    # We do not want a warning when we divide by 0 as we expect it
    with np.errstate(divide="ignore", invalid="ignore"):
        return (v.T / np.linalg.norm(v, axis=ax)).T


def is_almost_parallel_or_antiparallel(
    v1: np.ndarray, v2: np.ndarray, rtol: float = 1.0e-5, atol: float = 1.0e-8
) -> Union[np.bool_, np.ndarray]:
    """Check if two vectors are either parallel or antiparallel.

    Parameters
    ----------
    v1 : vector with shape (3,) or array of vectors
        axis ([x, y ,z]) or array of axis
    v2 : vector with shape (3,) or array of vectors
        axis ([x, y ,z]) or array of axis
    rtol : float
        The relative tolerance parameter
    atol : float
        The absolute tolerance parameter

    Returns
    -------
    bool or array of bool values with len n

    Examples
    --------
    two vectors each of shape (3,)

    >>> is_almost_parallel_or_antiparallel(np.array([0, 0, 1]), np.array([0, 0, 1]))
    True
    >>> is_almost_parallel_or_antiparallel(np.array([0, 0, 1]), np.array([0, 1, 0]))
    False

    array of vectors

    >>> is_almost_parallel_or_antiparallel(np.array([[0, 0, 1],[0,1,0]]), np.array([[0, 0, 2],[1,0,0]]))
    array([True,False])

    """
    return np.isclose(np.abs(row_wise_dot(normalize(v1), normalize(v2))), 1, rtol=rtol, atol=atol)


def find_random_orthogonal(v: np.ndarray) -> np.ndarray:
    """Find a unitvector in the orthogonal plane to v.

    Parameters
    ----------
    v : vector with shape (3,)
         axis ([x, y ,z])

    Returns
    -------
    vector which is either crossproduct with [0,1,0] or [1,0,0].

    Examples
    --------
    two vectors each of shape (3,)

    >>> find_random_orthogonal(np.array([1, 0, 0]))
    array([0, 0, 1])

    """
    if is_almost_parallel_or_antiparallel(v, np.array([1.0, 0, 0])):
        result = np.cross(v, [0, 1, 0])
    else:
        result = np.cross(v, [1, 0, 0])
    return normalize(result)


def find_orthogonal(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Return an orthogonal vector to 2 vectors.

    Parameters
    ----------
    v1 : vector with shape (3,)
         axis ([x, y ,z])
    v2 : vector with shape (3,)
         axis ([x, y ,z])

    Returns
    -------
        Returns the cross product of the two if they are not equal.

        Returns a random vector in the perpendicular plane if they are either parallel or antiparallel.
        (see :func:`find_random_orthogonal`

    Examples
    --------
    >>> find_orthogonal(np.array([1, 0, 0]),np.array([-1, 0, 0]))
    array([0, 0, -1])

    """
    if v1.ndim > 1 or v2.ndim > 1:
        raise ValueError("v1 and v2 need to be at max 1D (currently {}D and {}D".format(v1.ndim, v2.ndim))
    if is_almost_parallel_or_antiparallel(v1, v2):
        return find_random_orthogonal(v1)
    return normalize(np.cross(v1, v2))


def row_wise_dot(v1, v2, squeeze=False):
    """Calculate row wise dot product of two vectors."""
    v1, v2 = np.atleast_2d(v1, v2)
    out = np.sum(v1 * v2, axis=-1)
    if squeeze:
        return np.squeeze(out)
    return out


def find_unsigned_3d_angle(v1: np.ndarray, v2: np.ndarray) -> Union[np.ndarray, float]:
    """Find the angle (in rad) between two  3D vectors.

    Parameters
    ----------
    v1 : vector with shape (3,)  or array of vectors
        axis ([x, y ,z]) or array of axis
    v2 : vector with shape (3,) or array of vectors
        axis ([x, y ,z]) or array of axis

    Returns
    -------
        angle or array of angles between two vectors

    Examples
    --------
    two vectors: 1D

    >>> find_unsigned_3d_angle(np.array([-1, 0, 0]), np.array([-1, 0, 0]))
    0

    two vectors: 2D

    >>> find_unsigned_3d_angle(np.array([[-1, 0, 0],[-1, 0, 0]]), np.array([[-1, 0, 0],[-1, 0, 0]]))
    array([0,0])

    """
    v1_, v2_ = np.atleast_2d(v1, v2)
    v1_ = normalize(v1_)
    v2_ = normalize(v2_)
    out = np.arccos(row_wise_dot(v1_, v2_) / (np.linalg.norm(v1_, axis=-1) * np.linalg.norm(v2_, axis=-1)))
    if v1.ndim == 1:
        return np.squeeze(out)
    return out


def rotation_from_angle(axis: np.ndarray, angle: Union[float, np.ndarray]) -> Rotation:
    """Create a rotation based on a rotation axis and a angle.

    Parameters
    ----------
    axis : array with shape (3,) or (n, 3)
        normalized rotation axis ([x, y ,z]) or array of rotation axis
    angle : float or array with shape (n,)
        rotation angle or array of angeles in rad

    Returns
    -------
    rotation(s) : Rotation object with len n

    Examples
    --------
    Single rotation: 180 deg rotation around the x-axis

    >>> rot = rotation_from_angle(np.array([1, 0, 0]), np.deg2rad(180))
    >>> rot.as_quat().round(decimals=3)
    array([1., 0., 0., 0.])
    >>> rot.apply(np.array([[0, 0, 1.], [0, 1, 0.]])).round()
    array([[ 0., -0., -1.],
           [ 0., -1.,  0.]])

    Multiple rotations: 90 and 180 deg rotation around the x-axis

    >>> rot = rotation_from_angle(np.array([1, 0, 0]), np.deg2rad([90, 180]))
    >>> rot.as_quat().round(decimals=3)
    array([[0.707, 0.   , 0.   , 0.707],
           [1.   , 0.   , 0.   , 0.   ]])
    >>> # In case of multiple rotations, the first rotation is applied to the first vector
    >>> # and the second to the second
    >>> rot.apply(np.array([[0, 0, 1.], [0, 1, 0.]])).round()
    array([[ 0., -1.,  0.],
           [ 0., -1.,  0.]])

    """
    angle = np.atleast_2d(angle)
    axis = np.atleast_2d(axis)
    return Rotation.from_rotvec(np.squeeze(axis * angle.T))


def find_shortest_rotation(v1: np.ndarray, v2: np.ndarray) -> Rotation:
    """Find a quaternion that rotates v1 into v2 via the shortest way.

    Parameters
    ----------
    v1 : vector with shape (3,)
        axis ([x, y ,z])
    v2 : vector with shape (3,)
        axis ([x, y ,z])

    Returns
    -------
    rotation
        Shortest rotation that rotates v1 into v2

    Examples
    --------
    >>> goal = np.array([0, 0, 1])
    >>> start = np.array([1, 0, 0])
    >>> rot = find_shortest_rotation(start, goal)
    >>> rotated = rot.apply(start)
    >>> rotated
    array([0., 0., 1.])

    """
    if (not np.isclose(np.linalg.norm(v1, axis=-1), 1)) or (not np.isclose(np.linalg.norm(v2, axis=-1), 1)):
        raise ValueError("v1 and v2 must be normalized")
    axis = find_orthogonal(v1, v2)
    angle = find_unsigned_3d_angle(v1, v2)
    return rotation_from_angle(axis, angle)


def _rotate_sensor(
    data: SingleSensorData, rotation: Optional[Rotation], inplace: bool = False, gyr_avail: bool = True
) -> SingleSensorData:
    """Rotate the data of a single sensor with acc and gyro."""
    if inplace is False:
        data = data.copy()
    if rotation is None:
        return data
    if gyr_avail:
        data[SF_GYR] = rotation.apply(data[SF_GYR].to_numpy())
    data[SF_ACC] = rotation.apply(data[SF_ACC].to_numpy())
    return data


def rotate_dataset(
    dataset: SensorData, rotation: Union[Rotation, Dict[str, Rotation]], check_gyr: bool = False
) -> SensorData:
    """Apply a rotation to acc and gyro data of a dataset.

    Parameters
    ----------
    dataset
        dataframe representing a single or multiple sensors.
        In case of multiple sensors a df with MultiIndex columns is expected where the first level is the sensor name
        and the second level the axis names (all sensor frame axis must be present)
    rotation
        In case a single rotation object is passed, it will be applied to all sensors of the dataset.
        If a dictionary of rotations is applied, the respective rotations will be matched to the sensors based on the
        dict keys.
        If no rotation is provided for a sensor, it will not be modified.

    Returns
    -------
    rotated dataset
        This will always be a copy. The original dataframe will not be modified.

    Examples
    --------
    This will apply the same rotation to the left and the right foot

    >>> dataset = ...  # Sensordata with a left and a right foot sensor
    >>> rotate_dataset(dataset, rotation=rotation_from_angle(np.array([0, 0, 1]), np.pi))
    <copy of dataset with all axis rotated>

    This will apply different rotations to the left and the right foot

    >>> dataset = ...  # Sensordata with a left and a right foot sensor (sensors called "left" and "right")
    >>> rotate_dataset(dataset, rotation={'left': rotation_from_angle(np.array([0, 0, 1]), np.pi),
    ...     'right':rotation_from_angle(np.array([0, 0, 1]), np.pi / 2))
    <copy of dataset with all axis rotated>

    See Also
    --------
    gaitmap.utils.rotations.rotate_dataset_series: Apply a series of rotations to a dataset

    """
    dataset_type = is_sensor_data(dataset, frame="sensor", check_gyr=check_gyr)
    gyr_avail = has_gyroscope_data(dataset)
    if dataset_type == "single":
        if isinstance(rotation, dict):
            raise ValueError(
                "A Dictionary for the `rotation` parameter is only supported if a MultiIndex dataset (named sensors) is"
                " passed."
            )
        return _rotate_sensor(dataset, rotation, inplace=False, gyr_avail=gyr_avail)

    rotation_dict = rotation
    if not isinstance(rotation_dict, dict):
        rotation_dict = {k: rotation for k in get_multi_sensor_names(dataset)}

    if isinstance(dataset, dict):
        rotated_dataset = {**dataset}
        original_cols = None
    else:
        rotated_dataset = dataset.copy()
        original_cols = dataset.columns
    for key in rotation_dict.keys():
        test = _rotate_sensor(dataset[key], rotation_dict[key], inplace=False, gyr_avail=gyr_avail)
        rotated_dataset[key] = test

    if isinstance(dataset, pd.DataFrame):
        # Restore original order
        rotated_dataset = rotated_dataset[original_cols]
    return rotated_dataset


def convert_left_foot_to_fbf(data: SingleSensorData):
    """Convert the axes from the left foot sensor frame to the foot body frame (FBF).

    This function assumes that your dataset is already aligned to the gaitmap FSF.

    Parameters
    ----------
    data
        raw data frame containing acc and gyr data

    Returns
    -------
    converted data frame

    See Also
    --------
    gaitmap.utils.coordinate_conversion.convert_right_foot_to_fbf: conversion of right foot SingleSensorData
    gaitmap.utils.coordinate_conversion.convert_to_fbf: convert multiple sensors at the same time

    """
    is_single_sensor_data(data, check_gyr=False, frame="sensor", raise_exception=True)

    if "gyr_x" in data.columns:
        cols_bf = BF_COLS
    else:
        cols_bf = BF_ACC

    result = pd.DataFrame(columns=cols_bf)

    # Loop over all axes and convert each one separately
    for sf_col_name in data.columns:
        result[FSF_FBF_CONVERSION_LEFT[sf_col_name][1]] = FSF_FBF_CONVERSION_LEFT[sf_col_name][0] * data[sf_col_name]

    return result


def convert_right_foot_to_fbf(data: SingleSensorData):
    """Convert the axes from the right foot sensor frame to the foot body frame (FBF).

    This function assumes that your dataset is already aligned to the gaitmap FSF.

    Parameters
    ----------
    data
        raw data frame containing acc and gyr data

    Returns
    -------
    converted data frame

    See Also
    --------
    gaitmap.utils.coordinate_conversion.convert_left_foot_to_fbf: conversion of left foot SingleSensorData
    gaitmap.utils.coordinate_conversion.convert_to_fbf: convert multiple sensors at the same time

    """
    is_single_sensor_data(data, check_gyr=False, frame="sensor", raise_exception=True)

    if "gyr_x" in data.columns:
        cols_bf = BF_COLS
    else:
        cols_bf = BF_ACC

    result = pd.DataFrame(columns=cols_bf)

    # Loop over all axes and convert each one separately
    for sf_col_name in data.columns:
        result[FSF_FBF_CONVERSION_RIGHT[sf_col_name][1]] = FSF_FBF_CONVERSION_RIGHT[sf_col_name][0] * data[sf_col_name]

    return result


def _handle_foot(foot, foot_like, data, rot_func):
    result = dict()
    if foot_like:
        foot = [sensor for sensor in get_multi_sensor_names(data) if foot_like in sensor]
        if not foot:
            warnings.warn(
                "The substring {} is not contained in any sensor name. Available sensor names are: {}".format(
                    foot_like, get_multi_sensor_names(data)
                )
            )
    foot = foot or []
    for s in foot:
        if s not in data:
            raise KeyError("Sensordata contains no sensor with name " + s)
        result[s] = rot_func(data[s])
    return result


def convert_to_fbf(
    data: MultiSensorData,
    left: Optional[List[str]] = None,
    right: Optional[List[str]] = None,
    right_like: str = None,
    left_like: str = None,
):
    """Convert the axes from the sensor frame to the body frame for one MultiSensorDataset.

    This function assumes that your dataset is already aligned to the gaitmap FSF.
    Sensors that should not be transformed are kept untouched.
    Note, that the column names of all transformed dataset is changed to the respective body frame names.

    This function can handle multiple left and right sensors at the same time.

    Parameters
    ----------
    data
        MultiSensorDataset
    left
        List of strings indicating sensor names which will be rotated using definition of left conversion.
        This option can not be used in combination with `left_like`.
    right
        List of strings indicating sensor names which will be rotated using definition of right conversion
        This option can not be used in combination with `right_like`.
    left_like
        Consider all sensors containing this string in the name as left foot sensors.
        This option can not be used in combination with `left`.
    right_like
        Consider all sensors containing this string in the name as right foot sensors.
        This option can not be used in combination with `right`.


    Returns
    -------
    converted MultiSensorDataset

    Examples
    --------
    These examples assume that your dataset has two sensors called `left_sensor` and `right_sensor`.

    >>> dataset = ... # Sensordata in FSF
    >>> fbf_dataset = convert_to_fbf(dataset, left_like="left_", right_like="right_")

    Alternatively, you can specify the full sensor names.

    >>> fbf_dataset = convert_to_fbf(dataset, left=["left_sensor"], right_sensor=["right_sensor"])

    See Also
    --------
    gaitmap.utils.coordinate_conversion.convert_left_foot_to_fbf: conversion of left foot SingleSensorData
    gaitmap.utils.coordinate_conversion.convert_right_foot_to_fbf: conversion of right foot SingleSensorData

    """
    if not is_multi_sensor_data(data, frame="sensor", check_gyr=False):
        raise ValueError("No valid FSF MultiSensorDataset supplied.")

    if (left and left_like) or (right and right_like) or not any((left, left_like, right, right_like)):
        raise ValueError(
            "You need to either supply a list of names via the `left` or `right` arguments, or a single string for the "
            "`left_like` or `right_like` arguments, but not both!"
        )

    left_foot = _handle_foot(left, left_like, data, rot_func=convert_left_foot_to_fbf)
    right_foot = _handle_foot(right, right_like, data, rot_func=convert_right_foot_to_fbf)

    sensor_names = get_multi_sensor_names(data)
    result = {k: data[k] for k in sensor_names}
    result = {**result, **left_foot, **right_foot}

    # If original data is not synchronized (dictionary), return as dictionary
    if isinstance(data, dict):
        return result
    # For synchronized sensors, return as MultiIndex dataframe
    df = pd.concat(result, axis=1)
    # restore original order
    return df[sensor_names]
