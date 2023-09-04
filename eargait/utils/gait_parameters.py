"""Set of functions to estimate temporal gait parameters."""
import warnings
from typing import Dict, Union

import numpy as np
import pandas as pd

from eargait.utils.helper_datatype import EventList, SingleSensorEventList, is_event_list


def get_temporal_params(event_list: EventList, sampling_rate_hz: int) -> Union[Dict, pd.DataFrame]:
    """Get temporal parameters of event list.

    Parameters
    ----------
    event_list
        Detected event list
    sampling_rate_hz
        Sampling rate of data
    event_consistency:
        Delete temporal parameters of events which are not adjecent events in gait sequence.

    Returns
    -------
    temporal_params
        pd.DataFrames containing temporal gait parameters

    """
    kind = is_event_list(event_list)
    if kind == "single":
        return _get_single_temporal_params(event_list, sampling_rate_hz)
    temporal_params = {}
    for pos in event_list.keys():
        temporal_params[pos] = _get_single_temporal_params(event_list[pos], sampling_rate_hz)
    return temporal_params


def get_average_temporal_params(temporal_params: Union[Dict, pd.DataFrame]) -> Union[Dict, pd.DataFrame]:
    """Average temporal gait events (event, stance and swing time)."""
    return get_average_params(temporal_params)


def _get_single_temporal_params(
    event_list: SingleSensorEventList,
    sampling_rate_hz: int,
) -> pd.DataFrame:

    df = pd.DataFrame(index=event_list.index, columns=["stride_time", "stance_time", "swing_time"])
    for lateral in ["ipsilateral", "contralateral"]:
        tmp = event_list.loc[event_list.side == lateral]
        df.loc[event_list.side == lateral] = _get_single_temporal_params_single_lateral(tmp, sampling_rate_hz)
    for col in df.columns:
        df[col] = df[col].astype(float)
    df["step_time"] = event_list.ic.diff() / sampling_rate_hz
    df["side"] = event_list.side
    return df


def _get_single_temporal_params_single_lateral(
    event_list: SingleSensorEventList, sampling_rate_hz: int
) -> pd.DataFrame:
    df = pd.DataFrame(index=event_list.index, columns=["stride_time", "stance_time", "swing_time"])
    index = df.index[1::]
    df["stride_time"] = pd.Series(
        (event_list.ic.diff().shift(-1).iloc[0:-1] / sampling_rate_hz).to_numpy(), index=index
    )
    stance = pd.Series(
        (event_list.tc.iloc[1:].to_numpy() - event_list.ic.iloc[0:-1].to_numpy()) / sampling_rate_hz,
        index=index,
    )
    df["swing_time"] = pd.Series(
        np.array((event_list.ic.iloc[1::] - event_list.tc.iloc[1::])) / sampling_rate_hz, index=index
    )
    df["stance_time"] = stance
    _sanity_check_temporal_parameters(df)
    return df


def get_average_params(params: Union[Dict, pd.DataFrame]) -> Union[Dict, pd.DataFrame]:
    """Average temporal gait events (stride, stance and swing time)."""
    if not isinstance(params, Dict):
        if "side" in params.columns:
            params = params.drop(columns="side").astype(float)
        return params.describe().loc[["mean", "std"]]
    # else: multi sensor
    average_params = {}
    for pos, param in params.items():  # noqa
        if "side" in param.columns:
            param = param.drop(columns="side").astype(float)
        average_params[pos] = param.describe().loc[["mean", "std"]]
    return average_params


def get_average_spatial_params(spatial_params: Union[Dict, pd.DataFrame]) -> Union[Dict, pd.DataFrame]:
    """Average spatial parametets (stride_length, step_length)."""
    return get_average_params(spatial_params)


def combine_spatial_temporal(temporal, spatial):
    if not isinstance(temporal, Dict):
        return _combine_single_spatial_temporal(temporal, spatial)
    temporal_spatial = {}
    for sensor in temporal.keys():
        temporal_spatial[sensor] = _combine_single_spatial_temporal(temporal[sensor], spatial[sensor])
    return temporal_spatial


def get_average_spatiotemp_params(spatiotemp_params: Union[Dict, pd.DataFrame]) -> Union[Dict, pd.DataFrame]:
    """Average spatiotemporal parameter."""
    return get_average_params(spatiotemp_params)


def _combine_single_spatial_temporal(temp, spat):
    assert temp.side.equals(spat.side)
    spat = spat.drop(columns=["side"])
    return pd.merge(temp, spat, left_index=True, right_index=True)


def _check_stride_consistency(df: pd.DataFrame, event_list_type):
    """Check for sequential stride index and removes entries with missing index inbetween."""
    if event_list_type == "upper_body_sensor_unilateral":
        diff_index = (df.index.to_series().diff() == 1) | (df.index.to_series().diff() > 2)
    else:
        diff_index = df.index.to_series().diff() > 1
    df[diff_index] = np.nan
    return df


def _sanity_check_temporal_parameters(tmp: pd.DataFrame):
    df = tmp.copy()
    df = df.dropna(axis=0)
    assert np.round((df["swing_time"] + df["stance_time"]), 4).equals(np.round(df["stride_time"], 4))
    info = df.describe()
    if info.loc["mean"].stride_time < 0.8 or info.loc["mean"].stride_time > 1.5:
        message = "Unrealistic average stride time of  {:.2f}s".format(info.loc["mean"].stride_time)  # noqa
        warnings.warn(message)
    if info.loc["mean"].swing_time < 0.25 or info.loc["mean"].swing_time > 0.8:
        message = "Unrealistic average swing time of  {:.2f}s".format(info.loc["mean"].swing_time)  # noqa
        warnings.warn(message)
    if info.loc["mean"].stance_time < 0.4 or info.loc["mean"].stance_time > 1.0:
        message = "Unrealistic average stance time of  {:.2f}s".format(info.loc["mean"].stance_time)  # noqa
        warnings.warn(message)
