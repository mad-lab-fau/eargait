"""Feauture Extractor for Spatial Parameter Estimation Using ML Regressors."""
from typing import Dict, Union

import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy import signal
from scipy.fft import fft

from eargait.utils.helper_datatype import EventList, SingleSensorEventList, is_event_list
from eargait.utils.helper_gaitmap import BF_GYR, SensorData, SingleSensorData
from eargait.utils.helpers import butter_lowpass_filter


class FeatureExtractor:
    """Feature Extractor for Spatial Parameter Estimation Using ML Regressors."""

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def get_features(self, data: SensorData, gait_events: EventList) -> Union[Dict, pd.DataFrame]:
        kind = is_event_list(gait_events)
        if kind == "single":
            features = self._get_features_single(data, gait_events)
        else:
            features = {}
            for sensor, events in gait_events.items():
                features[sensor] = self._get_features_single(data[sensor], events)
                assert features[sensor].shape[0] != 0
        return features

    def get_ground_truth(self):
        raise NotImplementedError()

    def _get_features_single(self, data: SingleSensorData, gait_events: SingleSensorEventList) -> pd.DataFrame:
        return self._calculate_features(data, gait_events)

    def _calculate_features(self, data: SingleSensorData, gait_events: SingleSensorEventList):
        # butterworth lowpass filter
        filtered_data = butter_lowpass_filter(data, 20, self.sample_rate / 2, 4)
        filtered_data = filtered_data.drop(columns=BF_GYR, errors="ignore")

        # dominant feq amplitude
        amplitude = pd.Series(self._amplitude_dominant_freq(filtered_data))
        amplitude = amplitude.add_prefix("dom_freq_")

        # cadence
        cadence = pd.Series((data.index[-1] - data.index[0]) / self.sample_rate / gait_events.shape[0], index=["cad"])

        df_feat = pd.DataFrame()
        for i in range(gait_events.shape[0] - 1):
            start = gait_events["ic"].iloc[i]
            ende = gait_events["ic"].iloc[i + 1]
            if np.isnan(start) or np.isnan(ende):
                # raise ValueError("Nan start or stop. Analyse here what to do")
                nans = np.zeros((df_feat.shape[0]))
                nans[:] = np.nan
                features = pd.DataFrame(nans, index=df_feat.index, columns=[gait_events.index[i]])
                df_feat = pd.concat([df_feat, features], axis=1)
                continue

            filt_data_step = filtered_data.loc[int(start) : int(ende)]

            # Norm of ACC
            data_norm = pd.DataFrame(data=norm(filt_data_step, axis=1), columns=["acc_norm"])

            # get statistical signal characetistics (mean, median ...)
            feat_filt = self._signal_features_of_step(filt_data_step)
            feat_norm = self._signal_features_of_step(data_norm)

            # concatinate all features
            features = pd.concat([feat_filt, feat_norm, cadence, amplitude])
            features.name = gait_events.index[i]
            df_feat = pd.concat([df_feat, features], axis=1)
        return df_feat.T

    @staticmethod
    def _signal_features_of_step(data: pd.DataFrame, pre_fix: str = None):
        features = pd.Series(dtype=float)
        describe = data.describe()
        features = pd.concat([features, describe.loc["mean"].add_prefix("mean_")])
        features = pd.concat([features, describe.loc["std"].add_prefix("std_")])
        features = pd.concat([features, data.kurtosis(axis=0).add_prefix("kurtosis_")])
        features = pd.concat([features, data.median(axis=0).add_prefix("median_")])
        features = pd.concat([features, (describe.loc["max"] - describe.loc["min"]).abs().add_prefix("range_")])
        features = pd.concat([features, data.abs().sum(axis=0).add_prefix("sum_")])
        features = pd.concat([features, data.pow(2).sum(axis=0).add_prefix("sum_sqrd_")])
        features = pd.concat([features, data.diff(axis=0).abs().mean(axis=0).add_prefix("abs_deriv")])

        if "acc_si" in data.columns:
            cc = {}
            cc["cc_si_pa"] = np.max(signal.correlate(data["acc_si"], data["acc_pa"]))
            cc["cc_si_ml"] = np.max(signal.correlate(data["acc_si"], data["acc_ml"]))
            cc["cc_pa_ml"] = np.max(signal.correlate(data["acc_pa"], data["acc_ml"]))
            features = pd.concat([features, pd.Series(cc)])
        zc = {}
        for col in data.columns:
            key = "zc_" + col
            tmp = np.where(np.diff(np.signbit(data[col])))[0]
            zc[key] = tmp[0] if tmp.shape[0] != 0 else 0
        features = pd.concat([features, pd.Series(zc)])

        if pre_fix is not None:
            features = features.add_prefix(pre_fix + "_")
        return features

    def _amplitude_dominant_freq(self, data):
        locomotion_band = (0.3, 4)
        lower_bound = int(np.floor(self.sample_rate / locomotion_band[1]))
        # print("lower bound:", lower_bound)
        # (sampling_rate / locomotion_band_lower = upper bound of autocorrelation)
        upper_bound = int(np.ceil(self.sample_rate / locomotion_band[0]))
        ampl = {}
        for col in data.columns:
            signal_1d = data[col].to_numpy()
            signal_1d = np.reshape(signal_1d, (1, signal_1d.shape[0]))
            # autocorr from 0-upper motion band
            auto_corr = np.empty((signal_1d.shape[0], upper_bound + 1))
            for tau in range(upper_bound + 1):
                tmax = signal_1d.shape[1] - tau
                umax = signal_1d.shape[1] + tau
                auto_corr[:, tau] = (signal_1d[:, :tmax] * signal_1d[:, tau:umax]).sum(axis=1)
            # calculate dominant frequency in Hz
            dominant_frequency = (
                1 / (np.argmax(auto_corr[:, lower_bound:], axis=-1) + lower_bound).astype(float) * self.sample_rate
            )
            # spectrum
            signal_1d = signal_1d.reshape(-1)
            x_fft = fft(signal_1d)
            n_fft = signal_1d.shape[0]
            n = np.arange(n_fft)
            t_fft = n_fft / self.sample_rate
            freq = n / t_fft
            mini = np.argmin(np.abs(freq - dominant_frequency[0]))
            ampl[col] = np.abs(x_fft)[mini]
        return ampl


class FeatureExtractorDemographics(FeatureExtractor):
    """Feature Extractor for Spatial Parameter Estimation Using ML Regressors."""

    def __init__(self, sample_rate: int, height: int, gender: str, age: int, weight: int):
        self.height = height
        self.gender = gender
        self.age = age
        self.weight = weight
        super().__init__(sample_rate)

    @property
    def demographic_features(self):
        if self.gender not in ["f", "m", "w"]:
            raise ValueError("Gender not in ['f', 'm', 'w]")
        gen = 0 if self.gender == "m" else 1
        return pd.Series(data=[self.height, self.age, gen, self.weight], index=["height", "age", "gender", "weight"])

    def _get_features_single(self, data: SingleSensorData, gait_events: SingleSensorEventList) -> pd.DataFrame:
        features = self._calculate_features(data, gait_events)
        demos_df = pd.concat([self.demographic_features.to_frame().T] * features.shape[0])
        demos_df.index = features.index
        features = pd.concat([features, demos_df], axis=1)
        return features
