import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import copy
from config import *

def resample_data(y, new_len=100):
    x_old = np.linspace(0, 1, len(y))
    x_new =  np.linspace(0, 1, new_len)
    # f = spi.CubicSpline(x_old, old_y)
    f = spi.interp1d(x_old, y, kind='cubic')
    return f(x_new)

def time_outliers(adc_data, target_adc):
    
    # Extract breath durations from minima timestamps
    breath_durations = []
    minima_timestamps = []
    for minimum in adc_data.signal_minima:
        minima_timestamps.append(adc_data.timestamps[minimum])

    for i in range(1, len(minima_timestamps)):
        duration = minima_timestamps[i] - minima_timestamps[i-1]
        breath_durations.append((minima_timestamps[i-1], duration))

    # Enumerate outlier breaths based on duration percentiles
    lower_bound = np.percentile(breath_durations, PERCENTILE_THRESHOLD)
    upper_bound = np.percentile(breath_durations, 100 - PERCENTILE_THRESHOLD)

    # timestamps and durations of breaths to be deleted
    outlier_breaths = []
    for breath in breath_durations:
        if breath[1] < lower_bound or breath[1] > upper_bound:  
            outlier_breaths.append(breath)

    # discarding outlier breaths from adc_data (put them in non_outlier_adc_data)
    oryginal_signal = adc_data.adc_normalized_data[target_adc].copy()
    non_outlier_signal= adc_data.adc_normalized_data[target_adc].copy()
    time_outlier_signal = np.full_like(oryginal_signal, np.nan)

    for breath in outlier_breaths:
        outlier_start_time = breath[0]
        outlier_end_time = breath[0] + breath[1]

        start_index = np.searchsorted(adc_data.timestamps, outlier_start_time)
        end_index = np.searchsorted(adc_data.timestamps, outlier_end_time)

        time_outlier_signal[start_index:end_index] = oryginal_signal[start_index:end_index]
        non_outlier_signal[start_index:end_index] = np.nan

    adc_data.non_time_outlier_adc_data = non_outlier_signal
    adc_data.time_outlier_adc_data = time_outlier_signal

    if adc_data.plot_enabled:
        plt.title("Time outliers")
        plt.plot(adc_data.timestamps, oryginal_signal, label='Original Signal', color='gray')
        plt.plot(adc_data.timestamps, non_outlier_signal, label='Non-outlier Signal', color="green")
        plt.plot(adc_data.timestamps, time_outlier_signal, label='Outlier Signal', color='red')
        plt.legend()
        plt.show()

def amplitude_outliers(adc_data, target_adc):
    minima_indices = adc_data.signal_minima
    maxima_indices = adc_data.signal_maxima

    breath_segments = []
    for i in range(len(minima_indices) - 1):
        min_idx = minima_indices[i]
        max_idx = maxima_indices[i] if i < len(maxima_indices) else min_idx
        next_min_idx = minima_indices[i + 1]

        start_time = adc_data.timestamps[min_idx]
        end_time = adc_data.timestamps[next_min_idx]
        duration = end_time - start_time

        amplitude = adc_data.adc_normalized_data[target_adc][max_idx] - adc_data.adc_normalized_data[target_adc][min_idx]
        breath_segments.append((start_time, duration, min_idx, max_idx, next_min_idx, amplitude))

    amplitudes = [bs[5] for bs in breath_segments]
    if len(amplitudes) == 0:
        adc_data.non_amplitude_outlier_adc_data = adc_data.adc_normalized_data[target_adc].copy()
        adc_data.amplitude_outlier_adc_data = np.full_like(adc_data.adc_normalized_data[target_adc], np.nan)
        return

    lower_bound = np.percentile(amplitudes, PERCENTILE_THRESHOLD)
    upper_bound = np.percentile(amplitudes, 100 - PERCENTILE_THRESHOLD)

    # detect outlier breaths by amplitude
    outlier_breaths = []
    for breath in breath_segments:
        if breath[5] < lower_bound or breath[5] > upper_bound:
            outlier_breaths.append(breath)

    original_signal = adc_data.adc_normalized_data[target_adc].copy()
    non_outlier_signal = original_signal.copy()
    amplitude_outlier_signal = np.full_like(original_signal, np.nan)

    for breath in outlier_breaths:
        outlier_start_time = breath[0]
        outlier_end_time = breath[0] + breath[1]

        start_index = np.searchsorted(adc_data.timestamps, outlier_start_time)
        end_index = np.searchsorted(adc_data.timestamps, outlier_end_time)

        amplitude_outlier_signal[start_index:end_index] = original_signal[start_index:end_index]
        non_outlier_signal[start_index:end_index] = np.nan

    # store results on adc_data
    adc_data.non_amplitude_outlier_adc_data = non_outlier_signal
    adc_data.amplitude_outlier_adc_data = amplitude_outlier_signal

    if adc_data.plot_enabled:
        plt.title("Amplitude outliers")
        plt.plot(adc_data.timestamps, original_signal, label='Original Signal', color='gray')
        plt.plot(adc_data.timestamps, non_outlier_signal, label='Non-outlier Signal', color='green')
        plt.plot(adc_data.timestamps, amplitude_outlier_signal, label='Outlier Signal', color='red')
        plt.legend()
        plt.show()

def remove_outliers_and_remake_signal(adc_data, target_adc):
    oryginal_signal = copy.deepcopy(adc_data.adc_normalized_data)
    non_time_outlier_signal = adc_data.non_time_outlier_adc_data
    non_amplitude_outlier_signal = adc_data.non_amplitude_outlier_adc_data

    for i in range(len(oryginal_signal[target_adc])):
        if np.isnan(non_time_outlier_signal[i]) or np.isnan(non_amplitude_outlier_signal[i]):
            for j in range(ADC_COUNT):
                oryginal_signal[j][i] = np.nan

    if adc_data.plot_enabled:
        plt.plot(adc_data.timestamps, adc_data.adc_normalized_data[target_adc], label='Original Signal', color='gray')
        plt.plot(adc_data.timestamps, oryginal_signal[target_adc], label='Cleaned Signal', color='blue')
        plt.title("Clean adc_normalized_data")
        plt.legend()
        plt.show()


    clen_adc_normalized_data = ([], [], [], [], [], [])
    for i in range(len(oryginal_signal[target_adc])):
        if not np.isnan(oryginal_signal[target_adc][i]):
            clen_adc_normalized_data[0].append(adc_data.timestamps[i])
            for j in range(ADC_COUNT):
                clen_adc_normalized_data[j+1].append(oryginal_signal[j][i])

    # resampling the signal BUT
    # we go like this:
    # for each NaN filled hole we calculate it's length and subtract it from timestamps of all nodes after it
    # only after we get this NaN free signal we resample it to RESAMPLE_NODE_COUNT nodes
    nan_adjusted_timestamps = []
    nan_adjusted_data = [], [], [], [], []
    total_time_shift = 0
    first_nan_timestamp = None
    for i in range(len(adc_data.timestamps)):
        if not np.isnan(oryginal_signal[target_adc][i]):
            if first_nan_timestamp is not None:
                time_shift = adc_data.timestamps[i] - first_nan_timestamp
                total_time_shift += time_shift
                adjusted_timestamp = adc_data.timestamps[i] - total_time_shift
                nan_adjusted_timestamps.append(adjusted_timestamp)
                print(f"Adjusted timestamp: {adjusted_timestamp}, original: {adc_data.timestamps[i]}, total_time_shift: {total_time_shift}")
                for j in range(ADC_COUNT):
                    nan_adjusted_data[j].append(oryginal_signal[j][i])
                first_nan_timestamp = None
            else:
                adjusted_timestamp = adc_data.timestamps[i] - total_time_shift
                nan_adjusted_timestamps.append(adjusted_timestamp)
                for j in range(ADC_COUNT):
                    nan_adjusted_data[j].append(oryginal_signal[j][i])
        else:
            if first_nan_timestamp is None:
                first_nan_timestamp = adc_data.timestamps[i]
            

    if adc_data.plot_enabled:        
        plt.plot(nan_adjusted_timestamps, nan_adjusted_data[target_adc], label='Cleaned Signal', color='blue')
        plt.title("NaN adjusted timestamps")
        plt.legend()
        plt.show()

    # resampling and smoothing the data while keeping the same timestamps
    signal_duration = nan_adjusted_timestamps[-1] - nan_adjusted_timestamps[0]
    resampled_node_count = int(signal_duration // 100)
    resampled_data = [[] for _ in range(ADC_COUNT)]
    for i in range(ADC_COUNT):
        resampled_data[i] = resample_data(nan_adjusted_data[i], resampled_node_count)
    resampled_timestamps = resample_data(nan_adjusted_timestamps, resampled_node_count)
    if adc_data.plot_enabled:
        plt.plot(nan_adjusted_timestamps, nan_adjusted_data[target_adc], label='Cleaned Signal', color='blue')
        plt.plot(resampled_timestamps, resampled_data[target_adc], label='Resampled Signal', color='orange')
        plt.title("Resampled Signal")
        plt.legend()
        plt.show()

    adc_data.final_adc_data = resampled_data
    adc_data.final_adc_timestamps = resampled_timestamps

def outlier_detection(adc_data, target_adc):
    time_outliers(adc_data, target_adc)
    amplitude_outliers(adc_data, target_adc)
    remove_outliers_and_remake_signal(adc_data, target_adc)