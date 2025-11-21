import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from config import *

def resample_curve(y, new_len=100):
    x_old = np.linspace(0, 1, len(y))
    x_new = np.linspace(0, 1, new_len)
    f = interp1d(x_old, y, kind='cubic')
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
    oryginal_signal = adc_data.adc_normalized_data[target_adc].copy()
    non_time_outlier_signal = adc_data.non_time_outlier_adc_data
    non_amplitude_outlier_signal = adc_data.non_amplitude_outlier_adc_data

    for i in range(len(oryginal_signal)):
        if np.isnan(non_time_outlier_signal[i]) or np.isnan(non_amplitude_outlier_signal[i]):
            oryginal_signal[i] = np.nan

    if adc_data.plot_enabled:
        plt.plot(adc_data.timestamps, adc_data.adc_normalized_data[target_adc], label='Original Signal', color='gray')
        plt.plot(adc_data.timestamps, oryginal_signal, label='Cleaned Signal', color='blue')
        plt.title("Clean adc_normalized_data")
        plt.legend()
        plt.show()

    clen_adc_normalized_data = ([], [])
    for i in range(len(oryginal_signal)):
        if not np.isnan(oryginal_signal[i]):
            clen_adc_normalized_data[0].append(adc_data.timestamps[i])
            clen_adc_normalized_data[1].append(oryginal_signal[i])

    resampled_signal = resample_curve(clen_adc_normalized_data[1], RESAMPLE_NODE_COUNT)
    
    if adc_data.plot_enabled:
        print(f"new node lenght: {len(clen_adc_normalized_data[0])}")
        plt.plot(clen_adc_normalized_data[0], clen_adc_normalized_data[1])
        plt.title("Cleaned adc_normalized_data")
        plt.show()

        plt.plot(np.linspace(0, 1, len(resampled_signal)), resampled_signal)
        plt.title("Resampled cleaned adc_normalized_data")
        plt.show()

    adc_data.cleaned_and_resampled_adc_data = resampled_signal

def outlier_detection(adc_data, target_adc):
    time_outliers(adc_data, target_adc)
    amplitude_outliers(adc_data, target_adc)
    remove_outliers_and_remake_signal(adc_data, target_adc)