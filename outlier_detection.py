import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

PERCENTILE_THRESHOLD = 15  # % for both lower and upper bounds

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

    # Enumerate outlier breaths based on a boxplot method
    # q1 = np.percentile([bd[1] for bd in breath_durations], 25)
    # q3 = np.percentile([bd[1] for bd in breath_durations], 75)
    # iqr = q3 - q1
    # lower_bound = q1 - 1.5 * iqr
    # upper_bound = q3 + 1.5 * iqr

    # timestamps and durations of breaths to be deleted
    outlier_breaths = []
    for breath in breath_durations:
        if breath[1] < lower_bound or breath[1] > upper_bound:  
            outlier_breaths.append(breath)

    # print(outlier_breaths)

    # discarding outlier breaths from adc_data (put them in non_outlier_adc_data)
    adc_data.non_outlier_adc_data = adc_data.adc_normalized_data[target_adc].copy()
    adc_data.time_outlier_adc_data = adc_data.adc_normalized_data[target_adc].copy() * np.nan  # Initialize with NaNs
    for breath in outlier_breaths:
        outlier_start_time = breath[0]
        outlier_end_time = breath[0] + breath[1]

        start_index = np.searchsorted(adc_data.timestamps, outlier_start_time)
        end_index = np.searchsorted(adc_data.timestamps, outlier_end_time)

        # remove outlier segment by setting it to nan
        adc_data.time_outlier_adc_data[start_index:end_index] = adc_data.non_outlier_adc_data[start_index:end_index]
        adc_data.non_outlier_adc_data[start_index:end_index] = np.nan
    if adc_data.plot_enabled:
        plt.plot(adc_data.timestamps, adc_data.adc_normalized_data[target_adc], label='Original')
        for breath in outlier_breaths:
            plt.axvspan(breath[0], breath[0] + breath[1], color='red', alpha=0.3)
        plt.legend()
        plt.show()

def resample_curve(y, new_len=100):
    x_old = np.linspace(0, 1, len(y))
    x_new = np.linspace(0, 1, new_len)
    f = interp1d(x_old, y, kind='cubic')
    return f(x_new)

def amplitude_outliers(adc_data, target_adc):
    # TODO: we need to review top and bottom percentiles, we get rid of too many good tall ones and too little shrt things that lower the averages
    breath_amplitudes = []
    minima_indices = adc_data.signal_minima
    maxima_indices = adc_data.signal_maxima
    for i in range(len(minima_indices)-1):
        min_idx = minima_indices[i]
        max_idx = maxima_indices[i]
        next_min_idx = minima_indices[i+1]

        amplitude = adc_data.adc_normalized_data[target_adc][max_idx] - adc_data.adc_normalized_data[target_adc][min_idx]
        # print(f"Breath {i}: Amplitude = {amplitude}, indices {min_idx} to {next_min_idx}")
        breath_amplitudes.append((min_idx, max_idx, next_min_idx, amplitude))

    amplitudes = [ba[3] for ba in breath_amplitudes]
    lower_bound = np.percentile(amplitudes, PERCENTILE_THRESHOLD)
    upper_bound = np.percentile(amplitudes, 100 - PERCENTILE_THRESHOLD)
    if adc_data.plot_enabled:
        # for now plot adc_data and mark breaths with amplitude outliers
        plt.plot(adc_data.timestamps, adc_data.adc_normalized_data[target_adc], label='Original Signal')
        for breath in breath_amplitudes:
            if breath[3] < lower_bound or breath[3] > upper_bound:
                plt.axvspan(adc_data.timestamps[breath[0]], adc_data.timestamps[breath[2]], color='red', alpha=0.3)
                # print(f"Outlier Breath: Amplitude = {breath[3]} at indices {breath[0]} to {breath[2]}")

        plt.legend()
        plt.show()

def outlier_detection(adc_data, target_adc):
    time_outliers(adc_data, target_adc)
    amplitude_outliers(adc_data, target_adc)