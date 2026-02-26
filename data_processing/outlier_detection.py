from config import *

def calculate_breaths(adc_data, target_adc):
    local_minima = adc_data.signal_minima
    breaths = []
    for i in range(len(local_minima) - 1):
        breath = adc_data.adc_normalized_data[target_adc][local_minima[i]:local_minima[i + 1]]
        max_index = adc_data.signal_maxima[np.where((adc_data.signal_maxima > local_minima[i]) & (adc_data.signal_maxima < local_minima[i + 1]))]
        # print(f"max_index: {max_index}, max_value: {breath[max_index - local_minima[i]]}")
        max_timestamp = adc_data.timestamps[max_index]

        breaths.append({
            "breath": breath,
            "start_index": local_minima[i],
            "end_index": local_minima[i + 1],
            "start_timestamp": adc_data.timestamps[local_minima[i]],
            "end_timestamp": adc_data.timestamps[local_minima[i + 1]],
            "timestamps": adc_data.timestamps[local_minima[i]:local_minima[i + 1]],
            "duration": adc_data.timestamps[local_minima[i + 1]] - adc_data.timestamps[local_minima[i]],
            "max_index": max_index,
            "max_value": breath[max_index - local_minima[i]],
            "max_timestamp": max_timestamp,
            "amplitude": np.max(breath) - np.min(breath)
        })
    return breaths

def resample_data(y, new_len=100):
    x_old = np.linspace(0, 1, len(y))
    x_new =  np.linspace(0, 1, new_len)
    # f = spi.CubicSpline(x_old, old_y)
    f = spi.interp1d(x_old, y, kind='cubic')
    return f(x_new)

def time_outliers(adc_data, target_adc, breaths):
    breath_durations = [(breath["start_timestamp"], breath["duration"]) for breath in breaths]
    lower_bound = np.percentile([d[1] for d in breath_durations], PERCENTILE_THRESHOLD)
    upper_bound = np.percentile([d[1] for d in breath_durations], 100 - PERCENTILE_THRESHOLD)

    outlier_breaths = []
    for breath in breaths:
        if breath["duration"] < lower_bound or breath["duration"] > upper_bound:  
            outlier_breaths.append(breath)

    original_signal = adc_data.adc_normalized_data[target_adc].copy()
    non_outlier_signal= adc_data.adc_normalized_data[target_adc].copy()
    time_outlier_signal = np.full_like(original_signal, np.nan)
    # full_like - creates an array of the same shape as the original

    for outlier_breath in outlier_breaths:
        start_index = outlier_breath["start_index"]
        end_index = outlier_breath["end_index"]

        time_outlier_signal[start_index:end_index] = original_signal[start_index:end_index]
        non_outlier_signal[start_index:end_index] = np.nan

    adc_data.non_time_outlier_adc_data = non_outlier_signal
    adc_data.time_outlier_adc_data = time_outlier_signal

    if adc_data.plot_enabled:
        plt.title("Time outliers")
        plt.plot(adc_data.timestamps, original_signal, label='Original Signal', color='gray')
        plt.plot(adc_data.timestamps, non_outlier_signal, label='Non-outlier Signal', color="green")
        plt.plot(adc_data.timestamps, time_outlier_signal, label='Outlier Signal', color='red')
        plt.legend()
        plt.show()

def amplitude_outliers(adc_data, target_adc, breaths):
    amplitudes = [breath["amplitude"] for breath in breaths]
    lower_bound = np.percentile(amplitudes, PERCENTILE_THRESHOLD)
    upper_bound = np.percentile(amplitudes, 100 - PERCENTILE_THRESHOLD)

    outlier_breaths = []
    for outlier_breath in breaths:
        if outlier_breath["amplitude"] < lower_bound or outlier_breath["amplitude"] > upper_bound:
            outlier_breaths.append(outlier_breath)

    original_signal = adc_data.adc_normalized_data[target_adc].copy()
    non_outlier_signal = original_signal.copy()
    amplitude_outlier_signal = np.full_like(original_signal, np.nan)

    for outlier_breath in outlier_breaths:
        start_index = outlier_breath["start_index"]
        end_index = outlier_breath["end_index"]

        amplitude_outlier_signal[start_index:end_index] = original_signal[start_index:end_index]
        non_outlier_signal[start_index:end_index] = np.nan
        
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
    original_signal = copy.deepcopy(adc_data.adc_normalized_data)
    non_time_outlier_signal = adc_data.non_time_outlier_adc_data
    non_amplitude_outlier_signal = adc_data.non_amplitude_outlier_adc_data

    for i in range(len(original_signal[target_adc])):
        if np.isnan(non_time_outlier_signal[i]) or np.isnan(non_amplitude_outlier_signal[i]):
            for j in range(ADC_COUNT):
                original_signal[j][i] = np.nan

    if adc_data.plot_enabled:
        plt.plot(adc_data.timestamps, adc_data.adc_normalized_data[target_adc], label='Original Signal', color='gray')
        plt.plot(adc_data.timestamps, original_signal[target_adc], label='Cleaned Signal', color='blue')
        plt.title("Clean adc_normalized_data")
        plt.legend()
        plt.show()


    clen_adc_normalized_data = ([], [], [], [], [], [])
    for i in range(len(original_signal[target_adc])):
        if not np.isnan(original_signal[target_adc][i]):
            clen_adc_normalized_data[0].append(adc_data.timestamps[i])
            for j in range(ADC_COUNT):
                clen_adc_normalized_data[j+1].append(original_signal[j][i])

    # resampling the signal BUT
    # we go like this:
    # for each NaN filled hole we calculate it's length and subtract it from timestamps of all nodes after it
    # only after we get this NaN free signal we resample it to RESAMPLE_NODE_COUNT nodes
    nan_adjusted_timestamps = []
    nan_adjusted_data = [], [], [], [], []
    total_time_shift = 0
    first_nan_timestamp = None
    for i in range(len(adc_data.timestamps)):
        if not np.isnan(original_signal[target_adc][i]):
            if first_nan_timestamp is not None:
                time_shift = adc_data.timestamps[i] - first_nan_timestamp
                total_time_shift += time_shift
                adjusted_timestamp = adc_data.timestamps[i] - total_time_shift
                nan_adjusted_timestamps.append(adjusted_timestamp)
                print(f"Adjusted timestamp: {adjusted_timestamp}, original: {adc_data.timestamps[i]}, total_time_shift: {total_time_shift}")
                for j in range(ADC_COUNT):
                    nan_adjusted_data[j].append(original_signal[j][i])
                first_nan_timestamp = None
            else:
                adjusted_timestamp = adc_data.timestamps[i] - total_time_shift
                nan_adjusted_timestamps.append(adjusted_timestamp)
                for j in range(ADC_COUNT):
                    nan_adjusted_data[j].append(original_signal[j][i])
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
    breaths = calculate_breaths(adc_data, target_adc)
    time_outliers(adc_data, target_adc, breaths)
    amplitude_outliers(adc_data, target_adc, breaths)
    remove_outliers_and_remake_signal(adc_data, target_adc)