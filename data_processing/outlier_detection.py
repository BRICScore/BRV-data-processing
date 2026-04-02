from config import *

from signal import signal

from data_containers import BRVDataClean

def calculate_breaths(BRV_data_intermediate, target_adc):
    """
        Generate an array of dictionaries representing the separated breaths that contain the
        breath data, timestamps, and metadata for each breath.

        Parameters
        ----------
        timestamps : np.ndarray
            An array of timestamps.
        adc_normalized_data : list of np.ndarray
            A list of arrays containing the normalized ADC data for each channel.
        signal_maxima : np.ndarray
            An array of indices representing the local maxima in the signal.
        signal_minima : np.ndarray
            An array of indices representing the local minima in the signal.
        target_adc : int
            The index of the ADC to analyze for outliers.

        Returns
        -------
        breaths : list of dictionaries
            A list of dictionaries representing the separated breaths that contain the
            breath data, timestamps, and metadata for each breath.
        
        Side Effects
        ------------
        This function has no side effects
    """
    signal_maxima = BRV_data_intermediate.signal_maxima
    signal_minima = BRV_data_intermediate.signal_minima
    adc_normalized_data = BRV_data_intermediate.adc_normalized_data
    timestamps = BRV_data_intermediate.timestamps
    breaths = []

    for i in range(len(signal_minima) - 1):
        breath = adc_normalized_data[target_adc][signal_minima[i]:signal_minima[i + 1]]
        max_index = signal_maxima[np.where((signal_maxima > signal_minima[i]) & (signal_maxima < signal_minima[i + 1]))]
        # print(f"max_index: {max_index}, max_value: {breath[max_index - signal_minima[i]]}")
        max_timestamp = timestamps[max_index]

        breaths.append({
            "breath": breath,
            "start_index": signal_minima[i],
            "end_index": signal_minima[i + 1],
            "start_timestamp": timestamps[signal_minima[i]],
            "end_timestamp": timestamps[signal_minima[i + 1]],
            "timestamps": timestamps[signal_minima[i]:signal_minima[i + 1]],
            "duration": timestamps[signal_minima[i + 1]] - timestamps[signal_minima[i]],
            "max_index": max_index,
            "max_value": breath[max_index - signal_minima[i]],
            "max_timestamp": max_timestamp,
            "amplitude": np.max(breath) - np.min(breath)
        })
    return breaths

def resample_data(y, new_len):
    """
        Resample the given data to new_len using cubic spline interpolation.

        Parameters
        ----------
        y : numpy.ndarray
            The input data to be resampled.
        new_len : int
            The desired length of the resampled data.

        Returns
        -------
        numpy.ndarray
            The resampled data with length new_len.
        
        Side Effects
        ------------
        This function has no side effects
    """
        
    x_old = np.linspace(0, 1, len(y))
    x_new =  np.linspace(0, 1, new_len)
    # f = spi.CubicSpline(x_old, old_y)
    f = spi.interp1d(x_old, y, kind='cubic')
    return f(x_new)

def resample_adc_data_and_timestamps(data, timestamps, target_adc, plot_enabled=False):
    """
        Resample the adc_data and timestamps that has had outlier breaths removed to resampled_node_count (a node every 100ms)

        Parameters
        ----------
        data : ADCdata
            The ADCdata object containing the ADC data with the outlier breaths removed.
        timestamps : numpy.ndarray
            The timestamps corresponding to the data object.
        plot_enabled : bool
            The plotting flag that determines whether to plot the resampled data.
        target_adc : int
            The index of the ADC to analyze, all adcs will be resampled but only the target_adc will be plotted if plot_enabled is True.

        Returns
        -------
        resampled_data : list of numpy.ndarray
            A list of numpy arrays containing the resampled ADC data for each ADC.
        resampled_timestamps : numpy.ndarray
            A numpy array containing the resampled timestamps.           
        
        Side Effects
        ------------
        This function may plot the resampled data if the plot_enabled flag is set to true.
    """
    # timestamps[0] for sure is 0.0... for sure...
    signal_duration = timestamps[-1] - timestamps[0]
    resampled_node_count = int(signal_duration // 100)
    print(f"Signal duration: {signal_duration} ms, resampled_node_count: {resampled_node_count}")
    resampled_data = [[] for _ in range(ADC_COUNT)]
    resampled_timestamps = resample_data(timestamps, resampled_node_count)
    for i in range(ADC_COUNT):
        resampled_data[i] = resample_data(data[i], resampled_node_count)

    if plot_enabled:
        plt.plot(timestamps, data[target_adc], label='Cleaned Signal', color='blue')
        for j in range(ADC_COUNT):
            plt.plot(resampled_timestamps, resampled_data[j], label=f'Resampled Signal ADC {j}')
            # plt.scatter(resampled_timestamps, resampled_data[j], label=f'Resampled Signal ADC {j}', s=10)

        plt.title("Resampled Signal")
        plt.legend()
        plt.show()
    
    return resampled_data, resampled_timestamps

def time_outliers(adc_normalized_data, target_adc, breaths):
    """
        Generate a adc_signal that has had a PERCENTILE_THRESHOLD% of the shortest and longest 
        breaths removed as outliers

        Parameters
        ----------
        adc_data : ADCdata
            The ADCdata object containing the normalized ADC data and timestamps.
        target_adc : int
            The index of the ADC to analyze for outliers.
        breaths : list 
            A list of dictionaries representing the separated breaths returned by the calculate_breaths function.

        Returns
        -------
        non_outlier_signal : numpy.ndarray
            The normalized ADC signal with outlier breaths set to NaN.           
        
        Side Effects
        ------------
        This function may plot the removed data if the plot_enabled flag is set to true.
    """
    breath_durations = [(breath["start_timestamp"], breath["duration"]) for breath in breaths]
    lower_bound = np.percentile([d[1] for d in breath_durations], PERCENTILE_THRESHOLD)
    upper_bound = np.percentile([d[1] for d in breath_durations], 100 - PERCENTILE_THRESHOLD)

    outlier_breaths = []
    for breath in breaths:
        if breath["duration"] < lower_bound or breath["duration"] > upper_bound:  
            outlier_breaths.append(breath)

    original_signal = adc_normalized_data[target_adc].copy()
    non_outlier_signal= adc_normalized_data[target_adc].copy()
    time_outlier_signal = np.full_like(original_signal, np.nan)

    for outlier_breath in outlier_breaths:
        start_index = outlier_breath["start_index"]
        end_index = outlier_breath["end_index"]

        time_outlier_signal[start_index:end_index] = original_signal[start_index:end_index]
        non_outlier_signal[start_index:end_index] = np.nan

    # TODO: fix plot flags 
    # if adc_data.plot_enabled:
    #     plt.title("Time outliers")
    #     plt.plot(adc_data.timestamps, original_signal, label='Original Signal', color='gray')
    #     plt.plot(adc_data.timestamps, non_outlier_signal, label='Non-outlier Signal', color="green")
    #     plt.plot(adc_data.timestamps, time_outlier_signal, label='Outlier Signal', color='red')
    #     plt.legend()
    #     plt.show()

    return non_outlier_signal

def amplitude_outliers(adc_normalized_data, target_adc, breaths):
    """
        Generate a adc_signal that has had a PERCENTILE_THRESHOLD% of the lowest and tallest 
        breaths removed as outliers

        Parameters
        ----------
        adc_data : ADCdata
            The ADCdata object containing the normalized ADC data and timestamps.
        target_adc : int
            The index of the ADC to analyze for outliers.
        breaths : list 
            A list of dictionaries representing the separated breaths returned by the calculate_breaths function.

        Returns
        -------
        non_outlier_signal : numpy.ndarray
            The normalized ADC signal with outlier breaths set to NaN.           
        
        Side Effects
        ------------
        This function may plot the removed data if the plot_enabled flag is set to true.
    """
    amplitudes = [breath["amplitude"] for breath in breaths]
    lower_bound = np.percentile(amplitudes, PERCENTILE_THRESHOLD)
    upper_bound = np.percentile(amplitudes, 100 - PERCENTILE_THRESHOLD)

    outlier_breaths = []
    for outlier_breath in breaths:
        if outlier_breath["amplitude"] < lower_bound or outlier_breath["amplitude"] > upper_bound:
            outlier_breaths.append(outlier_breath)

    original_signal = adc_normalized_data[target_adc].copy()
    non_outlier_signal = original_signal.copy()
    amplitude_outlier_signal = np.full_like(original_signal, np.nan)

    for outlier_breath in outlier_breaths:
        start_index = outlier_breath["start_index"]
        end_index = outlier_breath["end_index"]

        amplitude_outlier_signal[start_index:end_index] = original_signal[start_index:end_index]
        non_outlier_signal[start_index:end_index] = np.nan

    # TODO: fix plot flags
    # if adc_data.plot_enabled:
    #     plt.title("Amplitude outliers")
    #     plt.plot(adc_data.timestamps, original_signal, label='Original Signal', color='gray')
    #     plt.plot(adc_data.timestamps, non_outlier_signal, label='Non-outlier Signal', color='green')
    #     plt.plot(adc_data.timestamps, amplitude_outlier_signal, label='Outlier Signal', color='red')
    #     plt.legend()
    #     plt.show()

    return non_outlier_signal

def remove_outliers_and_remake_signal(target_adc, non_time_outlier_signal, non_amplitude_outlier_signal, BRV_data_intermediate, BRV_data_clean):
    """
        Generate a adc_signal that has both time and amplitude outlier breaths removed.

        Parameters
        ----------
        adc_data : ADCdata
            The ADCdata object containing the normalized ADC data and timestamps.
        target_adc : int
            The index of the ADC to analyze for outliers.
        non_time_outlier_signal : numpy.ndarray
            The normalized ADC signal with time outlier breaths set to NaN.
        non_amplitude_outlier_signal : numpy.ndarray
            The normalized ADC signal with amplitude outlier breaths set to NaN.

        Returns
        -------
        none
        
        Side Effects
        ------------
        This function may plot the removed data if the plot_enabled flag is set to true and 
        sets the final_adc_data and final_adc_timestamps attributes of the adc_data object.
    """
    timestamps = BRV_data_intermediate.timestamps
    adc_normalized_data = BRV_data_intermediate.adc_normalized_data
    
    original_signal = copy.deepcopy(adc_normalized_data)

    for i in range(len(original_signal[target_adc])):
        if np.isnan(non_time_outlier_signal[i]) or np.isnan(non_amplitude_outlier_signal[i]):
            for j in range(ADC_COUNT):
                original_signal[j][i] = np.nan

    # TODO: fix plot flags
    # if adc_data.plot_enabled:
    #     plt.plot(adc_data.timestamps, adc_data.adc_normalized_data[target_adc], label='Original Signal', color='gray')
    #     plt.plot(adc_data.timestamps, original_signal[target_adc], label='Cleaned Signal', color='blue')
    #     plt.title("Clean adc_normalized_data")
    #     plt.legend()
    #     plt.show()

    clean_adc_normalized_timestamps = []
    clean_adc_normalized_data = [[] for _ in range(ADC_COUNT)]
    for i in range(len(original_signal[target_adc])):
        if not np.isnan(original_signal[target_adc][i]):
            clean_adc_normalized_timestamps.append(timestamps[i])
            for j in range(ADC_COUNT):
                clean_adc_normalized_data[j].append(original_signal[j][i])

    # for each NaN filled hole we calculate it's length and add it to the total time shift
    # then when we want to write any non-NaN data to the nan_adjusted_data and nan_adjusted_timestamps
    # we subtract the total time shift from the original timestamp to get the new timestamp
    # then after we get this NaN free signal we resample it to RESAMPLE_NODE_COUNT nodes
    nan_adjusted_timestamps = []
    nan_adjusted_data = [[] for _ in range(ADC_COUNT)]
    total_time_shift = 0
    first_nan_timestamp = None
    for i in range(len(timestamps)):
        if not np.isnan(original_signal[target_adc][i]):
            if first_nan_timestamp is not None:
                time_shift = timestamps[i] - first_nan_timestamp
                total_time_shift += time_shift
                adjusted_timestamp = timestamps[i] - total_time_shift
                nan_adjusted_timestamps.append(adjusted_timestamp)
                # print(f"Adjusted timestamp: {adjusted_timestamp}, original: {adc_data.timestamps[i]}, total_time_shift: {total_time_shift}")
                for j in range(ADC_COUNT):
                    nan_adjusted_data[j].append(original_signal[j][i])
                first_nan_timestamp = None
            else:
                adjusted_timestamp = timestamps[i] - total_time_shift
                nan_adjusted_timestamps.append(adjusted_timestamp)
                for j in range(ADC_COUNT):
                    nan_adjusted_data[j].append(original_signal[j][i])
        else:
            if first_nan_timestamp is None:
                first_nan_timestamp = timestamps[i]

    # TODO: fix plot flags
    # if adc_data.plot_enabled:        
    #     plt.plot(nan_adjusted_timestamps, nan_adjusted_data[target_adc], label='Cleaned Signal', color='blue')
    #     plt.title("NaN adjusted timestamps")
    #     plt.legend()
    #     plt.show()
    
    BRV_data_clean.adc_data, BRV_data_clean.timestamps = resample_adc_data_and_timestamps(
        nan_adjusted_data,
        nan_adjusted_timestamps,
        target_adc, True
    )
    

def outlier_detection(BRV_data_intermediate, target_adc):
    """
        This function serves as the main function for outlier detection. It organizes and calls other functions 
        in order to clean and resample the ADC data.

        Parameters
        ----------
        adc_data : ADCdata
            The ADCdata object containing the normalized ADC data and timestamps.
        target_adc : int
            The index of the ADC to analyze for outliers.

        Returns
        -------
        non_outlier_signal : numpy.ndarray
            The normalized ADC signal with outlier breaths set to NaN.           
        
        Side Effects
        ------------
        This function has no side effects.
    """
    BRV_data_clean = BRVDataClean()
    breaths = calculate_breaths(BRV_data_intermediate, target_adc)
    non_time_outlier_signal = time_outliers(BRV_data_intermediate.adc_normalized_data, target_adc, breaths)
    non_amplitude_outlier_signal = amplitude_outliers(BRV_data_intermediate.adc_normalized_data, target_adc, breaths)
    remove_outliers_and_remake_signal(
        target_adc, 
        non_time_outlier_signal, 
        non_amplitude_outlier_signal, 
        BRV_data_intermediate,
        BRV_data_clean)
    return BRV_data_clean