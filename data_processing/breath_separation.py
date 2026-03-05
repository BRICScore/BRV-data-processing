from config import *

def breath_separation(adc_data, target_adc):
    """
        Logically separate breaths in the normalized ADC signal by finding local maxima and minima and storing the data
        in the ADCdata object. A breath is defined as the signal between two consecutive minima. 

        Parameters
        ----------
        adc_data : ADCdata
            The ADCdata object containing the normalized ADC data and timestamps.
        target_adc : int
            The index of the ADC to analyze for outliers.
        
        Returns
        -------
        none
        
        Side Effects
        ------------
        This function may visualize the split breahts by coloring each breath differently if the plot_enabled flag is set to true
        and sets the signal_maxima and signal_minima attributes of the adc_data object.
    """

    adc_data.signal_maxima = find_local_maxima(adc_data=adc_data, target_adc=target_adc)
    adc_data.signal_minima = find_local_minima(adc_data=adc_data, target_adc=target_adc)

    # Plots each breath with different color
    if adc_data.plot_enabled:
        breaths = []
        for i in range(len(adc_data.signal_minima) - 1):
            breath = adc_data.adc_normalized_data[target_adc][adc_data.signal_minima[i]:adc_data.signal_minima[i + 1]]
            breaths.append(breath)

        plt.figure(figsize=(12, 6))
        for i, breath in enumerate(breaths):
            plt.plot(adc_data.timestamps[adc_data.signal_minima[i]:adc_data.signal_minima[i + 1]], breath)

        plt.title("Separated Breaths")
        plt.xlabel("Time (samples)")
        plt.ylabel("Normalized ADC Value")
        plt.show()

def find_local_maxima(adc_data, target_adc=TARGET_ADC):
    """
        Identify local maxima in the normalized ADC signal using scipy's find_peaks function. We use a statistical 
        threshold to filter out noise.

        Parameters
        ----------
        adc_data : ADCdata
            The ADCdata object containing the normalized ADC data and timestamps.
        target_adc : int
            The index of the ADC to analyze for outliers.
        
        Returns
        -------
        maxima : numpy.ndarray
            The indices of the local maxima in the normalized ADC signal.
        
        Side Effects
        ------------
        This function has no side effects.
    """

    signal = adc_data.adc_normalized_data[target_adc]
    std_dev_signal = np.std(signal)
    mean_signal = np.mean(signal)

    maxima, _ = scipy.signal.find_peaks(signal, distance=MIN_DISTANCE, height=mean_signal + std_dev_signal*STD_DEV_CONST)
    return maxima

def find_local_minima(adc_data, target_adc=TARGET_ADC):
    """
        Identify local minima in the normalized ADC signal using scipy's find_peaks function on an inverted 
        normalized_adc_data signal. We use a statistical threshold to filter out noise.

        Parameters
        ----------
        adc_data : ADCdata
            The ADCdata object containing the normalized ADC data and timestamps.
        target_adc : int
            The index of the ADC to analyze for outliers.
        
        Returns
        -------
        minima : numpy.ndarray
            The indices of the local minima in the normalized ADC signal.
        
        Side Effects
        ------------
        This function has no side effects.
    """

    signal = [-s for s in adc_data.adc_normalized_data[target_adc]]
    std_dev_signal = np.std(signal)
    mean_signal = np.mean(signal)

    minima, _ = scipy.signal.find_peaks(signal, distance=MIN_DISTANCE, height=mean_signal + std_dev_signal*STD_DEV_CONST)
    return minima
