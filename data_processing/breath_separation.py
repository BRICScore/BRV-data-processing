from config import *

def breath_separation(adc_data, target_adc):
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
    signal = adc_data.adc_normalized_data[target_adc]
    maxima, _ = scipy.signal.find_peaks(signal, distance=MIN_DISTANCE)
    std_dev_signal = np.std(signal)
    mean_signal = np.mean(signal)

    for peak in maxima:
        if signal[peak] < (mean_signal + std_dev_signal*STD_DEV_CONST):
            maxima = np.delete(maxima, np.where(maxima == peak))

    return maxima

def find_local_minima(adc_data, target_adc=TARGET_ADC):
    signal = [-s for s in adc_data.adc_normalized_data[target_adc]]
    minima, _ = scipy.signal.find_peaks(signal, distance=MIN_DISTANCE)
    std_dev_signal = np.std(signal)
    mean_signal = np.mean(signal)
    
    for minimum in minima:
        if signal[minimum] <= mean_signal + std_dev_signal*STD_DEV_CONST:
            minima = np.delete(minima, np.where(minima == minimum))

    return minima
