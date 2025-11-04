import numpy as np
import matplotlib.pyplot as plt

PERCENTILE_THRESHOLD = 10  # % for both lower and upper bounds

def time_outliers(adc_data, target_adc):
    
    # Extract breath durations from minima timestamps
    breath_durations = []
    minima_timestamps = []
    for minimum in adc_data.signal_minima:
        minima_timestamps.append(adc_data.timestamps[minimum])

    for i in range(1, len(minima_timestamps)):
        duration = minima_timestamps[i] - minima_timestamps[i-1]
        
        # print(f"Breath {i}: Start Time = {minima_timestamps[i-1]} ms, Duration = {duration} ms")
        breath_durations.append((minima_timestamps[i-1], duration))

    # Enumerate outlier breaths based on duration percentiles
    lower_bound = np.percentile(breath_durations, PERCENTILE_THRESHOLD)
    upper_bound = np.percentile(breath_durations, 100 - PERCENTILE_THRESHOLD)

    # timestamps and durations of breaths to be deleted
    outlier_breaths = []
    for breath in breath_durations:
        if breath[1] < lower_bound or breath[1] > upper_bound:  
            outlier_breaths.append(breath)

    # print(outlier_breaths)

    # Discard outlier breaths from adc_data (put them in non_outlier_adc_data)
    adc_data.non_outlier_adc_data = adc_data.adc_normalized_data[target_adc].copy()
    for breath in outlier_breaths:
        outlier_start_time = breath[0]
        outlier_end_time = breath[0] + breath[1]

        start_index = np.searchsorted(adc_data.timestamps, outlier_start_time)
        end_index = np.searchsorted(adc_data.timestamps, outlier_end_time)

        # Remove outlier segment by setting it to NaN
        adc_data.non_outlier_adc_data[start_index:end_index] = np.nan

    plt.plot(adc_data.timestamps, adc_data.adc_normalized_data[target_adc], label='Original')
    plt.plot(adc_data.timestamps, adc_data.non_outlier_adc_data, color='orange', label='Non-outlier Data')
    plt.legend()
    plt.show()

    
def magnitude_outliers(adc_data, target_adc):
    print("xd")
    



def outlier_detection(adc_data, target_adc):
    time_outliers(adc_data, target_adc)