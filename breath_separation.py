import numpy as np
from scipy.interpolate import UnivariateSpline, BSpline, splrep
import matplotlib.pyplot as plt

TARGET_ADC = 3
INDEX = 0

def breath_separation(adc_data, target_adc):
    local_maxima = find_local_maxima(adc_data=adc_data, target_adc=target_adc)
    smoothed_signal = curve_smoothing(adc_data=adc_data, peaks=local_maxima, target_adc=target_adc)
    
    #this returns len(signal)-1 intervals where i-th interval is the interval between points i and i+1
    interval_values = curves_to_intervals(signal=smoothed_signal, target_adc=target_adc)
    local_minima = intervals_to_minima(maxima=local_maxima, interval_values=interval_values, target_adc=target_adc)
    
    adc_data.signal_minima = local_minima
    adc_data.signal_maxima = local_maxima #step 6 - assigning those points to all signals (just save it in class)

    adc_data.smoothed_signal = final_smoothing_with_splines(adc_data, smoothed_signal, target_adc)
    """
    plt.plot(adc_data.adc_normalized_data[target_adc], label='Original')
    plt.plot(smoothed_signal, label='Smoothed')

    plt.scatter(local_maxima, smoothed_signal[local_maxima], color='red', label='Maxima')
    plt.scatter(local_minima, smoothed_signal[local_minima], color='blue', label='Minima')
    plt.legend()
    plt.show()


#step 1 - finding local maxima
def find_local_maxima(adc_data, target_adc=TARGET_ADC):
    maxima = []
    for i in range(len(adc_data.adc_normalized_data[target_adc])-2):
        if adc_data.adc_normalized_data[target_adc][i] < adc_data.adc_normalized_data[target_adc][i+1] > adc_data.adc_normalized_data[target_adc][i+2]:
            maxima.append(i+1)
    return maxima

#step 2 - curve smoothing between maxima
def curve_smoothing(adc_data, peaks, target_adc=TARGET_ADC):
    smoothed_signal = adc_data.adc_normalized_data[target_adc].copy()
    for i in range(len(peaks)-1):
        start_idx = peaks[i]
        end_idx = peaks[i+1]
        start_val = adc_data.adc_normalized_data[target_adc][start_idx]
        end_val = adc_data.adc_normalized_data[target_adc][end_idx]

        x = np.arange(start_idx, end_idx+1)
        y = adc_data.adc_normalized_data[target_adc][start_idx:end_idx+1]
        segment_length = len(x)
        if segment_length > 3:
            k_seg = 3 
        elif segment_length == 3:
            k_seg = 2 
        elif segment_length == 2:
            k_seg = 1 
        else:
            smoothed_signal[start_idx:end_idx+1] = y
            continue

        spline = UnivariateSpline(x, y, k=k_seg) #s is a smoothing factor to tweak
        spline.set_smoothing_factor(2)

        smoothed_segment = spline(x)
        smoothed_segment[0] = start_val
        smoothed_segment[-1] = end_val

        smoothed_signal[start_idx:end_idx+1] = smoothed_segment
    return smoothed_signal

#step 3 - dividing the smoothed curves in subintervals and calculating average value for all of them
def curves_to_intervals(signal, target_adc=TARGET_ADC):
    interval_values = np.zeros(shape=(len(signal)-1))
    for i in range(len(signal)-1): 
        #the derivative of a linear function between two points is a straight horizontal line equal to the "a" of linear func
        interval_values[i] = (signal[i+1]-signal[i])
    return interval_values

#step 4 and 5 - finding local minima in a union of two specific intervals (3*a1_avg = a2_avg)
def intervals_to_minima(maxima, interval_values, target_adc=TARGET_ADC):
    minima = []
    if maxima[0] != 0:
        minima.append(0)
    for i in range(len(maxima)-1):
        turn_point = maxima[i]+1
        rise_point = None #the three times rise point
        candidate_point = None
        point_found = False
        while turn_point < maxima[i+1] and not point_found: #when the derivative starts to grow
            if interval_values[turn_point-1] < interval_values[turn_point] and interval_values[turn_point] > 0:
                candidate_point = turn_point
                if 3*interval_values[turn_point-1] < interval_values[turn_point]:
                    rise_point = turn_point
            turn_point += 1
            if rise_point:
                minima.append(rise_point)
                point_found = True
            #return turn point else
        if not point_found:
            if candidate_point:
                minima.append(candidate_point)
            else:
                minima.append(turn_point-1)
    return minima

# step 7 - final smoothing with fifth degree splines
def final_smoothing_with_splines(adc_data, smoothed_signal, target_adc=TARGET_ADC):
    final_signal = smoothed_signal.copy()
    x = np.arange(0, len(smoothed_signal))
    y = smoothed_signal.copy()
    degree = 5 # spline degree
    tck = splrep(x, y)
    spline = BSpline(tck[0], tck[1], k=degree, extrapolate=False) #s is a smoothing factor to tweak
    smoothed_segment = spline(x)
    smoothed_segment[0] = smoothed_signal[0]
    smoothed_segment[-1] = smoothed_signal[-1]

    final_signal[0:len(smoothed_signal)] = smoothed_segment
    
    plt.plot(adc_data.adc_normalized_data[target_adc], label='Original')
    plt.plot(smoothed_signal, label='Smoothed')
    plt.plot(final_signal, label='Even more smoooooth')
    plt.legend()
    plt.show()
    
    return final_signal