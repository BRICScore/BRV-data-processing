import matplotlib.pyplot as plt
import numpy as np
import math
from config import *

def plot_data(input_file, adc_data, avg_breath_depth):
    plt.figure(figsize=(15, 10))
    plt.plot(adc_data.timestamps, adc_data.adc_normalized_data[0], label='ADC1')
    plt.plot(adc_data.timestamps, adc_data.adc_normalized_data[1], label='ADC2')
    plt.plot(adc_data.timestamps, adc_data.adc_normalized_data[2], label='ADC3')
    plt.plot(adc_data.timestamps, adc_data.adc_normalized_data[3], label='ADC4')
    plt.plot(adc_data.timestamps, adc_data.adc_normalized_data[4], label='ADC5')
    
    # horizontal line for average breath depth
    plt.hlines(y=[avg_breath_depth], xmin=adc_data.timestamps[0], xmax=adc_data.timestamps[-1], label=f"avg breath depth adc{TARGET_ADC}")
    
    plt.xlabel('Timestamp (ms)')
    plt.ylabel('ADC voltage deviation (V)')
    plt.title('ADC voltage changes over time (deviation from mean)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./results/{input_file}_adc_plot.png")
    plt.show()
    
def count_breaths(adc_data):
    # counting breaths (zero-crossings of the signal) --------------------------------------------------
    breath_counters = []
    for i in range(ADC_COUNT):
        breath_counters.append(len(np.where(np.diff(np.sign(adc_data.adc_normalized_data[i])))[0])/2)
    # due to the variable characteristics of the signals from each ADC depending on the person,
    # we take the signal with presumably the smallest number of disturbances as the counter
    adc_data.breath_count = np.max(breath_counters)
    #----------------------------------------------------------------------------------------------

def calculate_average_breath_depth(adc_data, target_adc=TARGET_ADC):
    breath_peaks = []
    breath_peak_indices = []
    last_peak_was_x_ago = 0
    min_spread_of_peaks = 20    # 10 Hz means the highest acceptable frequency of breaths is 1 per second (value/frequency)
    min_value_for_peak = 0.00015 # TODO: adjust based on empirical data
    for i in range(1,len(adc_data.adc_normalized_data[target_adc-1])-1):
        if adc_data.adc_normalized_data[target_adc-1][i-1] < adc_data.adc_normalized_data[target_adc-1][i] and adc_data.adc_normalized_data[target_adc-1][i] > adc_data.adc_normalized_data[target_adc-1][i+1]:
            if last_peak_was_x_ago > min_spread_of_peaks and adc_data.adc_normalized_data[target_adc-1][i] > min_value_for_peak:
                last_peak_was_x_ago = 0
                breath_peaks.append(adc_data.adc_normalized_data[target_adc-1][i])
                breath_peak_indices.append(i)
        last_peak_was_x_ago += 1
    adc_data.breath_peaks = breath_peaks
    adc_data.breath_peak_indices = breath_peak_indices
    avg_breath_depth = np.mean(breath_peaks)
    avg_breath_depth_std_dev = np.std(adc_data.adc_normalized_data[target_adc-1])
    return avg_breath_depth, avg_breath_depth_std_dev

def calculate_breathing_tract(adc_data):
    belt_share = np.zeros(shape=(ADC_COUNT))
    belt_share_std = np.zeros(shape=(ADC_COUNT))
    avg_sum = 0
    avg_sum_std = 0
    for i in range(1,ADC_COUNT+1):
        avg, avg_std = calculate_average_breath_depth(adc_data, target_adc=i)
        avg_sum += avg
        avg_sum_std += avg_std
        belt_share[i-1] = avg
        belt_share_std[i-1] = avg_std
    belt_share /= avg_sum
    belt_share_std /= avg_sum_std
    return belt_share, belt_share_std

# calculate by detecting where the data increases significantly
def detect_expiratory_pause(adc_data):
    # value to detect gradual slope as termination point
    # apart from detecting it by flipping the sign of derivative
    sensitivity = 0.5

    adc_data.breath_minimum_indices = []
    adc_data.breath_minima = []
    for i in range(len(adc_data.exhale_points)):
        minimum = adc_data.exhale_point_indices[i] + 2 # offset to counteract faulty exhale points
        pointFound = False
        while not pointFound:
            try:
                val_current = adc_data.adc_normalized_data[TARGET_ADC-1][minimum]
                val_next = adc_data.adc_normalized_data[TARGET_ADC-1][minimum+1]
                val_after_next = adc_data.adc_normalized_data[TARGET_ADC-1][minimum+2]
            except:
                val_current = 0.0
                val_next = 0.0
                val_after_next = 0.0
            if val_current > val_next:
                if val_next > val_after_next:
                    minimum += 1
                else:
                    pointFound = True
                    minimum += 1
            if minimum == 0 or minimum >= len(adc_data.timestamps)-1:
                break
        minimum = min(minimum, len(adc_data.adc_normalized_data[TARGET_ADC-1])-1)
        adc_data.breath_minimum_indices.append(minimum)
        adc_data.breath_minima.append(adc_data.adc_normalized_data[TARGET_ADC-1][minimum])

# wait for data to stop decreasing
def detect_exhale(adc_data):
    adc_data.exhale_point_indices = []
    adc_data.exhale_points = []
    for i in range(len(adc_data.breath_peaks)):
        exhale_point = adc_data.breath_peak_indices[i]
        pointFound = False
        while not pointFound:
            try:
                val_current = adc_data.adc_normalized_data[TARGET_ADC-1][exhale_point]
                val_next = adc_data.adc_normalized_data[TARGET_ADC-1][exhale_point+1]
                val_after_next = adc_data.adc_normalized_data[TARGET_ADC-1][exhale_point+2]
                val_after_after_next = adc_data.adc_normalized_data[TARGET_ADC-1][exhale_point+3]
            except:
                val_current = 0.0
                val_next = 0.0
                val_after_next = 0.0
                val_after_after_next = 0.0

            if val_current > val_next:
                if val_next > val_after_next:
                    if val_after_next > val_after_after_next:
                        pointFound = True
                    else:
                        exhale_point += 1
                else:
                    exhale_point += 1
            else:
                exhale_point += 1
            if exhale_point == 0 or exhale_point >= len(adc_data.timestamps)-1:
                break
        adc_data.exhale_point_indices.append(exhale_point)
        adc_data.exhale_points.append(adc_data.adc_normalized_data[TARGET_ADC-1][exhale_point])

# calculate by detecting where the data drops significantly
def detect_inspiratory_pause(adc_data):
    # the breath_peaks are calculated in the first call to calculate_average_breath_depth
    pass

# calculate start by going from the maxima backwards
def detect_inhale(adc_data):
    # value to detect gradual slope as termination point
    # apart from detecting it by flipping the sign of derivative
    sensitivity = 0.5

    adc_data.inhale_point_indices = []
    adc_data.inhale_points = []
    for i in range(len(adc_data.breath_peaks)):
        inhale_point = adc_data.breath_peak_indices[i]
        pointFound = False
        while not pointFound:
            val_current = adc_data.adc_normalized_data[TARGET_ADC-1][inhale_point]
            val_prev = adc_data.adc_normalized_data[TARGET_ADC-1][inhale_point-1]
            val_before_prev = adc_data.adc_normalized_data[TARGET_ADC-1][inhale_point-2]
            if val_current > val_prev:
                if val_prev > val_before_prev:
                    inhale_point -= 1
                else:
                    pointFound = True
                    inhale_point -= 1
            if inhale_point == 0 or inhale_point == len(adc_data.timestamps)-1:
                break
        adc_data.inhale_point_indices.append(inhale_point)
        adc_data.inhale_points.append(adc_data.adc_normalized_data[TARGET_ADC-1][inhale_point])

# calculations assuming local maxima is the end of inhale and start of inspiratory pause
# this function returns avg duration of each of the 4 breathing phases
def calculate_breathing_phases(adc_data):
    detect_inhale(adc_data)
    detect_inspiratory_pause(adc_data)
    detect_exhale(adc_data)
    detect_expiratory_pause(adc_data)
    phases_values = [0.0, 0.0, 0.0, 0.0]
    NPtimestamps = np.array(adc_data.timestamps)
    number_of_breaths = len(adc_data.breath_peaks)
    for i in range(number_of_breaths-1):
        phases_values[0] += NPtimestamps[adc_data.breath_peak_indices[i]] - NPtimestamps[adc_data.inhale_point_indices[i]]
        phases_values[1] += NPtimestamps[adc_data.exhale_point_indices[i]] - NPtimestamps[adc_data.breath_peak_indices[i]]
        phases_values[2] += NPtimestamps[adc_data.breath_minimum_indices[i]] - NPtimestamps[adc_data.exhale_point_indices[i]]
        if i != number_of_breaths-1:
            phases_values[3] += NPtimestamps[adc_data.inhale_point_indices[i+1]] - NPtimestamps[adc_data.breath_minimum_indices[i]]
    phases_values[0] /= number_of_breaths
    phases_values[1] /= number_of_breaths
    phases_values[2] /= number_of_breaths
    phases_values[3] /= number_of_breaths-1
    return phases_values


def display_calculated_breath_phases(adc_data):
    plt.plot(adc_data.timestamps, adc_data.adc_normalized_data[TARGET_ADC-1])
    NPtimestamps = np.array(adc_data.timestamps)
    plt.scatter(NPtimestamps[adc_data.inhale_point_indices], adc_data.inhale_points, c="blue") # start of inhale
    plt.scatter(NPtimestamps[adc_data.breath_peak_indices], adc_data.breath_peaks, c="red") # start of IP
    plt.scatter(NPtimestamps[adc_data.exhale_point_indices], adc_data.exhale_points, c="green") # start of exhale
    plt.scatter(NPtimestamps[adc_data.breath_minimum_indices], adc_data.breath_minima, c="magenta") # start of EP
    plt.legend(["signal","inhale start", "IP start", "exhale start", "EP start"])
    plt.xlabel("timestamp [ms]")
    plt.ylabel("signal deviation from average value")
    plt.show()

def basic_feature_extraction(adc_data, input_file):
    count_breaths(adc_data)
    avg_breath_depth, avg_breath_depth_std_dev = calculate_average_breath_depth(adc_data)
    phases_avg_values = calculate_breathing_phases(adc_data)
    display_calculated_breath_phases(adc_data) # do not move it takes values from two function calls above
    belt_share, belt_share_std = calculate_breathing_tract(adc_data)
    #-----------------------------------------------------------------------------------
    # nazewnictwo: feature_(nr_segmentu)_person-conditions(sit,lay,run)_(nr_próbki)
    # {"cecha1": 1.3, "cecha2": 0.45, …, "cecha12": [0.1, 0.2, 0.3, 0.4, 0.5]}
    if adc_data.plot_enabled:
        plot_data(input_file, adc_data, avg_breath_depth)
    with open(f"./features/features_{input_file}.jsonl", 'w') as o_f:
        o_f.write(f"{"{"}\"bpm\": {adc_data.breath_count/((adc_data.timestamps[-1] - adc_data.timestamps[0])/60_000)}, ")
        o_f.write(f"\"breath_depth\": {avg_breath_depth}, ")
        o_f.write(f"\"breath_depth_std\": {avg_breath_depth_std_dev*2}, ")
        o_f.write(f"\"belt_share\": [")
        for i in range(len(belt_share)):
            o_f.write(f"{belt_share[i]}")
            if i != len(belt_share)-1:
                o_f.write(", ")
        o_f.write("], ")
        o_f.write(f"\"belt_share_std\": [")
        for i in range(len(belt_share_std)):
            o_f.write(f"{belt_share_std[i]}")
            if i != len(belt_share_std)-1:
                o_f.write(", ")
        o_f.write("], ")
        o_f.write(f"\"breathing_phase_lengths\": [")
        for i in range(len(phases_avg_values)):
            o_f.write(f"{phases_avg_values[i]}")
            if i != len(phases_avg_values)-1:
                o_f.write(", ")
        o_f.write("]}\n")
    print(f"breath count for {input_file}: {adc_data.breath_count} for {adc_data.timestamps[-1] - adc_data.timestamps[0]}ms -> {adc_data.breath_count/((adc_data.timestamps[-1] - adc_data.timestamps[0])/60_000)} bpm")
    print(f"breath depth: {avg_breath_depth}")
    print(f"breath depth std: {avg_breath_depth_std_dev*2}")

    if adc_data.plot_enabled:
        plt.figure(figsize=(8,6))
        plt.title(f"{input_file} breath track")
        # TODO: substitute numbers with areas of the chest names
        plt.plot([1,2,3,4,5], belt_share, "-o", label="belt share in breathing")
        plt.plot([1,2,3,4,5], belt_share_std, "-o", label="belt share std")
        plt.xlabel("belt number")
        xint = range(1,ADC_COUNT+1)
        plt.xticks(xint)
        plt.ylabel("Relative share in deviation from average breath depth")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"./features/{input_file}_breath_track.png")
        plt.show()