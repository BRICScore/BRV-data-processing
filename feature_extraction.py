import matplotlib.pyplot as plt
import numpy as np

from main import ADC_COUNT, TARGET_ADC

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
    last_peak_was_x_ago = 0
    min_spread_of_peaks = 10    # TODO: adjust based on empirical data
    min_value_for_peak = 0.0002 # TODO: adjust based on empirical data
    for i in range(1,len(adc_data.adc_normalized_data[target_adc-1])-1):
        if adc_data.adc_normalized_data[target_adc-1][i-1] < adc_data.adc_normalized_data[target_adc-1][i] and adc_data.adc_normalized_data[target_adc-1][i] > adc_data.adc_normalized_data[target_adc-1][i+1]:
            if last_peak_was_x_ago > min_spread_of_peaks and adc_data.adc_normalized_data[target_adc-1][i] > min_value_for_peak:
                last_peak_was_x_ago = 0
                breath_peaks.append(adc_data.adc_normalized_data[target_adc-1][i])
            else:
                last_peak_was_x_ago += 1

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

def basic_feature_extraction(adc_data, input_file):
    count_breaths(adc_data)
    avg_breath_depth, avg_breath_depth_std_dev = calculate_average_breath_depth(adc_data)
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