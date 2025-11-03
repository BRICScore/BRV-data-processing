import sys
import json
import matplotlib.pyplot as plt
import numpy as np

from helper_functions import u2_to_i, adc_to_voltage
from signal_smoothing import *

RECORD_COUNT = 3000
MAX_24B = 2**23 - 1
MIN_24B = -2**23
VOLTAGE_RANGE = 0.4  # -0.2V to 0.2V
ADC_COUNT = 5
TARGET_ADC = 3  # ADC to analyze for functions
SEGMENT_LENGTH_MS = 120_000

class ADCdata:
    def __init__(self):
        self.timestamps = np.array([])
        self.adc_output_data = [np.array([]) for _ in range(ADC_COUNT)]
        self.adc_normalized_data = [np.array([]) for _ in range(ADC_COUNT)]
        self.adc_voltage_means = []
        self.breath_count = 0
        self.avg_breath_depth = 0
        self.avg_breath_depth_std_dev = 0


def parse_line_only_adc(line: str):
    parts = line.strip().split(',')
    hour = int(parts[0].split(':')[1])
    minute = int(parts[1].split(':')[1])
    second = int(parts[2].split(':')[1])
    millisecond = int(parts[3].split(':')[1])
    ms_timestamp = (hour * 3600 + minute * 60 + second) * 1000 + millisecond

    # bit merging
    def get_adc(start_index):
        return u2_to_i(0,
                       int(parts[start_index].split(':')[1]),
                       int(parts[start_index + 1].split(':')[1]),
                       int(parts[start_index + 2].split(':')[1]))

    adc_outputs = [get_adc(4 + i * 3) for i in range(ADC_COUNT)]
    return ms_timestamp, adc_outputs

def handle_input_data(input_file, adc_data):
    first_timestamp = None
    with open(f"./data/{input_file}", 'r') as i_f:
        for line in i_f:
            ms_timestamp, adc_outputs = parse_line_only_adc(line)
            if first_timestamp is None:
                first_timestamp = ms_timestamp
            adc_data.timestamps = np.append(adc_data.timestamps, ms_timestamp - first_timestamp)
            for i, v in enumerate(adc_outputs):
                adc_data.adc_output_data[i] = np.append(adc_data.adc_output_data[i], v)
                adc_data.adc_normalized_data[i] = np.append(adc_data.adc_normalized_data[i], round(adc_to_voltage(v), 10))

def handle_results_data(input_file, adc_data):    
    with open(f"./results/results_{input_file}_temp.jsonl", 'w') as o_f:
        for i in range(len(adc_data.timestamps)):
            record = {
                "timestamp": int(adc_data.timestamps[i]),
                "adc_outputs": [int(adc_data.adc_output_data[a][i]) for a in range(ADC_COUNT)],
                "adc_voltages": [float(adc_data.adc_normalized_data[a][i]) for a in range(ADC_COUNT)],
            }
            o_f.write(json.dumps(record) + "\n")

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
    plot_data(input_file, adc_data, avg_breath_depth)
    with open(f"./features/features_{input_file}.txt", 'w') as o_f:
        o_f.write(f"breath count for {input_file}: {adc_data.breath_count} for {adc_data.timestamps[-1] - adc_data.timestamps[0]}ms -> {adc_data.breath_count/((adc_data.timestamps[-1] - adc_data.timestamps[0])/60_000)} bpm\n")
        o_f.write(f"breath depth: {avg_breath_depth}\n")
        o_f.write(f"breath depth std: {avg_breath_depth_std_dev*2}\n")

    print(f"breath count for {input_file}: {adc_data.breath_count} for {adc_data.timestamps[-1] - adc_data.timestamps[0]}ms -> {adc_data.breath_count/((adc_data.timestamps[-1] - adc_data.timestamps[0])/60_000)} bpm")
    print(f"breath depth: {avg_breath_depth}")
    print(f"breath depth std: {avg_breath_depth_std_dev*2}")

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

def split_data_into_segments(input_file, adc_data):
    segment_index = 0
    total_segments = int(np.ceil(adc_data.timestamps[-1] / SEGMENT_LENGTH_MS))
    for segment_index in range(total_segments):
        segment_start = segment_index * SEGMENT_LENGTH_MS
        segment_end = segment_start + SEGMENT_LENGTH_MS
        segment_fill_percentage = 0
        with open(f"./results/results_{input_file}_segment_{segment_index}.jsonl", 'w') as o_f:
            for i in range(len(adc_data.timestamps)):
                if segment_start <= adc_data.timestamps[i] < segment_end:
                    record = {
                        "timestamp": int(adc_data.timestamps[i]),
                        "adc_outputs": [int(adc_data.adc_output_data[a][i]) for a in range(ADC_COUNT)],
                        "adc_voltages": [float(adc_data.adc_voltage_data[a][i]) for a in range(ADC_COUNT)],
                    }
                    o_f.write(json.dumps(record) + "\n")
                    segment_fill_percentage += 1
        segment_fill_percentage = (segment_fill_percentage*100 / SEGMENT_LENGTH_MS) * 100
        print(f"Segment {segment_index}: {segment_fill_percentage:.2f}% filled")

def process_file(input_file):
    
    adc_data = ADCdata()

    handle_input_data(input_file, adc_data)

    for i in range(ADC_COUNT):
            mean_voltage = np.mean(adc_data.adc_normalized_data[i])
            adc_data.adc_voltage_means.append(round(mean_voltage, 10))
            adc_data.adc_normalized_data[i] -= adc_data.adc_voltage_means[i]

    handle_results_data(input_file, adc_data)

    basic_feature_extraction(adc_data, input_file)
    # breath separation as described in the paper
    breath_separation(adc_data=adc_data, target_adc=TARGET_ADC)

def main():
    input_file = sys.argv[1]

    process_file(input_file)


if __name__ == "__main__":
    main()