import sys
import json
import matplotlib.pyplot as plt
import numpy as np

from helper_functions import *
from breath_separation import *
from feature_extraction import *

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