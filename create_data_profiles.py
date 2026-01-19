import os
import json
import sys
import numpy as np

sys.path.append("utils")
from config import *

def get_people_list(files):
    people = set()
    for file in files:
        file_person = file.split("_")[3]
        people.add(file_person)
    return list(people)

def parse_jsonl_line(line):
    data = json.loads(line)
    timestamp = data.get('timestamp', 0)
    adc_values = [data.get('adc_outputs')[i] for i in range(ADC_COUNT)]
    return timestamp, adc_values

def extract_breath_features(signal):
    depth = np.max(signal) - np.min(signal)
    length = len(signal)
    peak_index = np.argmax(signal)
    asymmetry = peak_index / length

    if length > 2:
        second_deriv = np.diff(signal, n=2)
        smoothness = 1 / (np.mean(np.abs(second_deriv)) + 1e-6)
    else:
        smoothness = 0
    return {
        "depth": float(depth),
        "length": int(length),
        "asymmetry": float(asymmetry),      # chat proposed this as a feature
        "smoothness": float(smoothness)     # chat proposed this as a feature
    }

def detect_breath_peaks(signal):
    peak_indices =[]
    min_distance = 20
    last_peak = -min_distance

    for i in range(1, len(signal) - 1):
        # print(signal[i], signal[i - 1], signal[i + 1])
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            if i - last_peak >= min_distance:
                peak_indices.append(i)
                last_peak = i

    return peak_indices

def calculate_breath_characteristics(people_files, all_breath_data):
    for file in people_files:
        with open(f'./results/{file}', 'r') as f:
            file_lines = f.read().strip().split("\n")
        timestamps = []
        adc_signals = [[] for _ in range(5)]

        for line in file_lines:
            if line:
                timestamp, adc_values = parse_jsonl_line(line)
                # print(f"Timestamp: {timestamp}, ADC Values: {adc_values}")
                timestamps.append(timestamp)
                for i in range(5):
                    adc_signals[i].append(adc_values[i])

        peak_indices = detect_breath_peaks(adc_signals[TARGET_ADC])
        # print(peak_indices)

        for i in range(len(peak_indices) - 1):
            start_idx = peak_indices[i]
            end_idx = peak_indices[i + 1]
            breath_signal = adc_signals[TARGET_ADC][start_idx:end_idx]
            breath_features = extract_breath_features(breath_signal)
            all_breath_data.append(breath_features)

def create_person_profile(all_breath_data):
    average_values = {
        "avg_depth": round(np.mean([breath['depth'] for breath in all_breath_data]), 10),
        "avg_length": round(np.mean([breath['length'] for breath in all_breath_data]), 10),
        "avg_asymmetry": round(np.mean([breath['asymmetry'] for breath in all_breath_data]), 10),
        "avg_smoothness": round(np.mean([breath['smoothness'] for breath in all_breath_data]), 10),
    }
    return average_values

def create_data_profiles():
    files = [file.name for file in os.scandir('./results')]
    people = get_people_list(files)
    # print(people)

    for person in people:
        people_files = [file for file in files if f"_{person}_" in file]
        # print(people_files)
        # depth, length, shape features
        all_breath_data = []
        calculate_breath_characteristics(people_files, all_breath_data)
        profile = create_person_profile(all_breath_data)
        print(f"Profile for {person}: {profile}")



    





def main():
    create_data_profiles()

if __name__ == "__main__":
    main()