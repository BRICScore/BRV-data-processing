import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import scipy

sys.path.append("utils")
from config import *

def get_people_list(files):
    people = set()
    for file in files:
        if pathlib.Path(file).suffix != '.jsonl':
            continue
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

    return {
        "depth": float(depth),
        "length": int(length),
        "inhale": 0.0,
        "inspiratory_pause": 0.0,
        "exhale": 0.0,
        "expiratory_pause": 0.0,
    }

def detect_breath_peaks(signal):
    maxima = []
    min_distance = 10

    mean_signal = np.mean(signal)
    mean_plus_signal = mean_signal + (mean_signal * 0.3)
    mean_minus_signal = mean_signal - (mean_signal * 0.3)
    maxima, _ = scipy.signal.find_peaks(signal, distance=min_distance)
    for peak in maxima:
        if signal[peak] < mean_plus_signal:
            maxima = np.delete(maxima, np.where(maxima == peak))

    minima = []
    minima, _ = scipy.signal.find_peaks([-s for s in signal], distance=min_distance)
    for minimum in minima:
        if signal[minimum] > mean_minus_signal:
            minima = np.delete(minima, np.where(minima == minimum))

    return maxima, minima

def get_mode_breath(all_breath_data, mode_values, average_values):
    mode_dist = float('inf')
    representative_breath = None

    weights = {
        'depth': 0.3,
        'length': 0.3,
        'inhale': 1.0,
        'exhale': 1.0,
    }

    for breath in all_breath_data:
        dist = 0
        dist += weights['depth'] * abs(breath['depth'] - average_values['avg_depth'])
        dist += weights['length'] * abs(breath['length'] - average_values['avg_length'])
        dist += weights['inhale'] * abs(breath['inhale'] - average_values['avg_inhale'])
        dist += weights['exhale'] * abs(breath['exhale'] - average_values['avg_exhale'])
        dist += weights['depth'] * abs(breath['depth'] - mode_values['mode_depth'])
        dist += weights['length'] * abs(breath['length'] - mode_values['mode_length'])
        dist += weights['inhale'] * abs(breath['inhale'] - mode_values['mode_inhale'])
        dist += weights['exhale'] * abs(breath['exhale'] - mode_values['mode_exhale']) 
        # dist += (breath['inspiratory_pause'] - mode_values['mode_inspiratory_pause'])
        # dist += (breath['expiratory_pause'] - mode_values['mode_expiratory_pause'])

        if dist < mode_dist:
            mode_dist = dist
            representative_breath = breath

    return representative_breath

def calculate_breathing_phases_for_breath(breath_signal, breath_timestamps, peak_index, start_idx, end_idx):
    if len(breath_signal) < 3:
        return 0.0, 0.0, 0.0, 0.0
    
    phases = [0.0, 0.0, 0.0, 0.0]  # inhale, inspiratory_pause, exhale, expiratory_pause
    phases[0] = breath_timestamps[start_idx] - breath_timestamps[peak_index]
    phases[1] = 0.0
    phases[2] = breath_timestamps[end_idx] - breath_timestamps[peak_index]
    phases[3] = 0.0

    return phases


def calculate_breath_characteristics(people_files, all_breath_data):
    for file in people_files:
        if not file.endswith('.jsonl'):
            continue
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
        maxima, minima = detect_breath_peaks(adc_signals[TARGET_ADC])

        # if 'MK' in file:
        #     plt.figure(figsize=(12, 6))
        #     plt.plot(timestamps, adc_signals[TARGET_ADC], label='ADC Signal', color='blue')
        #     plt.scatter([timestamps[i] for i in maxima], [adc_signals[TARGET_ADC][i] for i in maxima], color='red', label='Detected Peaks', marker='x')
        #     plt.scatter([timestamps[i] for i in minima], [adc_signals[TARGET_ADC][i] for i in minima], color='green', label='Detected Valleys', marker='o')
        #     plt.title(f'Breath Signal with Detected Peaks for file {file}')
        #     plt.xlabel('Time (ms)')
        #     plt.ylabel('ADC Voltage')
        #     plt.legend()
        #     plt.grid()
        #     plt.show()

        breath_features = {}
        for i in range(len(minima)-1):
            start_idx = minima[i]
            end_idx = minima[i+1]
            breath_signal = adc_signals[TARGET_ADC][start_idx:end_idx]
            breath_timestamps = timestamps[start_idx:end_idx]
            if len(breath_signal) == 0:
                continue
            breath_features = extract_breath_features(breath_signal)

            breath_features['file_name'] = file
            breath_features['signal'] = breath_signal
            breath_features['timestamps'] = breath_timestamps

            peaks_in_breath = []
            for peak in maxima:
                if start_idx <= peak < end_idx:
                    peaks_in_breath.append(peak)
                    break

            peaks_in_breath.sort()
            peak_in_breath = peaks_in_breath[0] if peaks_in_breath else None

            if peak_in_breath is not None:
                phases = calculate_breathing_phases_for_breath(adc_signals[TARGET_ADC], timestamps, peak_in_breath, start_idx, end_idx)
                breath_features['inhale'] = phases[0]
                breath_features['inspiratory_pause'] = phases[1]
                breath_features['exhale'] = phases[2]
                breath_features['expiratory_pause'] = phases[3]

                all_breath_data.append(breath_features)
            

def get_mode_param(all_breath_data, param):
    values = [breath[param] for breath in all_breath_data]
    non_zero_values = [v for v in values if v > 0]
    if len(non_zero_values) == 0:
        return 0.0
    hist, bin_edges = np.histogram(non_zero_values, bins=50)

    max_count = np.max(hist)
    # print(f"Histogram for {param}: {hist}, Bin edges: {bin_edges}")
    # print(f"Max count for {param}: {max_count}")
    # print(f"Values for {param}: {values}"   )

    max_bins = np.where(hist == max_count)[0]
    if len(max_bins) > 1:
        avg_value = np.mean(values)
        closest_bin = min(max_bins, key=lambda b: abs((bin_edges[b] + bin_edges[b + 1]) / 2 - avg_value))
        mode_upper_bin = closest_bin
    else:
        mode_upper_bin = np.argmax(hist)
        
    mode_depth = (bin_edges[mode_upper_bin] + bin_edges[mode_upper_bin + 1]) / 2
    return mode_depth

def create_person_profile(all_breath_data):
    # print(all_breath_data)

    average_values = {
        "avg_depth": round(np.mean([breath['depth'] for breath in all_breath_data]), 10),
        "avg_length": round(np.mean([breath['length'] for breath in all_breath_data]), 10),
        "avg_inhale": round(np.mean([breath['inhale'] for breath in all_breath_data]), 10),
        "avg_inspiratory_pause": round(np.mean([breath['inspiratory_pause'] for breath in all_breath_data]), 10),
        "avg_exhale": round(np.mean([breath['exhale'] for breath in all_breath_data]), 10),
        "avg_expiratory_pause": round(np.mean([breath['expiratory_pause'] for breath in all_breath_data]), 10),
    }

    mode_values = {
        "mode_depth": round(get_mode_param(all_breath_data, 'depth'), 10),
        "mode_length": round(get_mode_param(all_breath_data, 'length'), 10),
        "mode_inhale": round(get_mode_param(all_breath_data, 'inhale'), 10),
        "mode_inspiratory_pause": round(get_mode_param(all_breath_data, 'inspiratory_pause'), 10),
        "mode_exhale": round(get_mode_param(all_breath_data, 'exhale'), 10),
        "mode_expiratory_pause": round(get_mode_param(all_breath_data, 'expiratory_pause'), 10),
    }
    return average_values, mode_values

def create_data_profiles():
    files = [file.name for file in os.scandir('./results')]
    people = get_people_list(files)
    profiles ={}

    for person in people:
        people_files = []
        for file in files:
            if pathlib.Path(file).suffix != '.jsonl':
                continue
            if f"_{person}_" in file:
                people_files.append(file)
        
        all_breath_data = []
        calculate_breath_characteristics(people_files, all_breath_data)

        average_values, mode_values = create_person_profile(all_breath_data)    
        mode_breath = get_mode_breath(all_breath_data, mode_values, average_values)
        profiles[person] = (average_values, mode_values, mode_breath)
        # print(f"Profile for {person}: {(average_values, mode_values)}")
        
        # Plot mode breath on top of its source file
        if mode_breath:
            plot_mode_breath_on_file(mode_breath)
            # pass

    group_plot_mode_breaths(profiles)

def plot_mode_breath_on_file(mode_breath):
    file_name = mode_breath['file_name']
    file_path = f'./results/{file_name}'
    
    # Read the full file data
    with open(file_path, 'r') as f:
        file_lines = f.read().strip().split("\n")
    
    timestamps = []
    adc_signals = [[] for _ in range(5)]
    
    for line in file_lines:
        if line:
            timestamp, adc_values = parse_jsonl_line(line)
            timestamps.append(timestamp)
            for i in range(5):
                adc_signals[i].append(adc_values[i])
    
    mode_timestamps = mode_breath['timestamps']
    mode_signal = mode_breath['signal']

    plt.figure(figsize=(14, 7))
    plt.plot(timestamps, adc_signals[TARGET_ADC], label='Full Signal', color='blue', linewidth=1)
    plt.plot(mode_timestamps, mode_signal, label='Mode Breath', color='red', linewidth=2.5, alpha=0.8)
    plt.title(f'Mode Breath Highlighted on File: {file_name}')
    plt.xlabel('Time (ms)')
    plt.ylabel('ADC Voltage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'./results/mode_breath_{file_name.replace(".jsonl", ".png")}')
    plt.show()

def group_plot_mode_breaths(profiles):
    retimed_breaths_for_people = {}

    for person, (avg_values, mode_values, mode_breath) in profiles.items():
        if mode_breath is None:
            continue

        signal = mode_breath['signal']
        timestamps = mode_breath['timestamps']
        timestamps = [t - timestamps[0] for t in timestamps]
        retimed_breaths_for_people[person] = (timestamps, signal)

    plt.figure(figsize=(14, 7))
    for person, (timestamps, signal) in retimed_breaths_for_people.items():
        plt.plot(timestamps, signal, label=f'Mode Breath - {person}', linewidth=2.0, alpha=0.8)
    plt.title('Mode Breaths for All People')
    plt.xlabel('Time (ms)')
    plt.ylabel('ADC Voltage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'./results/mode_breaths_all_people.png')
    plt.show()

def main():
    create_data_profiles()

if __name__ == "__main__":
    main()