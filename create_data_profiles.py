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

    # for i in range(1, len(signal) - 1):
    #     # print(signal[i], signal[i - 1], signal[i + 1])
    #     if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
    #         if i - last_peak >= min_distance:
    #             peak_indices.append(i)
    #             last_peak = i

    peak_indices, _ = scipy.signal.find_peaks(signal, distance=min_distance)
    mean_signal = np.mean(signal)
    for peak in peak_indices:
        if signal[peak] < mean_signal:
            peak_indices = np.delete(peak_indices, np.where(peak_indices == peak))

    return peak_indices

def get_mode_breath(all_breath_data, mode_values):
    mode_dist = float('inf')
    mode_breath = None

    for breath in all_breath_data:
        dist = 0
        dist += (breath['depth'] - mode_values['mode_depth'])
        dist += (breath['length'] - mode_values['mode_length'])
        # dist += (breath['asymmetry'] - mode_values['mode_asymmetry'])
        # dist += (breath['smoothness'] - mode_values['mode_smoothness'])

        if dist < mode_dist:
            mode_dist = dist
            mode_breath = breath

    plt.figure(figsize=(10, 5))
    plt.plot(mode_breath['timestamps'], mode_breath['signal'], label='Mode Breath Signal', color='blue')
    plt.title('Mode Breath Signal')
    plt.xlabel('Time (ms)')
    plt.ylabel('ADC Voltage')
    plt.legend()
    plt.grid()
    plt.show()


    return mode_breath

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

        peak_indices = detect_breath_peaks(adc_signals[TARGET_ADC])

        for i in range(len(peak_indices) - 1):
            start_idx = peak_indices[i]
            end_idx = peak_indices[i + 1]
            breath_signal = adc_signals[TARGET_ADC][start_idx:end_idx]
            breath_features = extract_breath_features(breath_signal)
            breath_features['signal'] = breath_signal
            breath_features['timestamps'] = timestamps[start_idx:end_idx]
            all_breath_data.append(breath_features)

def get_mode_param(all_breath_data, param):
    values = [breath[param] for breath in all_breath_data]
    hist, bin_edges = np.histogram(values, bins=50)

    max_count = np.max(hist)
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
    average_values = {
        "avg_depth": round(np.mean([breath['depth'] for breath in all_breath_data]), 10),
        "avg_length": round(np.mean([breath['length'] for breath in all_breath_data]), 10),
        "avg_asymmetry": round(np.mean([breath['asymmetry'] for breath in all_breath_data]), 10),
        "avg_smoothness": round(np.mean([breath['smoothness'] for breath in all_breath_data]), 10),
    }

    mode_values = {
        "mode_depth": round(get_mode_param(all_breath_data, 'depth'), 10),
        "mode_length": round(get_mode_param(all_breath_data, 'length'), 10),
        "mode_asymmetry": round(get_mode_param(all_breath_data, 'asymmetry'), 10),
        "mode_smoothness": round(get_mode_param(all_breath_data, 'smoothness'), 10),
    }
    return average_values, mode_values

def visualize_profiles(profiles_dict):
    # To napisał copilot, wykresy są trudne

    """Plot average and mode values for all people"""
    people = list(profiles_dict.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Breathing Profiles: Average vs Mode', fontsize=16)
    
    # Extract data
    params = ['depth', 'length', 'asymmetry', 'smoothness']
    
    for idx, param in enumerate(params):
        ax = axes[idx // 2, idx % 2]
        
        avg_values = [profiles_dict[p][0][f'avg_{param}'] for p in people]
        mode_values = [profiles_dict[p][1][f'mode_{param}'] for p in people]
        
        x = np.arange(len(people))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, avg_values, width, label='Average', alpha=0.8)
        bars2 = ax.bar(x + width/2, mode_values, width, label='Mode', alpha=0.8)
        
        ax.set_ylabel(param.capitalize())
        ax.set_title(f'{param.capitalize()} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(people)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('./results/profiles_comparison.png', dpi=150)
    plt.show()

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
        # print(people_files)
        # depth, length, shape features
        all_breath_data = []
        calculate_breath_characteristics(people_files, all_breath_data)
        average_values, mode_values = create_person_profile(all_breath_data)    
        mode_breath = get_mode_breath(all_breath_data, mode_values)
        profiles[person] = (average_values, mode_values, mode_breath)
        print(f"Profile for {person}: {(average_values, mode_values)}")

        
        

    visualize_profiles(profiles)

def main():
    create_data_profiles()

if __name__ == "__main__":
    main()