import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
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
    min_distance = 15

    mean_signal = np.mean(signal)
    maxima, _ = scipy.signal.find_peaks(signal, distance=min_distance)
    for peak in maxima:
        if signal[peak] < mean_signal:
            maxima = np.delete(maxima, np.where(maxima == peak))

    minima = []
    minima, _ = scipy.signal.find_peaks([-s for s in signal], distance=min_distance)
    for minimum in minima:
        if signal[minimum] > mean_signal:
            minima = np.delete(minima, np.where(minima == minimum))

    return maxima, minima

def get_mode_breath(all_breath_data, mode_values):
    mode_dist = float('inf')
    mode_breath = None

    for breath in all_breath_data:
        dist = 0
        dist += (breath['depth'] - mode_values['mode_depth'])
        dist += (breath['length'] - mode_values['mode_length'])
        # dist += (breath['inhale'] - mode_values['mode_inhale'])
        # dist += (breath['inspiratory_pause'] - mode_values['mode_inspiratory_pause'])
        # dist += (breath['exhale'] - mode_values['mode_exhale']) 
        # dist += (breath['expiratory_pause'] - mode_values['mode_expiratory_pause'])

        if dist < mode_dist:
            mode_dist = dist
            mode_breath = breath

    return mode_breath

def calculate_breathing_phases_for_breath(breath_signal, breath_timestamps, peak_index, minima, start_idx, end_idx):
    if len(breath_signal) < 3:
        return 0.0, 0.0, 0.0, 0.0
    
    local_minima = [min - start_idx for min in minima if start_idx <= min < end_idx]
    
    min_before_peak = 0
    for min_idx in local_minima:
        if min_idx < peak_index:
            min_before_peak = min_idx

    min_after_peak = len(breath_signal) - 1
    for min_idx in local_minima:
        if min_idx > peak_index:
            min_after_peak = min_idx
            break

    phases = [0.0, 0.0, 0.0, 0.0]  # inhale, inspiratory_pause, exhale, expiratory_pause
    phases[0] = breath_timestamps[peak_index] - breath_timestamps[min_before_peak] if min_before_peak < peak_index else 0.0
    phases[1] = 0.0
    phases[2] = breath_timestamps[min_after_peak] - breath_timestamps[peak_index] if min_after_peak > peak_index else 0.0
    phases[3] = breath_timestamps[-1] - breath_timestamps[min_after_peak] if min_after_peak < len(breath_signal) - 1 else 0.0

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
        # plt.figure(figsize=(12, 6))
        # plt.plot(timestamps, adc_signals[TARGET_ADC], label='ADC Signal', color='blue')
        # plt.scatter([timestamps[i] for i in maxima], [adc_signals[TARGET_ADC][i] for i in maxima], color='red', label='Detected Peaks', marker='x')
        # plt.scatter([timestamps[i] for i in minima], [adc_signals[TARGET_ADC][i] for i in minima], color='green', label='Detected Valleys', marker='o')
        # plt.title(f'Breath Signal with Detected Peaks for file {file}')
        # plt.xlabel('Time (ms)')
        # plt.ylabel('ADC Voltage')
        # plt.legend()
        # plt.grid()
        # plt.show()
        breath_features = {}
        for i in range(len(maxima)-1):
            start_idx = maxima[i]
            end_idx = maxima[i+1]
            breath_signal = adc_signals[TARGET_ADC][start_idx:end_idx]
            breath_timestamps = timestamps[start_idx:end_idx]
            if len(breath_signal) == 0:
                continue
            breath_features = extract_breath_features(breath_signal)

        for i in range(len(minima)-1):
            start_idx = minima[i]
            end_idx = minima[i+1]
            breath_signal = adc_signals[TARGET_ADC][start_idx:end_idx]
            breath_timestamps = timestamps[start_idx:end_idx]
            

            if len(breath_signal) == 0:
                continue

            # nw czemu ale dla minimów sié wywala, więć jest jak wyżej XD
            # breath_features = extract_breath_features(breath_signal)
            breath_features['signal'] = breath_signal
            breath_features['timestamps'] = breath_timestamps

            peak_in_breath = None
            for peak in maxima:
                if start_idx < peak < end_idx:
                    peak_in_breath = peak - start_idx
                    break

            if peak_in_breath is not None:
                phases = calculate_breathing_phases_for_breath(breath_signal, breath_timestamps, peak_in_breath, minima, start_idx, end_idx)
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

def visualize_profiles(profiles_dict):
    # To napisał copilot, wykresy są trudne

    people = list(profiles_dict.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Breathing Profiles: Average vs Mode', fontsize=16)
    
    params = ['depth', 'length']
    for idx, param in enumerate(params):
        avg_values = [profiles_dict[person][0][f'avg_{param}'] for person in people]
        mode_values = [profiles_dict[person][1][f'mode_{param}'] for person in people]
        
        x = np.arange(len(people))
        width = 0.35
        
        axes[idx].bar(x - width/2, avg_values, width, label='Average', color='blue')
        axes[idx].bar(x + width/2, mode_values, width, label='Mode', color='orange')
        
        axes[idx].set_xlabel('People')
        axes[idx].set_ylabel(f'{param.capitalize()}')
        axes[idx].set_title(f'Average vs Mode {param.capitalize()}')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(people)
        axes[idx].legend()
        axes[idx].grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def visualize_mode_breaths(profiles_dict):
    """Plot mode breath signals for all people"""
    people = list(profiles_dict.keys())
    n_people = len(people)
    
    fig, axes = plt.subplots(n_people, 1, figsize=(12, 2 * n_people))
    if n_people == 1:
        axes = [axes]
    
    fig.suptitle('Mode Breath Signals by Person', fontsize=16)
    
    for idx, person in enumerate(people):
        mode_breath = profiles_dict[person][2]
        
        if mode_breath is None:
            axes[idx].text(0.5, 0.5, f'No mode breath data for {person}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'{person}')
            continue
        
        signal = np.array(mode_breath['signal'])
        # Normalize signal to [0, 1]
        signal_min = np.min(signal)
        signal_max = np.max(signal)
        if signal_max > signal_min:
            normalized_signal = (signal - signal_min) / (signal_max - signal_min)
        else:
            normalized_signal = signal
        
        axes[idx].plot(range(len(normalized_signal)), normalized_signal, linewidth=2, color='blue')
        axes[idx].set_title(f'{person}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Sample Index', fontsize=10)
        axes[idx].set_ylabel('Normalized ADC Value', fontsize=10)
        axes[idx].grid(alpha=0.3, linestyle='--')
        axes[idx].fill_between(range(len(normalized_signal)), normalized_signal, alpha=0.2, color='blue')
        axes[idx].set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig('./results/mode_breaths.png', dpi=150)
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
    visualize_mode_breaths(profiles)

from visualize_data import FeatureData, create_indices_for_features, feature_loading
import math

def plot_profiles(profiles):
    length = len(profiles)
    fig, ax = plt.subplots(length, 1)  
    i = 0
    no_of_points = 10
    alpha = 0.3
    total_length = 0
    biggest_height = 0
    for key, values in profiles.items():
        biggest_height = max(biggest_height, values["breath_depth_mode"])
        total_length = max(total_length, (values["inhale_length_mode"]+values["ip_length_mode"]+values["exhale_length_mode"]+values["ep_length_mode"])/1000)
    for key, values in profiles.items():
        bps = 1/(values["bpm_mode"]/60)
        inhale_x = np.linspace(0, values["inhale_length_mode"]/1000, no_of_points)
        inhale_y = [values["breath_depth_mode"] * math.sin(math.pi * 2 * inhale_x[x] / (4*values["inhale_length_mode"]/1000)) for x in range(no_of_points)]
        ax[i].plot(inhale_x, inhale_y)
        ax[i].add_patch(patches.Rectangle(
        (0.0, 0.0),   # (x,y)
        values["inhale_length_mode"]/1000,          # width
        values["breath_depth_mode"],          # height
        color = "blue",
        alpha = alpha)) #transparency

        ip_start = values["inhale_length_mode"]/1000
        ip_x = np.linspace(ip_start, ip_start+values["ip_length_mode"]/1000, no_of_points)
        ip_y = [inhale_y[-1] for _ in ip_x]
        ax[i].plot(ip_x, ip_y)
        ax[i].add_patch(patches.Rectangle(
        (ip_x[0], 0.0),   # (x,y)
        values["ip_length_mode"]/1000,          # width
        values["breath_depth_mode"],          # height
        color = "orange",
        alpha = alpha)) #transparency

        exhale_start = ip_start+values["ip_length_mode"]/1000
        exhale_x = np.linspace(0, values["exhale_length_mode"]/1000, no_of_points)
        exhale_y = [values["breath_depth_mode"] * math.cos(math.pi * 2 * exhale_x[x] / (4*values["exhale_length_mode"]/1000)) for x in range(no_of_points)]
        exhale_x += exhale_start
        ax[i].add_patch(patches.Rectangle(
        (exhale_x[0], 0.0),   # (x,y)
        values["exhale_length_mode"]/1000,          # width
        values["breath_depth_mode"],          # height
        color = "red",
        alpha = alpha)) #transparency

        ep_start = exhale_x[-1]
        ep_x = np.linspace(ep_start, ep_start+values["ep_length_mode"]/1000, no_of_points)
        ep_y = [exhale_y[-1] for _ in ep_x]
        ax[i].plot(ep_x, ep_y)
        ax[i].add_patch(patches.Rectangle(
        (ep_x[0], 0.0),   # (x,y)
        values["ep_length_mode"]/1000,          # width
        values["breath_depth_mode"],          # height
        color = "green",
        alpha = alpha)) #transparency

        ax[i].plot(exhale_x, exhale_y)
        ax[i].set_xlabel("time [s]")
        ax[i].set_xlim(xmin=0.0, xmax=total_length)
        ax[i].set_ylabel("breath_depth")
        ax[i].set_ylim(ymin=0.0, ymax=biggest_height)
        ax[i].set_title(key)
        i += 1
    fig.tight_layout(pad=0.5)
    plt.show()


def create_profile_from_features():
    feature_data = FeatureData()
    create_indices_for_features(feature_data)

    feature_data.features = feature_loading(feature_data)
    # print(feature_data.person_indices)
    people_profiles = {}
    for person in feature_data.person_initials:
        records = feature_data.features[feature_data.person_indices[person]]
        # print(records[0])
        dicts = []
        for i in range(len(feature_data.person_indices[person])):
            dicts.append({
                "bpm": records[i][0],
                "breath_depth": records[i][1],
                "inhale_length": records[i][13],
                "ip_length": records[i][14],
                "exhale_length": records[i][15],
                "ep_length": records[i][16]
            })
        
        people_profiles[person] = {
            "bpm_mode": get_mode_param(dicts, "bpm"),
            "breath_depth_mode": get_mode_param(dicts, "breath_depth"),
            "inhale_length_mode": get_mode_param(dicts, "inhale_length"),
            "ip_length_mode": get_mode_param(dicts, "ip_length"),
            "exhale_length_mode": get_mode_param(dicts, "exhale_length"),
            "ep_length_mode": get_mode_param(dicts, "ep_length")
        }
    
    plot_profiles(profiles=people_profiles)
        

def main():
    create_data_profiles()
    create_profile_from_features()


if __name__ == "__main__":
    main()