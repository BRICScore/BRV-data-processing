import sys
from visualize_data import FeatureData, create_indices_for_features, feature_loading

sys.path.append("utils")
from config import *

def parse_jsonl_line(line):
    """
        Parses a single line of a JSONL file and extracts the timestamp and ADC values.

        Parameters
        ----------
        line : str
            A single line from a JSONL file containing a JSON object with 'timestamp' and 'adc_outputs'.
        
        Returns
        -------
        timestamp : int
            The timestamp extracted from the JSON object.
        adc_values : list of int
            A list of ADC output values extracted from the JSON object.
        
        Side Effects
        ------------
        This function has no side effects.
    """

    data = json.loads(line)
    timestamp = data.get('timestamp', 0)
    adc_values = [data.get('adc_outputs')[i] for i in range(ADC_COUNT)]
    return timestamp, adc_values

def extract_breath_features(signal):
    """ 
        Initializes breath features and extracts depth and length from a single breath signal.

        Parameters
        ----------
        signal : list of float
            The normalized ADC signal corresponding to a single breath.

        Returns
        -------
        dict
            A dictionary containing the initialized and extracted breath features:
            depth, length, inhale duration, inspiratory pause duration, exhale duration
            and expiratory pause duration.

        Side Effects        
        ------------
        This function has no side effects.
    """

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
    """
    Detects breath peaks (minima and maxima) in the given signal using the scipy.signal.find_peaks function.

    Parameters
    ----------
    signal : list of float
        The normalized ADC signal from which to detect breath peaks.

    Returns
    -------
    tuple: list of int (minima), list of int (maxima)
        A tuple containing the indices of the detected minima and maxima.

    Side Effects
    ------------
    This function has no side effects.
    """

    inverted_signal = [-x for x in signal]
    mean_signal = np.mean(signal)
    std_dev_signal = np.std(signal)
    maxima, _ = scipy.signal.find_peaks(signal, distance=MIN_DISTANCE, height=mean_signal + std_dev_signal*STD_DEV_CONST)
    minima, _ = scipy.signal.find_peaks(inverted_signal, distance=MIN_DISTANCE, height=mean_signal + std_dev_signal*STD_DEV_CONST)

    return minima, maxima

def get_mode_breath(all_breath_data, mode_values, average_values):
    """ 
    Identifies the most representative breath (mode breath) from the given breath dataset. The choice
    is based on a weighted distance metric that considers the breath's depth, length and breathing 
    phases' durations (inhale and exhale, the inspiratory and expiratory pauses have been excluded).
    
    Parameters
    ----------
    all_breath_data : list of dict
        A list of dictionaries, where each dictionary contains the features of a single breath.
    mode_values : dict
        A dictionary containing the mode values for breath features (depth, length, inhale duration,
        exhale duration, inspiratory pause duration, expiratory pause duration).
    average_values : dict
        A dictionary containing the average values for breath features (depth, length, inhale duration,
        exhale duration, inspiratory pause duration, expiratory pause duration).

    Returns
    -------
    representative_breath : dict
        A dictionary containing the features of the most representative breath (mode breath) from the dataset.

    Side Effects
    ------------
    This function has no side effects.
    """

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
    """ 
    Calculates the durations of the breathing phases (inhale, inspiratory pause, exhale, expiratory pause) for a single breath.

    Parameters
    ----------
    breath_signal : list of float
        The normalized ADC signal corresponding to a single breath.
    breath_timestamps : list of int
        The timestamps corresponding to the breath signal.
    peak_index : int
        The index of the peak (maximum) in the breath signal.
    start_idx : int
        The index of the start of the breath (first minimum).
    end_idx : int
        The index of the end of the breath (the minimum after the peak).

    Returns
    -------
    phases : list of float
        A list containing the durations of the breathing phases: [inhale, inspiratory pause, exhale, expiratory pause].
    
    Side Effects
    ------------
    This function has no side effects.
    """

    if len(breath_signal) < 3:
        return 0.0, 0.0, 0.0, 0.0
    
    phases = [0.0, 0.0, 0.0, 0.0]  # inhale, inspiratory_pause, exhale, expiratory_pause
    phases[0] = breath_timestamps[start_idx] - breath_timestamps[peak_index]
    phases[1] = 0.0
    phases[2] = breath_timestamps[end_idx] - breath_timestamps[peak_index]
    phases[3] = 0.0

    return phases

def extract_breath_data_from_file(file):
    """
    Extracts minima and maxima indices from a given file (from the ./results directory)
    as well as the corresponding timestamps and ADC signals.

    Parameters
    ----------
    file : str
        The name of the file from which to extract breath data. The file should be located in the ./results directory.

    Returns
    -------
    dict
        A dictionary containing the extracted breath data:
        - minima: A list of indices of detected minima in the ADC signal.
        - maxima: A list of indices of detected maxima in the ADC signal.
        - timestamps: A list of timestamps corresponding to the ADC signal samples.
        - adc_signals: A list of lists, each inner list contains the ADC signal values for one of the 5 ADC channels.

    Side Effects
    ------------
    This function has no side effects.
    """

    with open (f'./results/{file}', 'r') as f:
        file_lines = f.read().strip().split("\n")
    timestamps = []
    adc_signals = [[] for _ in range(5)]

    for line in file_lines:
        if line:
            timestamp, adc_values = parse_jsonl_line(line)
            timestamps.append(timestamp)
            for i in range(5):
                adc_signals[i].append(adc_values[i])
    minima, maxima = detect_breath_peaks(adc_signals[TARGET_ADC])
    return { "minima": minima, "maxima": maxima, "timestamps": timestamps, "adc_signals": adc_signals }

def calculate_breath_characteristics(people_files):
    """
    Calculates breath characteristics (depth, length, inhale duration, exhale duration) for each breath in the given files.

    Parameters
    ----------
    people_files : list of str
        A list of file names (located in the ./results directory) from which to extract and calculate breath
        characteristics - they describe the breathing patterns of a single person.

    Returns
    -------
    all_breath_data : list of dict
        A list of dictionaries, where each dictionary contains the features of a single breath extracted from the given files.

    Side Effects
    ------------
    This function has no side effects.
    """

    all_breath_data = []
    for file in people_files:
        file_breath_data = extract_breath_data_from_file(file)
        minima = file_breath_data["minima"]
        maxima = file_breath_data["maxima"]
        timestamps = file_breath_data["timestamps"]
        adc_signals = file_breath_data["adc_signals"]

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

            # Find all maxima within the breath
            peaks_in_breath = []
            for peak in maxima:
                if start_idx <= peak < end_idx:
                    peaks_in_breath.append(peak)
                    break

            # If there is more than one peak we want to take the "highest" one
            peak_in_breath = np.max(peaks_in_breath) if peaks_in_breath else None

            if peak_in_breath is not None:
                phases = calculate_breathing_phases_for_breath(adc_signals[TARGET_ADC], timestamps, peak_in_breath, start_idx, end_idx)
                breath_features['inhale'] = phases[0]
                breath_features['inspiratory_pause'] = phases[1]
                breath_features['exhale'] = phases[2]
                breath_features['expiratory_pause'] = phases[3]

                all_breath_data.append(breath_features)  
    return all_breath_data          

def get_mode_param(all_breath_data, param):
    """ 
    Calculates the mode value for a given breath parameter (e.g., depth, length, inhale duration) from the breath dataset.

    Parameters
    ---------- 
    all_breath_data : list of dict
        A list of dictionaries, where each dictionary contains the features of a single breath.
    param : str
        The name of the breath parameter for which to calculate the mode (e.g., 'depth', 'length', 'inhale', 'exhale').

    Returns
    -------
    mode_value : float
        The mode value for the specified breath parameter, calculated using a histogram-based approach.

    Side Effects
    ------------
    This function has no side effects.
    """

    values = [breath[param] for breath in all_breath_data]
    non_zero_values = [v for v in values if v != 0]
    if len(non_zero_values) == 0:
        return 0.0
    
    hist, bin_edges = np.histogram(non_zero_values, bins=50)
    max_count = np.max(hist)
    max_bins = np.where(hist == max_count)[0]

    if len(max_bins) > 1:
        mean_value = np.mean(values)
        closest_bin = min(max_bins, key=lambda b: abs((bin_edges[b] + bin_edges[b + 1]) / 2 - mean_value))
        mode_upper_bin = closest_bin
    else:
        mode_upper_bin = np.argmax(hist)
        
    mode_depth = (bin_edges[mode_upper_bin] + bin_edges[mode_upper_bin + 1]) / 2
    return mode_depth

def calculate_mode_and_avg_features(all_breath_data):
    """
    Calculates the average and mode values for breath characteristics (depth, length, inhale duration,
    exhale duration) from the breath dataset.

    Parameters
    ----------
    all_breath_data : list of dict
        A list of dictionaries, where each dictionary contains the features of a single breath.

    Returns
    -------
    average_values : dict
        A dictionary containing the average values for breath characteristics (depth, length, inhale duration,
        exhale duration).
    mode_values : dict
        A dictionary containing the mode values for breath characteristics (depth, length, inhale duration,
        exhale duration).
    
    Side Effects
    ------------
    This function has no side effects.
    """

    rounding_precision = 10
    average_values = {
        "avg_depth": round(np.mean([breath['depth'] for breath in all_breath_data]), rounding_precision),
        "avg_length": round(np.mean([breath['length'] for breath in all_breath_data]), rounding_precision),
        "avg_inhale": round(np.mean([breath['inhale'] for breath in all_breath_data]), rounding_precision),
        "avg_inspiratory_pause": round(np.mean([breath['inspiratory_pause'] for breath in all_breath_data]), rounding_precision),
        "avg_exhale": round(np.mean([breath['exhale'] for breath in all_breath_data]), rounding_precision),
        "avg_expiratory_pause": round(np.mean([breath['expiratory_pause'] for breath in all_breath_data]), rounding_precision),
    }

    mode_values = {
        "mode_depth": round(get_mode_param(all_breath_data, 'depth'), rounding_precision),
        "mode_length": round(get_mode_param(all_breath_data, 'length'), rounding_precision),
        "mode_inhale": round(get_mode_param(all_breath_data, 'inhale'), rounding_precision),
        "mode_inspiratory_pause": round(get_mode_param(all_breath_data, 'inspiratory_pause'), rounding_precision),
        "mode_exhale": round(get_mode_param(all_breath_data, 'exhale'), rounding_precision),
        "mode_expiratory_pause": round(get_mode_param(all_breath_data, 'expiratory_pause'), rounding_precision),
    }
    return average_values, mode_values

def create_data_profiles():
    """
    Creates data profiles for each person in the analyzed dataset (./results) by processing their respective files
    and calculating a set of characteristics.

    Parameters
    ----------
    None

    Returns
    -------
    people_profiles : dict
        A dictionary where each key is a person's identifier (e.g., initials) and the value is another dictionary containing:
        - average_values: A dictionary of average breath characteristics (depth, length, inhale duration, exhale duration).
        - mode_values: A dictionary of mode breath characteristics (depth, length, inhale duration, exhale duration).
        - mode_breath: A dictionary containing the signal and timestamps of the mode breath for that person.

    Side Effects
    ------------
    This function has no side effects.
    """
    files = [file.name for file in os.scandir('./results')]

    people_files = dict()
    people_profiles = dict()
    for file in files:
        if pathlib.Path(file).suffix != '.jsonl':
            continue
        person = file.split("_")[3]
        if person not in people_files:
            people_files[person] = []
        people_files[person].append(file)

    for person in people_files:
        all_breath_data = calculate_breath_characteristics(people_files[person])

        average_values, mode_values = calculate_mode_and_avg_features(all_breath_data)    
        mode_breath = get_mode_breath(all_breath_data, mode_values, average_values)
        people_profiles[person] = {"average_values": average_values, "mode_values": mode_values, "mode_breath": mode_breath}

    return people_profiles

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
    people_profiles = create_data_profiles()

    for person in people_profiles:
        plot_mode_breath_on_file(people_profiles[person]["mode_breath"])

    group_plot_mode_breaths(people_profiles)
    create_profile_from_features()

if __name__ == "__main__":
    main()