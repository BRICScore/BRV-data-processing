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
    """The amount of zero-crossings in a signal divided by two.

    Parameters
    ----------
    adc_data : ADC_Data
        Data for current input file

    Returns
    -------
        None

    Side Effects
    ------------
        Sets the value in adc_data.breath_count
    """
    breath_counters = []
    for i in range(ADC_COUNT):
        breath_counters.append(len(np.where(np.diff(np.sign(adc_data.adc_normalized_data[i])))[0])/2)

    adc_data.breath_count = np.max(breath_counters)
    #adc_data.breath_count = breath_counters[TARGET_ADC]
    #----------------------------------------------------------------------------------------------

def calculate_average_breath_depth(adc_data, target_adc=TARGET_ADC):
    """
    Calculates breath depth for segment for specified ADC and returns values
    based on mean and standard deviation.
    
    Parameters
    ----------
    adc_data : ADC_Data
        Data for current input file

    Returns
    -------
    tuple: 
        (mean: float, std: float)
    
    Side Effects
    ------------
        Creates breath_peaks and their indices in adc_data for the segment.
    
    """
    min_spread_of_peaks = MIN_DISTANCE    # 10 Hz means the highest acceptable frequency of breaths is 1 per second (value/frequency)
    signal = adc_data.adc_normalized_data[target_adc]
    std_dev_signal = np.std(signal)
    mean_signal = np.mean(signal)
    min_value_for_peak = mean_signal + std_dev_signal*STD_DEV_CONST
    breath_peak_info = scipy.signal.find_peaks(x=adc_data.adc_normalized_data[target_adc],
                                                  height=min_value_for_peak,
                                                  distance=min_spread_of_peaks)
    breath_peak_indices, breath_dict = breath_peak_info
    breath_peaks = breath_dict['peak_heights']
    adc_data.breath_peaks = breath_peaks
    adc_data.breath_peak_indices = breath_peak_indices
    try:
        breath_peaks[0]
    except:
        return min_value_for_peak, min_value_for_peak
    avg_breath_depth = np.mean(breath_peaks)
    avg_breath_depth_std_dev = np.std(adc_data.adc_normalized_data[target_adc])
    return avg_breath_depth, avg_breath_depth_std_dev

def calculate_breathing_tract(adc_data):
    """
    Calculates the weights for each ADC(belt) where all of them sum up to 1.
    Each of the values is the percentage of the belt's stretch compared to others.
    
    Parameters
    ----------
    adc_data : ADC_Data
        Data for current input file

    Returns
    -------
    (belt_share, belt_share_std) : 
        tuple with 2 NDArrays with subsequent belt values
    
    Side Effects
    ------------
        This function overrides the adc_data.breath_peaks and adc_data.breath_peak_indices
    
    """
    belt_share = np.zeros(shape=(ADC_COUNT))
    belt_share_std = np.zeros(shape=(ADC_COUNT))
    avg_sum = 0
    avg_sum_std = 0
    for i in range(ADC_COUNT):
        avg, avg_std = calculate_average_breath_depth(adc_data, target_adc=i)
        avg_sum += avg
        avg_sum_std += avg_std
        belt_share[i-1] = avg
        belt_share_std[i-1] = avg_std
    belt_share /= avg_sum
    belt_share_std /= avg_sum_std
    return belt_share, belt_share_std

def detect_ep_end(adc_data):
    """
    Find all the points where phase 4 of breathing ends.
    
    Parameters
    ----------
    adc_data : ADC_Data
        Data for current input file

    Returns
    -------
        None
    
    Side Effects
    ------------
        This function registers the points in adc_data.breath_end_points
        and adc_data.breath_end_point_indices
    
    """
    adc_data.breath_end_points = []
    adc_data.breath_end_point_indices = []
    for i in range(len(adc_data.breath_minima)):
        breath_end = adc_data.breath_minimum_indices[i]
        pointFound = False
        while not pointFound:
            try:
                val_current = adc_data.adc_normalized_data[TARGET_ADC][breath_end]
                val_next = adc_data.adc_normalized_data[TARGET_ADC][breath_end+1]
                val_after_next = adc_data.adc_normalized_data[TARGET_ADC][breath_end+2]
            except:
                val_current = 0.0
                val_next = 0.0
                val_after_next = 0.0
            if val_current < val_next:
                if val_next < val_after_next:
                    pointFound = True
                else:
                    breath_end += 1
            else:
                breath_end += 1
            if breath_end == 0 or breath_end >= len(adc_data.timestamps)-1:
                break
        breath_end = min(breath_end, len(adc_data.adc_normalized_data[TARGET_ADC])-1)
        adc_data.breath_end_point_indices.append(breath_end)
        adc_data.breath_end_points.append(adc_data.adc_normalized_data[TARGET_ADC][breath_end])

# calculate by detecting where the data increases significantly
def detect_expiratory_pause(adc_data):
    """
    Find all the points where phase 4 of breathing starts.
    
    Parameters
    ----------
    adc_data : ADC_Data
        Data for current input file

    Returns
    -------
        None
    
    Side Effects
    ------------
        This function registers the points in adc_data.breath_minima
        adc_data.breath_minimum_indices
    
    """

    adc_data.breath_minimum_indices = []
    adc_data.breath_minima = []
    for i in range(len(adc_data.exhale_points)):
        minimum = adc_data.exhale_point_indices[i] + 2 # offset to counteract faulty exhale points
        pointFound = False
        while not pointFound:
            try:
                val_current = adc_data.adc_normalized_data[TARGET_ADC][minimum]
                val_next = adc_data.adc_normalized_data[TARGET_ADC][minimum+1]
                val_after_next = adc_data.adc_normalized_data[TARGET_ADC][minimum+2]
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
            else:
                minimum += 1
            if minimum == 0 or minimum >= len(adc_data.timestamps)-1:
                break
        minimum = min(minimum, len(adc_data.adc_normalized_data[TARGET_ADC])-1)
        adc_data.breath_minimum_indices.append(minimum)
        adc_data.breath_minima.append(adc_data.adc_normalized_data[TARGET_ADC][minimum])

# wait for data to stop decreasing
def detect_exhale(adc_data):
    """
    Find all the points where phase 3 of breathing starts.
    
    Parameters
    ----------
    adc_data : ADC_Data
        Data for current input file

    Returns
    -------
        None
    
    Side Effects
    ------------
        This function registers the points in adc_data.exhale_points
        and adc_data.exhale_point_indices
    
    """
    adc_data.exhale_point_indices = []
    adc_data.exhale_points = []
    for i in range(len(adc_data.breath_peaks)):
        exhale_point = adc_data.breath_peak_indices[i]
        pointFound = False
        while not pointFound:
            try:
                val_current = adc_data.adc_normalized_data[TARGET_ADC][exhale_point]
                val_next = adc_data.adc_normalized_data[TARGET_ADC][exhale_point+1]
                val_after_next = adc_data.adc_normalized_data[TARGET_ADC][exhale_point+2]
                val_after_after_next = adc_data.adc_normalized_data[TARGET_ADC][exhale_point+3]
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
        adc_data.exhale_points.append(adc_data.adc_normalized_data[TARGET_ADC][exhale_point])

# calculate by detecting where the data drops significantly
def detect_inspiratory_pause(adc_data):
    # the breath_peaks are calculated in the first call to calculate_average_breath_depth
    pass

# calculate start by going from the maxima backwards
def detect_inhale(adc_data):
    """
    Find all the points where phase 1 of breathing starts.
    
    Parameters
    ----------
    adc_data : ADC_Data
        Data for current input file

    Returns
    -------
        None
    
    Side Effects
    ------------
        This function registers the points in adc_data.inhale_points
        and adc_data.inhale_point_indices
    
    """

    adc_data.inhale_point_indices = []
    adc_data.inhale_points = []
    for i in range(len(adc_data.breath_peaks)):
        inhale_point = adc_data.breath_peak_indices[i]
        pointFound = False
        while not pointFound:
            val_current = adc_data.adc_normalized_data[TARGET_ADC][inhale_point]
            val_prev = adc_data.adc_normalized_data[TARGET_ADC][inhale_point-1]
            val_before_prev = adc_data.adc_normalized_data[TARGET_ADC][inhale_point-2]
            if val_current > val_prev:
                if val_prev > val_before_prev:
                    inhale_point -= 1
                else:
                    pointFound = True
                    inhale_point -= 1
            else:
                inhale_point -= 1
            if inhale_point <= 0 or inhale_point == len(adc_data.timestamps)-1:
                break
        adc_data.inhale_point_indices.append(inhale_point)
        adc_data.inhale_points.append(adc_data.adc_normalized_data[TARGET_ADC][inhale_point])


def calculate_breathing_phases(adc_data):
    """
    Calculations assuming local maxima is the end of inhale and start of inspiratory pause.
    
    Parameters
    ----------
    adc_data : ADC_Data
        Data for current input file

    Returns
    -------
    phase_values : 
        NDArray(4) where each following value is the value of a subsequent breathing phase
    
    Side Effects
    ------------
        This function has no side effects
    
    """
    detect_inhale(adc_data)
    detect_inspiratory_pause(adc_data)
    detect_exhale(adc_data)
    detect_expiratory_pause(adc_data)
    detect_ep_end(adc_data)
    phases_values = [0.0, 0.0, 0.0, 0.0]
    NPtimestamps = np.array(adc_data.timestamps)
    number_of_breaths = len(adc_data.breath_peaks)
    for i in range(number_of_breaths-1):
        phases_values[0] += NPtimestamps[adc_data.breath_peak_indices[i]] - NPtimestamps[adc_data.inhale_point_indices[i]]
        phases_values[1] += NPtimestamps[adc_data.exhale_point_indices[i]] - NPtimestamps[adc_data.breath_peak_indices[i]]
        phases_values[2] += NPtimestamps[adc_data.breath_minimum_indices[i]] - NPtimestamps[adc_data.exhale_point_indices[i]]
        phases_values[3] += NPtimestamps[adc_data.breath_end_point_indices[i]] - NPtimestamps[adc_data.breath_minimum_indices[i]]
        # until outliers are not dealt with
        """
        if i != number_of_breaths-1:
            phases_values[3] += NPtimestamps[adc_data.inhale_point_indices[i+1]] - NPtimestamps[adc_data.breath_minimum_indices[i]]
        """
    try:
        phases_values[0] /= number_of_breaths
        phases_values[1] /= number_of_breaths
        phases_values[2] /= number_of_breaths
        phases_values[3] /= number_of_breaths
    except:
        return [0.0, 0.0, 0.0, 0.0]
    return phases_values

def display_calculated_breath_phases(adc_data):
    plt.plot(adc_data.timestamps, adc_data.adc_normalized_data[TARGET_ADC])
    NPtimestamps = np.array(adc_data.timestamps)
    plt.scatter(NPtimestamps[adc_data.inhale_point_indices], adc_data.inhale_points, c="blue") # start of inhale
    plt.scatter(NPtimestamps[adc_data.breath_peak_indices], adc_data.breath_peaks, c="red") # start of IP
    plt.scatter(NPtimestamps[adc_data.exhale_point_indices], adc_data.exhale_points, c="green") # start of exhale
    plt.scatter(NPtimestamps[adc_data.breath_minimum_indices], adc_data.breath_minima, c="magenta") # start of EP
    plt.scatter(NPtimestamps[adc_data.breath_end_point_indices], adc_data.breath_end_points, c="yellow") # end of EP
    plt.legend(["signal","inhale start", "IP start", "exhale start", "EP start"])
    plt.xlabel("timestamp [ms]")
    plt.ylabel("signal deviation from average value")
    plt.show()

def calculate_respiratory_tract(adc_data):
    """
    This function is responsible for extracting the values for
    representation of the feature in the name using RMS

    Parameters
    ----------
    adc_data : ADC_Data
        Data for current input file

    Returns
    -------

    """
    # TODO
    TO_GENERATE = 20
    x = np.outer(np.ones(TO_GENERATE), np.array([1,2,3,4,5]))
    y = np.outer(np.linspace(1,20,20), np.ones(ADC_COUNT))
    print(y)
    z = []
    for i in range(ADC_COUNT):
        coefficients = calculate_breath_shape(adc_data=adc_data, target=i)
        a3, a2, a1, a0 = coefficients
        z.append([a3*p**3 + a2*p**2 + a1*p + a0 for p in range(1,20+1)])
    z = np.array(z).T
    print(z)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis', edgecolor='green')
    ax.set_title('Surface Plot')
    ax.view_init(20, -20)
    plt.show()

def calculate_breath_variability(adc_data):
    """
    This function is responsible for extracting the values for
    representation of the feature in the name using RMS (Root Mean Square)

    Parameters
    ----------
    adc_data : ADC_Data
        Data for current input file

    Returns
    -------
    rms : a NDArray with 5 elements representing RMS from each ADC

    """
    n = len(adc_data.adc_normalized_data[0])
    rms = np.zeros(shape=ADC_COUNT, dtype=np.float32)
    for i in range(ADC_COUNT):
        rms[i] = np.sqrt(np.sum(np.square(adc_data.adc_normalized_data[i]))/n)
    print(rms)
    return rms

def calculate_breath_shape(adc_data, target=TARGET_ADC):
    """
    This function interpolates the breath data of every single breath and estabilishes
    the best coefficients for the polynomial of a single breath calculating from
    all breaths in a segment.
    
    Parameters
    ----------
    adc_data : ADC_Data
        Data for current input file
    
    Returns
    -------
    coefficient_means : NDArray with mean coefficients calculated from all
        breaths from a segment by approximating a cubic polynomial

    """
    all_coefficients = []

    for i in range(len(adc_data.inhale_points)):
        # print(adc_data.inhale_point_indices[i], adc_data.inhale_point_indices[i+1])
        # print(len(adc_data.breath_end_point_indices), len(adc_data.inhale_point_indices))
        start, end = adc_data.inhale_point_indices[i], adc_data.breath_end_point_indices[i]
        x = [number for number in range(start,end+1)]
        x -= np.min(x)
        y = adc_data.adc_normalized_data[target][start:end+1]
        c = np.polyfit(x, y, 3)

        #plt.plot(x,y)
        domain = np.linspace(x[0], x[-1], 20)
        # print(c)
        a3, a2, a1, a0 = c
        y2 = [a3*x2**3 + a2*x2**2 + a1*x2 + a0 for x2 in domain]
        #plt.scatter(domain, y2)
        #plt.show()
        
        all_coefficients.append(c)
    coefficient_means = np.mean(np.array(all_coefficients), axis=0) #column-wise mean
    return coefficient_means

def display_specgram(adc_data, target=TARGET_ADC):
    """
    """
    segment_data = []
    min_length = np.min(np.array([adc_data.breath_end_point_indices[i] - adc_data.inhale_point_indices[i] for i in range(len(adc_data.inhale_points))]))
    for i in range(len(adc_data.inhale_points)):
        # print(adc_data.inhale_point_indices[i], adc_data.inhale_point_indices[i+1])
        # print(len(adc_data.breath_end_point_indices), len(adc_data.inhale_point_indices))
        start, end = adc_data.inhale_point_indices[i], adc_data.breath_end_point_indices[i]
        x = adc_data.adc_normalized_data[target][start:end+1]
        segment_data.append(x) #[:min_length]
        plt.specgram(x[:min_length])
        #plt.show()
    # sns.heatmap(segment_data)
    for breath in segment_data:
        plt.scatter([i for i in range(len(breath))], breath)
    plt.show()
        

def basic_feature_extraction(adc_data, input_file="test.txt"):
    """
    This function extracts all implemented features from the segment passed 
    and prints them in "extracted_features.jsonl"
    
    Parameters
    ----------
    adc_data : ADC_Data
        Data for current input file
    input_file : 
        name of the file(segment) with features being extracted

    Returns
    -------
        None
    
    Side Effects
    ------------
        This function appends an entry (feature vector) to the extracted_features.jsonl file.
        If you want to use it make sure to check whether the file should exist and have entries.
    
    """
    count_breaths(adc_data)
    bpm = adc_data.breath_count/((adc_data.timestamps[-1] - adc_data.timestamps[0])/60_000)
    if bpm < MIN_BPM or bpm > MAX_BPM: #discard criteria
        print(f"{input_file} discarded for inadequate breath count ({bpm})")
        return
    avg_breath_depth, avg_breath_depth_std_dev = calculate_average_breath_depth(adc_data)

    phases_avg_values = calculate_breathing_phases(adc_data)
    if phases_avg_values[INHALE_INDEX] < MIN_INHALE_OR_EXHALE_LENGTH or phases_avg_values[EXHALE_INDEX] < MIN_INHALE_OR_EXHALE_LENGTH:
        print(f"{input_file} discarded for inadequate phase lengths {phases_avg_values}")
        return
    if adc_data.debug_plot_enabled:
        display_calculated_breath_phases(adc_data) # do not move it takes values from two function calls above
    belt_share, belt_share_std = calculate_breathing_tract(adc_data)
    # calculate_breath_shape(adc_data)
    # calculate_breath_variability(adc_data=adc_data)
    # calculate_respiratory_tract(adc_data=adc_data)
    display_specgram(adc_data=adc_data)
    #-----------------------------------------------------------------------------------
    # nazewnictwo: feature_time_person_conditions(sit,lay,run)_(nr_próbki)_(nr_segmentu)
    # {"cecha1": 1.3, "cecha2": 0.45, …, "cecha12": [0.1, 0.2, 0.3, 0.4, 0.5]}
    if adc_data.plot_enabled:
        plot_data(input_file, adc_data, avg_breath_depth)
    with open(f"./features/extracted_features.jsonl", 'a') as o_f:
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
        o_f.write("], ")
        temp_feature_name = input_file.split("_")
        o_f.write(f"\"person\": \"{temp_feature_name[PERSON_ID]}_{temp_feature_name[ACTIVITY_ID]}\"")
        o_f.write("}\n")
    print(f"breath count for {input_file}: {adc_data.breath_count} for {adc_data.timestamps[-1] - adc_data.timestamps[0]}ms -> {adc_data.breath_count/((adc_data.timestamps[-1] - adc_data.timestamps[0])/60_000)} bpm")
    print(f"breath depth: {avg_breath_depth}")
    print(f"breath depth std: {avg_breath_depth_std_dev*2}")

    if adc_data.plot_enabled:
        plt.figure(figsize=(8,6))
        plt.title(f"{input_file} breath track")

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