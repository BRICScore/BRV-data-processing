import argparse, sys

sys.path.append("data_processing")
sys.path.append("feature_processing")
sys.path.append("utils")
sys.path.append("brics_types")
sys.path.append("data_containers")

from ..data_processing.outlier_detection import *

from ..data_containers import BRVDataIntermediate
from ..data_containers import MeasurementData
from ..utils.config import *



def breath_separation(BRV_data_intermediate : BRVDataIntermediate, target_adc : int, plot_enabled : bool=False):
    """ 
        Logically separate breaths in the normalized ADC signal by finding local maxima and minima and storing the data
        in the ADCdata object. A breath is defined as the signal between two consecutive minima. 

        Parameters
        ----------
        BRV_data_intermediate : BRVDataIntermediate
            The BRVDataIntermediate object containing the normalized ADC data and timestamps.
        target_adc : int
            The index of the ADC to analyze for outliers.
        plot_enabled : bool
            A flag that turns plotting on and off
        
        Returns
        -------
        none
        
        Side Effects
        ------------
        This function may visualize the split breahts by coloring each breath differently if the plot_enabled flag is set to true
        and sets the signal_maxima and signal_minima attributes of the BRV_data_intermediate object.
    """

    def find_signal_extrema(adc_normalized_data, target_adc, invert=False):
        signal = adc_normalized_data[target_adc]
        if invert:
            signal = [-s for s in signal]
        std_dev_signal = np.std(signal)
        mean_signal = np.mean(signal)

        maxima, _ = scipy.signal.find_peaks(signal, distance=MIN_DISTANCE, height=mean_signal + std_dev_signal*STD_DEV_CONST)
        return maxima

    signal_maxima = find_signal_extrema(adc_normalized_data=BRV_data_intermediate.adc_normalized_data, target_adc=target_adc, invert=False)
    signal_minima = find_signal_extrema(adc_normalized_data=BRV_data_intermediate.adc_normalized_data, target_adc=target_adc, invert=True)

    BRV_data_intermediate.signal_maxima = signal_maxima
    BRV_data_intermediate.signal_minima = signal_minima

    # Plots each breath with different color
    if plot_enabled:
        breaths = []
        for i in range(len(BRV_data_intermediate.signal_minima) - 1):
            breath = BRV_data_intermediate.adc_normalized_data[target_adc][BRV_data_intermediate.signal_minima[i]:BRV_data_intermediate.signal_minima[i + 1]]
            breaths.append(breath)

        plt.figure(figsize=(12, 6))
        for i, breath in enumerate(breaths):
            plt.plot(BRV_data_intermediate.timestamps[BRV_data_intermediate.signal_minima[i]:BRV_data_intermediate.signal_minima[i + 1]], breath)

        plt.title("Separated Breaths")
        plt.xlabel("Time (samples)")
        plt.ylabel("Normalized ADC Value")
        plt.show()

def parse_adc_data_line(line: str):
    """
        Parse a single line of ADC data from the input file, extracting the timestamp and ADC output values.

        Parameters
        ----------
        line: str
            A single line of text from the input file containing a singular json record of the ADC data.

        Returns
        -------
        tuple: int, int[]
            A tuple containing the timestamp (int) and a list of ADC output values (list of int).
        
        Side Effects
        ------------
        This function has no side effects.
    """

    def u2_to_i(b1, b2, b3):
        value = (b1 << 16) | (b2 << 8) | b3
        return value - (1 << 24) if value & (1 << 23) else value

    # bit merging
    def extract_adc_data(start_index):
        return u2_to_i(int(parts[start_index].split(':')[1]),
                       int(parts[start_index + 1].split(':')[1]),
                       int(parts[start_index + 2].split(':')[1]))
    
    parts = line.strip().split(',')
    hour = int(parts[0].split(':')[1])
    minute = int(parts[1].split(':')[1])
    second = int(parts[2].split(':')[1])
    millisecond = int(parts[3].split(':')[1])
    ms_timestamp = (hour * 3600 + minute * 60 + second) * 1000 + millisecond

    adc_outputs = [extract_adc_data(4 + i * 3) for i in range(ADC_COUNT)]   # 4 - skip timestamp, i*3 - each ADC has 3 bytes of data 
    return ms_timestamp, adc_outputs

def handle_input_data(input_file : Path):
    """
        Read raw ADC data from input file, parse it line by line by calling parse_adc_data_line function
        and add a timestamp to the data. Then store the data in adc_data.adc_output_data and adc_data.adc_normalized_data
        for further processing.

        Parameters
        ----------
        input_file: Path
            The path to the raw input file containing the ADC data

        Returns
        -------
        tuple: np.ndarray, list[np.ndarray]
            A tuple containing the timestamps (np.ndarray) and a list of ADC output values (list of np.ndarray).

        Side Effects
        ------------
        This function modifies the adc_data object by writing data into the adc_data.adc_output_data and 
        adc_data.adc_normalized_data attributes and writing the timestamps into adc_data.timestamps attribute.
    """

    def adc_to_voltage(adc_value):
        return (adc_value + 2**23) * 10**(-9) * 23.84
    
    first_timestamp = None
    timestamps = np.array([])
    adc_output_data = [np.array([]) for _ in range(ADC_COUNT)]
    with open(f"data/{input_file}", 'r') as i_f:
        for line in i_f:
            ms_timestamp, adc_outputs = parse_adc_data_line(line)
            if first_timestamp is None:
                first_timestamp = ms_timestamp
            timestamps = np.append(timestamps, ms_timestamp - first_timestamp)
            for i, v in enumerate(adc_outputs):
                adc_output_data[i] = np.append(adc_output_data[i], round(adc_to_voltage(v), 10))
    return timestamps, adc_output_data

def process_raw_file(input_file: Path, plot_enabled: bool = False):
    """
        This function serves as the entry point for processing the raw ADC data files.
        It organizes and calls the necessary functions to read, parse, normalize, separate breaths,
        detect outliers, and split the data into segments.

        Parameters
        ----------
        input_file: Path
            The path to the raw input file containing the ADC data
        plot_enabled : bool
            A flag that turns plotting on and off

        Returns
        -------
        BRV_data_intermediate : BRVDataIntermediate
            The BRVDataIntermediate object containing the normalized ADC data and timestamps further defined in project's DTP.
        
        Side Effects
        ------------
        This function modifies the BRV_data_intermediate object by writing data into its
        timestamps and adc_normalized_data attributes.
    """

    BRV_data_intermediate = BRVDataIntermediate()
    timestamps, adc_normalized_data = handle_input_data(input_file)

    for i in range(ADC_COUNT):
            mean_voltage = np.mean(adc_normalized_data[i])
            adc_normalized_data[i] -= mean_voltage

    np_adc_normalized_data = np.vstack(
        adc_normalized_data
    )

    BRV_data_intermediate.timestamps = timestamps
    BRV_data_intermediate.adc_normalized_data = np_adc_normalized_data
    breath_separation(BRV_data_intermediate, TARGET_ADC, plot_enabled=plot_enabled)
    return BRV_data_intermediate

def initial_data_processing(BRV_measurement_data: MeasurementData, target_adc: int = TARGET_ADC, plot_enabled: bool = False):
    input_file = BRV_measurement_data.metadata.filepath_raw

    BRV_data_intermediate = process_raw_file(input_file, plot_enabled=plot_enabled)
    BRV_measurement_data.data_clean = outlier_detection(BRV_data_intermediate, target_adc=target_adc, plot_enabled=plot_enabled)
    # split_data_into_segments(input_file, BRV_measurement_data.data_clean)