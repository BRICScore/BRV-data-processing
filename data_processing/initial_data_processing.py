from breath_separation import *
from outlier_detection import *

from ADC_data import ADCdata
from config import *

def u2_to_i(b1, b2, b3):
    """
        Convert three 8-bit values into a single signed 24-bit integer.

        Parameters
        ----------
        b1: int
            The first byte of the data
        b2: int
            The second byte of the data
        b3: int
            The third byte of the data
        
        Returns
        -------
        int
            The resulting signed 24-bit integer value.
        
        Side Effects
        ------------
        This function has no side effects.
    """

    value = (b1 << 16) | (b2 << 8) | b3
    return value - (1 << 24) if value & (1 << 23) else value

def adc_to_voltage(adc_value):
    """
        Convert a 24-bit signed integer from ADC to voltage according to the BRV tensometers specification.

        Parameters
        ----------
        adc_value: int
            The 24-bit signed integer value from the ADC.
        
        Returns
        -------
        float
            The resulting voltage in volts.
        
        Side Effects
        ------------
        This function has no side effects.
    """

    return (adc_value + 2**23) * 10**(-9) * 23.84

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

    adc_outputs = [extract_adc_data(4 + i * 3) for i in range(ADC_COUNT)]
    return ms_timestamp, adc_outputs

def handle_input_data(input_file, adc_data):
    """
        Read raw ADC data from input file, parse it line by line by calling parse_adc_data_line function
        and add a timestamp to the data. Then store the data in adc_data.adc_output_data and adc_data.adc_normalized_data
        for further processing.

        Parameters
        ----------
        input_file: str
            The name of the raw input file containing the ADC data
        adc_data: ADCdata
            An instance of the ADCdata class where the parsed ADC data and timestamps will be stored.

        Returns
        -------
        None

        Side Effects
        ------------
        This function modifies the adc_data object by writing data into the adc_data.adc_output_data and 
        adc_data.adc_normalized_data attributes and writing the timestamps into adc_data.timestamps attribute.
    """
    
    first_timestamp = None
    with open(f"./data/{input_file}", 'r') as i_f:
        for line in i_f:
            ms_timestamp, adc_outputs = parse_adc_data_line(line)
            if first_timestamp is None:
                first_timestamp = ms_timestamp
            adc_data.timestamps = np.append(adc_data.timestamps, ms_timestamp - first_timestamp)
            for i, v in enumerate(adc_outputs):
                adc_data.adc_output_data[i] = np.append(adc_data.adc_output_data[i], v)
                adc_data.adc_normalized_data[i] = np.append(adc_data.adc_normalized_data[i], round(adc_to_voltage(v), 10))

def split_data_into_segments(input_file, adc_data):
    """
        Split the resampled ADC data into segments that contain values from a specific time window,
        and save each segment into a separate JSONL file.

        Parameters
        ----------
        input_file: str
            The name of the raw input file containing the ADC data, the name consists of labels
            that are used to create the output file names.
        adc_data: ADCdata
            An instance of the ADCdata class containing the resampled ADC data and timestamps.

        Returns
        -------
        None          
        
        Side Effects
        ------------
        This function creates multiple JSONL files in the "./results" directory, each containing a segment of the ADC data.
    """

    segment_index = 0
    total_segments = int(np.ceil(adc_data.final_adc_timestamps[-1] / SEGMENT_LENGTH_MS))
    filename = input_file.split("_")
    time = filename[1]
    person = filename[2]
    condition = filename[3]
    no_of_sample = filename[4]

    for segment_index in range(total_segments):
        segment_start = segment_index * SEGMENT_LENGTH_MS
        segment_end = segment_start + SEGMENT_LENGTH_MS
        with open(f"./results/clean_{time}_{segment_index}_{person}_{condition}_{no_of_sample.split(".")[0]}.jsonl", 'w') as o_f:
            for i in range(len(adc_data.final_adc_timestamps)):
                if segment_start <= adc_data.final_adc_timestamps[i] < segment_end:
                    record = {
                        "timestamp": int(adc_data.final_adc_timestamps[i]),
                        "adc_outputs": [adc_data.final_adc_data[a][i] for a in range(ADC_COUNT)]
                    }
                    o_f.write(json.dumps(record) + "\n")

def process_file(parser):
    """
        This function serves as the entry point for processing the raw ADC data files.
        It organizes and calls the necessary functions to read, parse, normalize, separate breaths,
        detect outliers, and split the data into segments.

        Parameters
        ----------
        parser: argparse.ArgumentParser
            An instance of ArgumentParser that contains the command-line arguments for input file and plot option.

        Returns
        -------
        None           
        
        Side Effects
        ------------
        This function modifies the adc_data object by writing data into adc_data.adc_normalized_data and 
        adc_data.adc_voltafe_means attributes.
    """

    args = parser.parse_args()
    input_file = args.input_file
    adc_data = ADCdata()

    adc_data.plot_enabled = args.plot
    adc_data.debug_plot_enabled = args.debugplot

    handle_input_data(input_file, adc_data)

    for i in range(ADC_COUNT):
            mean_voltage = np.mean(adc_data.adc_normalized_data[i])
            adc_data.adc_voltage_means.append(round(mean_voltage, 10))
            adc_data.adc_normalized_data[i] -= adc_data.adc_voltage_means[i]

    breath_separation(adc_data=adc_data, target_adc=TARGET_ADC) # from breath_separation.py
    outlier_detection(adc_data=adc_data, target_adc=TARGET_ADC) # from outlier_detection.py
    split_data_into_segments(input_file, adc_data)