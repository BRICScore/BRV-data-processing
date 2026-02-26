from breath_separation import *
# from feature_extraction import *
from feature_extraction import basic_feature_extraction
from outlier_detection import *

from ADC_data import ADCdata
from config import *

def u2_to_i(value, b1, b2, b3):
    value = (b1 << 16) | (b2 << 8) | b3
    return value - (1 << 24) if value & (1 << 23) else value

def adc_to_voltage(adc_value):
    return (adc_value + 2**23) * 10**(-9) * 23.84

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
    total_segments = int(np.ceil(adc_data.final_adc_timestamps[-1] / SEGMENT_LENGTH_MS))
    filename = input_file.split("_")
    time = filename[1]
    person = filename[2]
    condition = filename[3]
    no_of_sample = filename[4]
    for segment_index in range(total_segments):
        segment_start = segment_index * SEGMENT_LENGTH_MS
        segment_end = segment_start + SEGMENT_LENGTH_MS
        segment_fill_percentage = 0
        with open(f"./results/clean_{time}_{segment_index}_{person}_{condition}_{no_of_sample.split(".")[0]}.jsonl", 'w') as o_f:
            for i in range(len(adc_data.final_adc_timestamps)):
                if segment_start <= adc_data.final_adc_timestamps[i] < segment_end:
                    record = {
                        "timestamp": int(adc_data.final_adc_timestamps[i]),
                        "adc_outputs": [adc_data.final_adc_data[a][i] for a in range(ADC_COUNT)]
                    }
                    o_f.write(json.dumps(record) + "\n")
                    segment_fill_percentage += 1
        segment_fill_percentage = (segment_fill_percentage*100 / SEGMENT_LENGTH_MS) * 100
        print(f"Segment {segment_index}: {segment_fill_percentage:.2f}% filled")

def process_file(parser):
    args = parser.parse_args()
    input_file = args.input_file
    adc_data = ADCdata()

    adc_data.plot_enabled = args.plot

    handle_input_data(input_file, adc_data)

    for i in range(ADC_COUNT):
            mean_voltage = np.mean(adc_data.adc_normalized_data[i])
            adc_data.adc_voltage_means.append(round(mean_voltage, 10))
            adc_data.adc_normalized_data[i] -= adc_data.adc_voltage_means[i]

    # handle_results_data(input_file, adc_data)
    breath_separation(adc_data=adc_data, target_adc=TARGET_ADC) # from breath_separation.py
    outlier_detection(adc_data=adc_data, target_adc=TARGET_ADC) # from outlier_detection.py
    # split_data_into_segments(input_file, adc_data)
    # basic_feature_extraction(adc_data, input_file)              # from feature_extraction.py