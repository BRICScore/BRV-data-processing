import os, sys
import numpy as np
import json
sys.path.append("feature_processing")
sys.path.append("utils")

from feature_extraction import *
from config import ADC_COUNT, ACCEPTABLE_DATA_LOSS
from data_containers import *

RESULTS_PATH = './results'


def parse_results_line(line):
    feature_vector = []
    for key in line:
        if isinstance(line[key], list):
            for val in line[key]:
                feature_vector.append(val)
        else:
            feature_vector.append(line[key])
    return feature_vector

def load_measurement_data(measurement_data): # create dummy data before pipeline is integrated TODO
    if measurement_data:
        return measurement_data
    else:
        bd = BioData(person_id="198111", age=20, gender="female", health="healthy", condition="regular",
                     weight=69, height=165)
        labels = MeasurementLabels(activity="sit", person_data=bd)
        metadata = MeasurementMetadata(_id="1234567890", timestamp=2137420, duration_ms=1_800_000,
                                       filepath_raw="./bruh", filepath_clean="./essa", filepath_features="./frfr",
                                       labels=labels)
        clean = BRVDataClean(timestamps=None, adc_data=None)
        measure_data = MeasurementData(metadata=metadata, data_clean=clean, data_features=None)
        return measure_data


def extract_features(measurement_data=None):
    """

    """

    file = None
    with open("features/extracted_features.jsonl", "w"):
        pass
    with os.scandir(RESULTS_PATH) as es:
        for e in es:
            features = []

            measure_data = load_measurement_data(measurement_data=measurement_data)
            #measure_data.data_clean = BRVDataClean()
            # measure_data.data_features = BRVDataFeatures()
            # measure_data.metadata = MeasurementMetadata()
            print(e.name)
            if e.is_file() and e.name.endswith('.jsonl'):
                feature_vector = []
                with open(e.path, encoding='utf-8') as f:
                    file = f.read().split("\n")

                    if len(file) < (SEGMENT_LENGTH_MS / 100) * (1.0 - ACCEPTABLE_DATA_LOSS):
                        continue
                    for f_line in file:
                        if f_line != '': # last newline produces empty string
                            feature_vector = parse_results_line(json.loads(f_line))
                            features.append(feature_vector)
                    print(e.name)
                NPFeatures = np.array(features)
                measure_data.data_clean.adc_data = np.transpose(NPFeatures[:, 1:])
                measure_data.data_clean.timestamps = np.transpose(NPFeatures[:, 0])
                basic_feature_extraction(measure_data=measure_data, input_file=e.name)

# to comment out when pipeline comes TODO
extract_features()