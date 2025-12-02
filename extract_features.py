import os, sys
import numpy as np
import json
sys.path.append("feature_processing")
sys.path.append("utils")
from feature_extraction import *
from config import ADC_COUNT, ACCEPTABLE_DATA_LOSS
from ADC_data import ADCdata
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

file = None
with open("features/extracted_features.jsonl", "w"):
    pass
with os.scandir(RESULTS_PATH) as es:
    for e in es:
        features = []
        adc_data = ADCdata()
        if e.is_file() and e.name.endswith('.jsonl'):
            feature_vector = []
            with open(e.path, encoding='utf-8') as f:
                file = f.read().split("\n")
                #print("len:",len(file))
                #print("threshold:", (SEGMENT_LENGTH_MS / 100) * (1.0 - ACCEPTABLE_DATA_LOSS))
                if len(file) < (SEGMENT_LENGTH_MS / 100) * (1.0 - ACCEPTABLE_DATA_LOSS):
                    continue
                for f_line in file:
                    if f_line != '': # last newline produces empty string
                        feature_vector = parse_results_line(json.loads(f_line))
                        features.append(feature_vector)
                print(e.name)
            NPFeatures = np.array(features)
            adc_data.adc_normalized_data = np.transpose(NPFeatures[:, 1:])
            adc_data.timestamps = np.transpose(NPFeatures[:, 0])
            basic_feature_extraction(adc_data=adc_data, input_file=e.name)
