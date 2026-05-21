import os, sys
import numpy as np
import json
sys.path.append("feature_processing")
sys.path.append("utils")

from .feature_extraction import *
from ..utils.config import *
from ..data_containers import MeasurementMetadata
from ..data_containers import MeasurementData
from pathlib import Path

RESULTS_PATH = './results'


def parse_results_line(line):
    """Used to convert a json-coded string into a usable list vector

    Parameters
    ----------
    line : string(json-shaped)

    Returns
    -------
        feature_vector: list with all entries stripped off json keys

    Side Effects
    ------------
        This function has no side effects.
    """
    feature_vector = []
    for key in line:
        if isinstance(line[key], list):
            for val in line[key]:
                feature_vector.append(val)
        else:
            feature_vector.append(line[key])
    return feature_vector

def load_measurement_data(measurement_data): # create dummy data before pipeline is integrated TODO
    """
    This function is used as a placeholder for filling measurement data
    if, for example, you want to test the feature extraction function without calling
    the entire pipeline

    Parameters
    ----------
    measurement_data : BRVMeasurementData | None
        this argument is None in testing environment

    Returns
    -------
    measurement_data : BRVMeasurementData
        it is always filled either with existing data or dummy data.

    Side Effects
    ------------
        This function loads dummy data if no measurement data was provided.
    """
    if measurement_data:
        return measurement_data
    else:
        bd = BioData(person_id="111111", age=67, gender="female", health="healthy", condition="regular",
                     weight=69, height=165)
        labels = MeasurementLabels(activity="sit", person_data=bd)
        metadata = MeasurementMetadata()
        metadata._id = "1234567890"
        metadata.timestamp = 2137420
        metadata.duration_ms = 1_800_000
        metadata.filepath_raw = Path("./bruh")
        metadata.filepath_clean = Path("./essa")
        metadata.filepath_features = Path("./frfr")
        metadata.labels = labels
        # metadata = MeasurementMetadata(_id="1234567890", timestamp=2137420, duration_ms=1_800_000,
        #                                filepath_raw=Path("./bruh"), filepath_clean=Path("./essa"), filepath_features=Path("./frfr"),
        #                                labels=labels)
        clean = BRVDataClean()
        measure_data = MeasurementData()
        measure_data.metadata = metadata
        measure_data.data_clean = clean
        measure_data.data_features = BRVDataFeatures()
        # measure_data = MeasurementData(metadata=metadata, data_clean=clean, data_features=BRVDataFeatures())
        return measure_data


def extract_features(measurement_data=None):
    """
    This function is responsible for delivering BRVDataFeatures to
    its parent class MeasurementData **<-TODO** by loading the contents of
    segments and calling feature extraction on them.

    It is meant to be callable as a step of the final pipeline.

    Parameters
    ----------
    measurement_data : BRVMeasurementData | None
        this argument is None in testing environment

    Returns
    -------
        None

    Side Effects
    ------------
        This function stores appropriate data in measurement_data.BRVDataFeatures TODO
        and in ***extracted_features.jsonl*** file.
    """

    file = None
    measure_data = load_measurement_data(measurement_data=measurement_data)
    features_filename = str(measure_data.metadata.filepath_clean)
    temp = features_filename.split("\\")[-1]
    with open(f"./features/{"features_"+temp}", "w"):
        pass

    with os.scandir(RESULTS_PATH) as es:
        for e in es:
            features = []
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
                NPFeatures = np.array(features)
                measure_data.data_clean.adc_data = np.transpose(NPFeatures[:, 1:])
                measure_data.data_clean.timestamps = np.transpose(NPFeatures[:, 0])
                basic_feature_extraction(measure_data=measure_data, input_file=e.name)

# extract_features()