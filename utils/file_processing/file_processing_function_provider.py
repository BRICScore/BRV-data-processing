from brics_types import MeasurementType
from data_containers import MeasurementData
from typing import Callable, TextIO
import json
from numpy import array as nparray

class FileProcessingFunctionProvider:

    @staticmethod
    def __processing_raw(measurement_data: MeasurementData, filehook: TextIO) -> None:
        # does nothing, is there for future proofing
        return
    
    @staticmethod
    def __processing_clean(measurement_data: MeasurementData, filehook: TextIO) -> None:
        rows_timestamps = [] # could preallocate but its guessing
        rows_adc_data = []
        line = filehook.readline()
        while line:
            json_line = json.loads(line) # should be a json line fitting the BRVDataClean format
            rows_timestamps.append(json_line["timestamp"])
            rows_adc_data.append(json_line["adc_outputs"])
            line = filehook.readline()
            
        measurement_data.data_clean.timestamps = nparray(rows_timestamps)
        measurement_data.data_clean.adc_data = nparray(rows_adc_data)          
            
        return
    
    @staticmethod
    def __processing_feature(measurement_data: MeasurementData, filehook: TextIO) -> None:
        field_map = measurement_data.data_features.__dict__
        for key in field_map.keys():
            field_map[key] = []
        line = filehook.readline()
        while line:
            json_line = json.loads(line) # should be a json line fitting the BRVDataFeatures format
            for key in field_map.keys():
                field_map[key].append(json_line[key])
            line = filehook.readline()

        for key, val in field_map.items():
            setattr(measurement_data.data_features, key, nparray(val))
        
        return

    processing_map: dict [str, Callable[[MeasurementData, TextIO], None]] = {"raw" : __processing_raw, "clean": __processing_clean, "features": __processing_feature}
    
    @staticmethod
    def provide_function(target: MeasurementType) -> Callable[[MeasurementData, TextIO], None]:
        return FileProcessingFunctionProvider.processing_map[target]

        



        
        