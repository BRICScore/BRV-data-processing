from brics_types import MeasurementType
from data_containers import MeasurementData
from typing import Callable, TextIO
import json
from numpy import array

class FileProcessingFunctionProvider:

    def __processing_raw(measurement_data: MeasurementData, filehook: TextIO) -> None:
        # does nothing, is there for future proofing
        return
    
    def __processing_clean(measurement_data: MeasurementData, filehook: TextIO) -> None:
        rows_timestamps = [] # could preallocate but its guessing
        rows_adc_data = []
        line = filehook.readline()
        while line:
            json_line = json.loads(line) # should be a json line fitting the BRVDataClean format
            rows_timestamps.append(json_line["timestamps"])
            rows_adc_data.append(json_line["adc_data"])
            line = filehook.readline()
            
        measurement_data.data_clean.timestamps = array(measurement_data.data_clean.timestamps, rows_timestamps, axis=0)
        measurement_data.data_clean.adc_data = array(measurement_data.data_clean.timestamps, rows_adc_data, axis=0)              
            
        return
    
    def __processing_feature(measurement_data: MeasurementData, filehook: TextIO) -> None:
        #TODO: Implement a processing for feature files
        return

    def __init__(self):
        self.processing_map: dict [str, Callable[[MeasurementData, TextIO], None]] = {"raw" : self.__processing_raw, "clean": self.__processing_clean, "features": self.__processing_feature}
    
    def provide_function(self,target: MeasurementType) -> Callable[[MeasurementData, TextIO], None]:
        return self.processing_map[target]

        



        
        