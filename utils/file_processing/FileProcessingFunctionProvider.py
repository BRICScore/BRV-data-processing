from brics_types import MeasurementType
from data_containers import MeasurementData
from typing import Callable, TextIO

class FileProcessingFunctionProvider:

    def __processing_raw(measurement_data: MeasurementData, filehook: TextIO) -> None:
        #TODO: Copy processing from process_file()
        return
    
    def __processing_clean(measurement_data: MeasurementData, filehook: TextIO) -> None:
        #TODO: Implement a processing for clean files
        return
    
    def __processing_feature(measurement_data: MeasurementData, filehook: TextIO) -> None:
        #TODO: Implement a processing for feature files
        return

    def __init__(self):
        self.processing_map: dict [str, Callable[[MeasurementData, TextIO], None]] = {"raw" : self.__processing_raw, "clean": self.__processing_clean, "feature": self.__processing_feature}
    
    def provide_function(self,target: MeasurementType) -> Callable[[MeasurementData, TextIO], None]:
        return self.processing_map[target]

        



        
        