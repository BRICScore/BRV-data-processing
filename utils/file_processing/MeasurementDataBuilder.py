from brics_types import MeasurementType
from data_containers import MeasurementData, MeasurementMetadata
from .FileProcessingFunctionProvider import FileProcessingFunctionProvider
from pathlib import Path
from typing import TextIO
import json

class MeasurementDataBuilder:

    def __init__(self, measurement_data_container: MeasurementData):
        self.data = measurement_data_container
        self.provider = FileProcessingFunctionProvider()

    def build_data(self, filepath: Path, target: MeasurementType) -> None:
        func = self.provider.provide_function(target)
        with open(filepath, "r") as filehook:
            if self.__check_for_metadata():
                self.__consume_metadata()
            func(self.data, filehook)
            
        return
    
    def __check_for_metadata(filehook: TextIO) -> bool:
        line = filehook.readline()
        jsonline = json.loads(line)
        filehook.seek(0)
        return "_id" in jsonline

    def __consume_metadata(self, filehook: TextIO) -> None:
        line = filehook.readline()
        jsonline = json.loads(line)
        self.__convert_metadata(jsonline)
        return 
    
    def __convert_metadata(self, jsondict: dict) -> None:
        return