from brics_types import MeasurementType
from data_containers import MeasurementData, MeasurementMetadata
from .FileProcessingFunctionProvider import FileProcessingFunctionProvider
from pathlib import Path
from typing import Any, TextIO
import json
from config import MEASUREMENT_ZIP_PATH

class MeasurementDataBuilder:

    def __init__(self, measurement_data_container: MeasurementData):
        self.data = measurement_data_container
        self.provider = FileProcessingFunctionProvider()

    def build_data(self, filepath: Path, target: MeasurementType) -> None:
        func = self.provider.provide_function(target)
        with open(filepath, "r") as filehook:
            if self.__check_for_metadata():
                self.__consume_metadata()
                self.__correct_paths()
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
    
    def __convert_metadata(self, jsondict: dict[str, Any]) -> None:
        self.data.metadata = MeasurementMetadata(**jsondict)
        return
    
    def __correct_paths(self, filehook: TextIO) -> None:
        self.data.metadata.filepath_raw = None
        self.data.metadata.filepath_clean = None
        self.data.metadata.filepath_features = None
        if ("raw" in filehook.name):
            self.data.metadata.filepath_raw = Path(filehook.name)
        elif ("clean" in filehook.name):
            self.data.metadata.filepath_clean = Path(filehook.name)
        elif ("feature" in filehook.name):
            self.data.metadata.filepath_features = Path(filehook.name)
        return