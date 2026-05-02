from brics_types import MeasurementType
from data_containers import MeasurementData, MeasurementMetadata
from pathlib import Path
from typing import Any, TextIO
import json
from utils.config import MEASUREMENT_ZIP_PATH
from . import FileProcessingFunctionProvider

class MeasurementDataBuilder:
    """
        Class for building data of the MeasurementData object from files.

        Attributes
        ----------
        measurement_data_container: MeasurementDataAttributes
            An object to be filled with data - previously existing data will be replaced.

        Methods
        ----------
        build_data(filepath: Path, target: MeasurementType):
            Returns nothing, fills provided MeasurementDate's fields with data from the specifed file according to the target type.

    """

    def __init__(self, measurement_data_container: MeasurementData):
        self.data = measurement_data_container

    def build_data(self, filepath: Path, target: MeasurementType) -> None:
        """
            Fills the provided MeasurementDate's fields with data from the specifed file according to the target type.

            Arguments
            ----------
            filepath: Path
                The path to the file containing the data.
            target: MeasurementType
                The type of the measurement contained in the target file.

            Returns
            ----------
                Nothing
        """
        func = FileProcessingFunctionProvider.provide_function(target)
        with open(filepath, "r") as filehook:
            if self.__check_for_metadata(filehook):
                self.__consume_metadata(filehook)
                self.__correct_paths(filehook)
            func(self.data, filehook)
            
        return
    
    def __check_for_metadata(self, filehook: TextIO) -> bool:
        """
            Checks if metadata is present in the file provided.

            Arguments
            ----------
            filehook: TextIO
                Hook to the opened file.

            Returns
            ----------
                True if metadata is present, False if not.
        """
        line = filehook.readline()
        jsonline = json.loads(line)
        filehook.seek(0)
        return "filepath_raw" in jsonline

    def __consume_metadata(self, filehook: TextIO) -> None:
        """
            Consumes metadata present in the file provided.

            Arguments
            ----------
            filehook: TextIO
                Hook to the opened file.

            Returns
            ----------
                True if metadata is present, False if not.
        """
        line = filehook.readline()
        jsonline = json.loads(line)
        self.__convert_metadata(jsonline)
        return 
    
    def __convert_metadata(self, jsondict: dict[str, Any]) -> None:
        """
            Converts metadata into MeasurementMetadata object.

            Arguments
            ----------
            jsondict: dict[str, Any]
                Dictionary to be unpacked into a MeasurementMetadata object.

            Returns
            ----------
                Nothing
        """
        self.data.metadata = MeasurementMetadata(**jsondict)
        return
    
    def __correct_paths(self, filehook: TextIO) -> None:
        """
            Corrects paths in the MeasurementMetadata by using info from filehook.

            Arguments
            ----------
            filehook: TextIO
                Hook to the opened file.

            Returns
            ----------
                Nothing
        """
        if ("raw" in filehook.name):
            self.data.metadata.filepath_raw = Path(filehook.name)
        elif ("clean" in filehook.name):
            self.data.metadata.filepath_clean = Path(filehook.name)
        elif ("feature" in filehook.name):
            self.data.metadata.filepath_features = Path(filehook.name)
        return