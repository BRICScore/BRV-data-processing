from typing import Literal
from pathlib import Path
import tempfile
import shutil
from utils.config import MEASUREMENT_ZIP_PATH
from brics_types import MeasurementType


class MeasurementDirectoryProvider:
    """
            Class for providing the correct measurement zip file directory. Requires a measurement zip archive to exist.

            Methods
            ----------
            provide_directory(target: MeasurementType)
                Returns a Path to the extracted, temporary directory containing the files of the chosen type.

    """

    def __init__(self):
        self.folder_path = self.__unpack_zip()
        return

    def provide_directory(self, target: MeasurementType) -> Path:
        """
            Provides a Path to the temporary directory containing the files matching the chosen type.

            Arguments
            ----------
            target: MeasurementType
                The type of files to be contained in the returned directory.

            Returns
            ----------
            A Path to the temporary directory. 
        """
        resolved_folder_path = self.__resolve_folder_path(target)
        return resolved_folder_path

    def __unpack_zip(self) -> Path:
        """
            Unpacks the measurement zip archive into a temporary directory.

            Returns
            ----------
            A Path to the temporary directory containing the entire archive. 
        """
        folder_path = Path(tempfile.mkdtemp())
        shutil.unpack_archive(MEASUREMENT_ZIP_PATH, folder_path)
        return folder_path
    
    def __resolve_folder_path(self, target: MeasurementType) -> Path:
        """
            Builds a path into the directory with the chosen type of files.

            Arguments
            ----------
            target: MeasurementType
                The type of files to be contained in the returned directory.

            Returns
            ----------
            A Path to the temporary directory containing the chosen type of files. 
        """
        resolved_folder_path = self.folder_path / target
        return resolved_folder_path

    def __del__(self):
        """
            Deconstructor for automatic deletion of the temporary directory.
        """
        shutil.rmtree(self.folder_path)

    