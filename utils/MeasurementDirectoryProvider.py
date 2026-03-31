from typing import Literal
from pathlib import Path
import tempfile
import shutil
from config import MEASUREMENT_ZIP_PATH
from brics_types.MeasurementType import MeasurementType


class MeasurementDirectoryProvider:

    def __init__(self, target: MeasurementType):
        self.target = target
        self.folder_path = self._unpack_zip()
        return

    def provide_directory(self, target: MeasurementType) -> Path:
        resolved_folder_path = self._resolve_folder_path(target)
        return resolved_folder_path

    def _unpack_zip(self) -> Path:
        folder_path = Path(tempfile.mkdtemp())
        shutil.unpack_archive(MEASUREMENT_ZIP_PATH, folder_path)
        return Path
    
    def _resolve_folder_path(self) -> Path:
        resolved_folder_path = self.folder_path / "measurement_dataset" / self.target
        return resolved_folder_path

    