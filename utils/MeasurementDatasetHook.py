from pathlib import Path
import database_access.requests as db
import shutil
import tempfile
import os
from config import MEASUREMENT_ZIP_PATH

class MeasurementDatasetHook:
    
    def __init__(self):
        try:
            self.folder_path = Path(tempfile.mkdtemp())
            shutil.unpack_archive(MEASUREMENT_ZIP_PATH, self.folder_path)
            self.folder_path = self.folder_path / "measurements_dataset"
            self.number_of_files = len([name for name in os.listdir(self.folder_path) if os.path.isfile(name)])  
            self.index = 0
            
        except Exception:
            raise Exception("Missing a measurement zip directory")
        
    def __iter__(self):
        return self
    
    def __next__(self):
        #TODO: Implement iterating through files
        return
        
