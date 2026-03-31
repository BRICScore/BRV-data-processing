from pathlib import Path
from typing import Literal
import os
from config import MEASUREMENT_ZIP_PATH
import MeasurementDirectoryProvider as mdp

class MeasurementDatasetHook:
    
    def __init__(self, target: mdp.MeasurementType):
        try:
            self.provider = mdp.MeasurementDirectoryProvider()
            self.folder_path = self.provider.provide_directory(target)
            self.number_of_files = len([name for name in os.listdir(self.folder_path) if os.path.isfile(name)])  
            self.index = 0
            
        except Exception:
            raise Exception("Missing a measurement zip directory or target invalid")
        
    def __iter__(self):
        return self
    
    def __next__(self):
        #TODO: Implement iterating through files
        return
        
