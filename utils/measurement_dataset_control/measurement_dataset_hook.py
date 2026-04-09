from pathlib import Path
from typing import Literal
import os
from config import MEASUREMENT_ZIP_PATH
import measurement_directory_provider as mdp

class MeasurementDatasetHook:
    """
            Class providing an iterable interface for accessing files inside a measurement zip file of a chosen type.

            Attributes
            ----------
            target : MeasurementType
                Specifies the type of files to be accessed.

    """
    
    def __init__(self, target: mdp.MeasurementType):
        try:
            self.provider = mdp.MeasurementDirectoryProvider()
            self.folder_path = self.provider.provide_directory(target)
            self.files = [name for name in os.listdir(self.folder_path) if os.path.isfile(name)]
            self.number_of_files = len(self.files)  
            self.index = -1
            
        except Exception:
            raise Exception("Missing a measurement zip directory or target invalid")
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index == self.number_of_files - 1:
            raise StopIteration
        self.index = self.index + 1
        return self.files[self.index]
        
