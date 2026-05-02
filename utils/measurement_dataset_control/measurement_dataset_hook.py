from pathlib import Path
from typing import Literal
import os
from . import MeasurementDirectoryProvider
from brics_types import MeasurementType
class MeasurementDatasetHook:
    """
            Class providing an iterable interface for accessing files inside a measurement zip file of a chosen type.

            Attributes
            ----------
            target : MeasurementType
                Specifies the type of files to be accessed.

    """
    
    def __init__(self, target: MeasurementType):
        
        self.provider = MeasurementDirectoryProvider()
        self.folder_path = self.provider.provide_directory(target)
        self.files = [self.folder_path / Path(name) for name in os.listdir(self.folder_path)]
        self.number_of_files = len(self.files)  
        self.index = -1
            
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index == self.number_of_files - 1:
            raise StopIteration
        self.index = self.index + 1
        return self.files[self.index]
        
