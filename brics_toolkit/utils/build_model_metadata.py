from pathlib import Path
from ..data_containers import ModelMetadata
import time


def build_model_metadata(filepaths: list[str] = [], id: str = None) -> ModelMetadata:
    """
            build model metadata for uploads and internal processing.

            Parameters
            ----------
            filepaths : list[str]
                Optional argument, for assigning filepaths automatically as opposed to manual input
            
            Returns
            -------
            ModelMetadata object created.
            
            Side Effects
            ------------
            None 
    """
    metadata = ModelMetadata()

    if id:
        metadata.id = id

    print("MeasurementMetadata parameters for creation")

    if not filepaths:
        metadata.filepath_weights = Path(input("filepath_weights: "))
        metadata.filepath_pth = Path(input("filepath_pth: "))
        metadata.filepath_scaler = Path(input("filepath_scaler: "))
    else:
        filepath_clean = Path(filepaths[1])
        metadata.filepath_weights = Path(filepaths[0])
        metadata.filepath_pth = filepath_clean
        metadata.filepath_scaler = Path(filepaths[2])

    metadata.timestamp = time.time()

    return metadata
        

    