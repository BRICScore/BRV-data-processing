import json
from pathlib import Path
from typing import Any
from data_containers import MeasurementMetadata
from brics_types import allowed_activity_types, allowed_gender_types, allowed_condition_types
import time


def input_measurement_metadata(filepaths: list[str] = []) -> MeasurementMetadata:
    """
            Input measurement metadata for uploads and internal processing.

            Parameters
            ----------
            filepaths : list[str]
                Optional argument, for assigning filepaths automatically as opposed to manual input
            
            Returns
            -------
            MeasurementMetadata object created.
            
            Side Effects
            ------------
            None 
    """
    metadata = MeasurementMetadata()

    print("MeasurementMetadata parameters for creation")
    timestamp = time.time()

    if not filepaths:
        metadata.filepath_raw = Path(input("filepath_raw: "))
        metadata.filepath_clean = Path(input("filepath_clean: "))
        metadata.filepath_features = Path(input("filepath_features: "))
        filepath_clean = metadata.filepath_clean
    else:
        filepath_clean = Path(filepaths[1])
        metadata.filepath_raw = Path(filepaths[0])
        metadata.filepath_clean = filepath_clean
        metadata.filepath_features = Path(filepaths[2])



    with open(filepath_clean, "rb") as workMeasurementFileHook:
            jsonlines = workMeasurementFileHook.readlines()
            jsonlines = jsonlines[::-1]
            parsed_line = json.loads(jsonlines[0])

    metadata.duration_ms = parsed_line["timestamp"]
    metadata.timestamp = time.time()

    activity: str = input("activity: ")
    if activity in allowed_activity_types:
        metadata.labels.activity = activity #type: ignore
    else:
        raise ValueError(f"invalid activity type: {activity}")

    metadata.labels.person_data.person_id = input("person_id: ")
    metadata.labels.person_data.age = int(input("age: "))

    gender: str = input("gender: ")
    if gender in allowed_gender_types:
        metadata.labels.person_data.gender = gender #type: ignore
    else:
        raise ValueError(f"invalid gender type: {gender}")
    
    metadata.labels.person_data.health = input("health: ")
    
    condition: str = input("condition: ")
    if condition in allowed_condition_types:
        metadata.labels.person_data.condition = condition #type: ignore
    else:
        raise ValueError(f"invalid condition type: {condition}")
    
    metadata.labels.person_data.weight = int(input("weight:"))
    metadata.labels.person_data.height = int(input("height:"))

    return metadata
        

    