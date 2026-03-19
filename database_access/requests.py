import requests
import os
import json
import time
from pathlib import Path

def uploadMeasurement(filepath_raw: str, filepath_work: str):
    headers = {"CF-Access-Client-Id": os.getenv("ACCESS_CLIENT_ID"), "CF-Access-Client-Secret": os.getenv("ACCESS_CLIENT_SECRET")}
    form_data = {}
    #TODO: MOVE UI TO AN APP/WEBPAGE
    print("Fill measurement metadata:")
    form_data["person_id"] = str(input("Person ID (currently email): "))


    workMeasurementFileHook = open(filepath_work, "rb")
    jsonlines = workMeasurementFileHook.readlines()
    jsonlines = jsonlines[::-1]
    parsed_line = json.loads(jsonlines[0])
    form_data["duration_ms"] = parsed_line["timestamp"]
    form_data["timestamp"] = time.time()

    print("Fill in labels for the measurement:")

    labels = {}

    level = input("Level (raw/clean): ")
    labels["level"] = level

    activity = input("Activity: ")
    labels["activity"] = activity

    age = int(input("Age (from 0 to 100): "))
    gender = input("Biological gender (male/female): ")
    health = input("Health condition: ")
    condition = input("Athletic condition: ")

    labels["bio"] = {"age": age, "gender": gender, "health": health, "condition": condition}

    form_data["labels"] = json.dumps(labels)
    
    files = {"measurement_file_raw" : open(filepath_raw, "rb"), "measurement_file_clean": open(filepath_work, "rb")}

    r = requests.put('https://brics-api.electimore.xyz/measurement/upload', headers=headers, files=files, data=form_data)

    print(r.status_code)
    print(r.text)
    return
    
def downloadMeasurement():
    headers = {"CF-Access-Client-Id": os.getenv("ACCESS_CLIENT_ID"), "CF-Access-Client-Secret": os.getenv("ACCESS_CLIENT_SECRET")}
    #TODO: MOVE UI TO AN APP/WEBPAGE
    print("Fill measurement download query parameters")

    person_id = input("Person ID (currently email): ").strip() or None
    
    length_min = int(input("Minimum length in minutes: ").strip() or 0)
    length_max = int(input("Maximum length in minutes: ").strip() or 24*60) # default value - 1 day
    
    age_min = int(input("Minimum age of subject: ").strip() or 0)
    age_max = int(input("Maximum age of subject: ").strip() or 100)
    
    level = input("Enter acceptable levels (raw/clean) seperated by space or press enter for all").strip().split() or None
    gender = input("Enter genders (male/female) seperated by space or skip for all").strip().split() or None
    activity = input("Enter activities seperated by space or skip or all").strip().split() or None
    condition = input("Enter conditions seperated by space or skip or all").strip().split() or None
    health = input("Enter health statuses seperated by space or skip or all").strip().split() or None

    query_data = {
        "person_id": person_id,
        "length_min": length_min * 60000, #length kept in db in milliseconds
        "length_max": length_max * 60000,
        "age_min": age_min,
        "age_max": age_max,
        "level": level,
        "gender": gender,
        "activity": activity,
        "condition": condition,
        "health": health
    }    
    
    r = requests.get('https://brics-api.electimore.xyz/measurement/download', headers=headers, params=query_data)

    path = Path.home() / "Downloads" / "measurements_dataset.zip"
    if r.status_code == 200:
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print(r.status_code)
    return

def deleteMeasurement():
    headers = {"CF-Access-Client-Id": os.getenv("ACCESS_CLIENT_ID"), "CF-Access-Client-Secret": os.getenv("ACCESS_CLIENT_SECRET")}
    query_data = {}
    measurement_id = input("Fill id of measurement to delete: ")

    query_data["measurement_id"] = measurement_id 
    r = requests.delete('https://brics-api.electimore.xyz/measurement/delete', headers=headers, params=query_data)

    print(r.status_code)
    print(r.text)