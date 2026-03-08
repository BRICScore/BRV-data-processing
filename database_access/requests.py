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

    print('Input labels for the measurement - FORMAT: "label1", "label2", ...')
    labels = str(input())
    labels = f"[{labels}]"
    form_data["labels"] = labels

    files = {"measurement_file_raw" : open(filepath_raw, "rb"), "measurement_file_work": open(filepath_work, "rb")}

    r = requests.put('https://brics-api.electimore.xyz/measurement/upload', headers=headers, files=files, data=form_data)

    print(r.status_code)
    print(r.text)
    return
    
def downloadMeasurement():
    headers = {"CF-Access-Client-Id": os.getenv("ACCESS_CLIENT_ID"), "CF-Access-Client-Secret": os.getenv("ACCESS_CLIENT_SECRET")}
    query_data = {}
    #TODO: MOVE UI TO AN APP/WEBPAGE
    print("Fill measurement download query parameters")
    person_id = str(input("Person ID (currently email): "))
    if person_id != "":
        query_data["person_id"] = person_id
    print('Input labels for the query - FORMAT: "label1", "label2", ...')
    labels = f"[{str(input("Labels: "))}]"
    if labels != "[]":
        query_data["labels"] = labels
    length_min = input("Minimum length in minutes: ")
    if length_min == "":
        length_min = 0
        query_data["length_min"] = length_min * 60000 # duration in DB kept in milliseconds
    length_max = input("Maximum length in minutes: ")
    if length_max == "":
        length_max = 24*60 # 1 day
    if length_max != None:
        query_data["length_max"] = length_max * 60000 
    quality = str(input("Quality - work or raw: "))
    if quality != "":
        query_data["quality"] = quality
    
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
    r = requests.get('https://brics-api.electimore.xyz/measurement/download', headers=headers, params=query_data)

    print(r.status_code)
    print(r.text)