import os
import json
import numpy as np

class SequenceData():
    def __init__(self):
        self.data_rows = 0
        self.data = None
        self.data_labels = []
        self.existing_labels = []

def load_segment_data(data):
    the_data = []
    for file in os.scandir("./results"):
        if file.name.endswith(".jsonl"):
            name_split = file.name.split("_")
            label = name_split[-3]+"_"+name_split[-2]
            if label not in data.existing_labels:
                data.existing_labels.append(label)
            values = []
            timestamps = []
            with open("./results/"+file.name) as f:
                for _ in range(900):
                    record = f.readline()
                    if not record:
                        break
                    record_data = json.loads(record)
                    timestamps.append(record_data["timestamp"])
                    values.append(record_data["adc_outputs"][2])
            if len(values) == 900:
                the_data.append([timestamps, values])
                data.data_labels.append(label)
                # print(timestamps[0], values[0])
        # print(file.name)
    data.data = np.array(the_data)

def validate_data_with_lstm():
    data = SequenceData()
    load_segment_data(data)
    # print(len(data.data))
    # print(len(data.data_labels))

if __name__ == "__main__":
    validate_data_with_lstm()