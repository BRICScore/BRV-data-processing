import os
import json
import numpy as np

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

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
    print(data.existing_labels)

    X = []

    for seq in data.data:
        timestamps, values = seq  # each shape (900,)
        sequence = np.stack([timestamps, values], axis=1)  # (900, 2)
        X.append(sequence)

    X = np.array(X)
    X[:, :, 0] = (X[:, :, 0] - X[:, :, 0].mean()) / X[:, :, 0].std()
    X[:, :, 1] = (X[:, :, 1] - X[:, :, 1].mean()) / X[:, :, 1].std()

    label_encoder = LabelEncoder()
    label_encoder.fit(data.existing_labels)

    y_int = label_encoder.transform(data.data_labels)

    num_classes = len(data.existing_labels)
    y = to_categorical(y_int, num_classes)

    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val, y_train_int, y_test_int = train_test_split(
        X, y, y_int, test_size=0.2, random_state=42, stratify=y_int
    )

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(900, 2)),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32
    )

    y_pred = model.predict(X_val)
    predicted_classes = np.argmax(y_pred, axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_classes)
    predicted_labels = np.array(predicted_labels)

    y_pred_all = model.predict(X)
    predicted_classes_all = np.argmax(y_pred_all, axis=1)
    predicted_labels_all = np.array(label_encoder.inverse_transform(predicted_classes_all))

    true_test_labels = label_encoder.inverse_transform(y_test_int)

    accuracy_test = (predicted_labels == true_test_labels).mean() * 100
    print(f"Test accuracy: {accuracy_test:.2f}%")
    ###############

    true_labels = np.array(data.data_labels)

    matches = predicted_labels_all == true_labels
    accuracy = matches.mean() * 100

    print(f"Overall accuracy: {accuracy:.2f}%")



if __name__ == "__main__":
    validate_data_with_lstm()