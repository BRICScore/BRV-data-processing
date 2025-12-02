import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
import json
import random

NO_OF_FEATURES_AFTER_ALG = 2
FEATURES_PATH = './features'

class FeatureData():
    def __init__(self):
        self.feature_files = []
        self.feature_colors = []
        self.features = None
        self.feature_count = None
        self.features_pca = None

        self.feature_index = 0
        self.person_colors = {} # dictionary for colors of data points for different person labels
        self.person_indices = {} # dictionary holding arrays of indices in feature data for people
        self.person_initials = [] # array holding all initials for labels in legend

def visualize_data():
    feature_data = FeatureData()
    # feature_data.features = feature_loading(feature_data)
    feature_data.features = feature_loading2(feature_data)
    PCA_algorithm(feature_data)
    plot_pca_data(feature_data)

def parse_features_line(line):
    feature_vector = []
    for key in line:
        if isinstance(line[key], list):
            for val in line[key]:
                feature_vector.append(val)
        elif isinstance(line[key], str):
            print(f"{line[key]}")
        else:
            feature_vector.append(line[key])
    return feature_vector

def parse_features_line2(line, feature_data):
    feature_vector = []
    person = None
    for key in line:
        if isinstance(line[key], list):
            for val in line[key]:
                feature_vector.append(val)
        elif isinstance(line[key], str):
            # print(f"{line[key]}")
            person = line[key]
        else:
            feature_vector.append(line[key])
    if person not in feature_data.person_initials:
        feature_data.person_initials.append(person)
        color = random.randrange(0, 2**24)
        hex_color = hex(color)
        color_part = hex_color[2:]
        while len(color_part) < 6:
                color_part = "0" + color_part
        rand_color = "#" + color_part
        feature_data.person_colors[person] = rand_color
        feature_data.person_indices[person] = []
    feature_data.person_indices[person].append(feature_data.feature_index)
    feature_data.feature_index += 1
    return feature_vector

def feature_loading(feature_data):
    features = []
    color_index = 0
    file = None
    with os.scandir(FEATURES_PATH) as es:
        for e in es:
            if e.is_file() and e.name.endswith('.jsonl'):
                feature_vector = []
                color = random.randrange(0, 2**24)
                hex_color = hex(color)
                color_part = hex_color[2:]
                while len(color_part) < 6:
                    color_part = "0" + color_part
                rand_color = "#" + color_part
                with open(e.path, encoding='utf-8') as f:
                    file = f.read().split("\n")
                    for f_line in file:
                        if f_line != '': # last newline produces empty string
                            feature_vector = parse_features_line(json.loads(f_line))
                            features.append(feature_vector)
                            feature_data.feature_colors.append(rand_color)
                    feature_data.feature_files.append(e.name)
                color_index += 1
    return np.array(features)

def feature_loading2(feature_data):
    features = []
    with open("./features/extracted_features.jsonl", "r") as file:
        record = file.readline()
        while record:
            feature_vector = parse_features_line2(json.loads(record), feature_data)
            features.append(feature_vector)
            record = file.readline()
    # print(feature_data.person_initials)
    # print(feature_data.person_indices)
    # print(feature_data.person_colors)
    # print(features)
    return np.array(features)

def standarize_data(feature_data):
    feature_data.feature_count = len(feature_data.features[0])
    for i in range(feature_data.feature_count):
        std = np.std(feature_data.features[:, i]) # change the variance if 0 not to divide by it
        feature_data.features[:, i] = (feature_data.features[:, i] - np.mean(feature_data.features[:, i])) / ( 1 if std == 0 else std)

def PCA_algorithm(feature_data):
    standarize_data(feature_data)
    pca = PCA(n_components=NO_OF_FEATURES_AFTER_ALG)
    feature_data.features_pca = pca.fit_transform(feature_data.features)

def plot_pca_data(feature_data):
    if NO_OF_FEATURES_AFTER_ALG == 2:
        """
        plt.title(f"Representing {feature_data.feature_count} features with {NO_OF_FEATURES_AFTER_ALG} using PCA")
        plt.scatter(feature_data.features_pca[:, 0], feature_data.features_pca[:, 1], c=feature_data.feature_colors)
        for i in range(len(feature_data.feature_files)):
            plt.text(feature_data.features_pca[i, 0], feature_data.features_pca[i, 1], feature_data.feature_files[i])
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
        """
        plt.title(f"Representing {feature_data.feature_count} features with {NO_OF_FEATURES_AFTER_ALG} using PCA")
        for person in feature_data.person_initials:
            records = feature_data.features_pca[feature_data.person_indices[person]]
            plt.scatter(records[:,0], records[:,1], c=feature_data.person_colors[person])
        plt.legend(feature_data.person_initials)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
    if NO_OF_FEATURES_AFTER_ALG == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_title(f"Representing {feature_data.feature_count} features with {NO_OF_FEATURES_AFTER_ALG} using PCA")
        ax.scatter(feature_data.features_pca[:,0], feature_data.features_pca[:,1], feature_data.features_pca[:,2])
        for i in range(len(feature_data.feature_files)):
            ax.text(feature_data.features_pca[i, 0], feature_data.features_pca[i, 1], feature_data.features_pca[i, 2], feature_data.feature_files[i])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()
def main():
    visualize_data()

if __name__ == "__main__":
    main()