import matplotlib.pyplot as plt
import numpy as np
import itertools
import sys
from sklearn.decomposition import PCA
sys.path.append("feature_processing")
from eigenvalues_extraction import *
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
    feature_data.features = feature_loading(feature_data)
    PCA_algorithm(feature_data)
    extract_eigenvalues(feature_data)

    plot_pca_data(feature_data)

def parse_features_line(line, feature_data):
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
    with open("./features/extracted_features.jsonl", "r") as file:
        record = file.readline()
        while record:
            feature_vector = parse_features_line(json.loads(record), feature_data)
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

def calculate_covariance_matrix(feature_data):
    no_of_data_points = len(feature_data.features)
    A = np.zeros(shape=(feature_data.feature_count, feature_data.feature_count))
    for x1 in range(feature_data.feature_count):
        x1_mean = np.mean(feature_data.features[:,x1])
        for x2 in range(feature_data.feature_count):
            x2_mean = np.mean(feature_data.features[:,x2])
            for i in range(no_of_data_points):
                A[x1][x2] += (feature_data.features[i][x1]*x1_mean) * (feature_data.features[i][x2]*x2_mean)
    A /= no_of_data_points-1
    return A

def PCA_algorithm(feature_data):
    standarize_data(feature_data)
    pca = PCA(n_components=NO_OF_FEATURES_AFTER_ALG)
    feature_data.features_pca = pca.fit_transform(feature_data.features)
    print("Graph directions:",pca.components_)
    for feature_direction in pca.components_:
        print("Feature importance in each direction:", np.argsort(np.abs(feature_direction))[::-1])

def plot_pca_data(feature_data):
    if NO_OF_FEATURES_AFTER_ALG == 2:
        plt.title(f"Representing {feature_data.feature_count} features with {NO_OF_FEATURES_AFTER_ALG} using PCA")
        for person in feature_data.person_initials:
            records = feature_data.features_pca[feature_data.person_indices[person]]
            plt.scatter(records[:,0], records[:,1], c=feature_data.person_colors[person])
        plt.legend(feature_data.person_initials)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
    if NO_OF_FEATURES_AFTER_ALG >= 3:
        feature_data.person_colors = {"JD_sit": "red", "MJ_sit": "green", "MK_sit": "blue"}
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_title(f"Representing {feature_data.feature_count} features with {NO_OF_FEATURES_AFTER_ALG} using PCA")
        for person in feature_data.person_initials:
            records = feature_data.features_pca[feature_data.person_indices[person]]
            ax.scatter(records[:,0], records[:,1], records[:,2], c=feature_data.person_colors[person])
        ax.legend(feature_data.person_initials)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        plt.show()
        """
        feature_numbers = []
        for i in range(NO_OF_FEATURES_AFTER_ALG):
            feature_numbers.append(i)
        graph_dimensions = list(itertools.combinations(feature_numbers,2))
        print(graph_dimensions)
        for graph in graph_dimensions:
            plt.title(f"Representing {feature_data.feature_count} features with {NO_OF_FEATURES_AFTER_ALG} using PCA")
            for person in feature_data.person_initials:
                records = feature_data.features_pca[feature_data.person_indices[person]]
                plt.scatter(records[:,graph[0]], records[:,graph[1]], c=feature_data.person_colors[person])
            plt.legend(feature_data.person_initials)
            plt.xlabel(f'Feature {graph[0]}')
            plt.ylabel(f'Feature {graph[1]}')
            plt.show()
        """
def main():
    visualize_data()

if __name__ == "__main__":
    main()