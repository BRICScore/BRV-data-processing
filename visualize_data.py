import matplotlib.pyplot as plt
import numpy as np
import itertools
import sys
import math
from typing import Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import MDS
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
import seaborn as sns


sys.path.append("feature_processing")
from eigenvalues_extraction import *
from profile_extraction import extract_data_profiles
import json
import random

NO_OF_FEATURES_AFTER_ALG = 2
FEATURES_PATH = './features'

class FeatureData():
    def __init__(self):
        self.feature_files = []
        self.feature_colors = []
        self.features: Optional[np.ndarray] = None
        self.feature_count = None
        self.features_pca: Optional[np.ndarray] = None

        self.feature_index = 0
        self.feature_keys = {}
        self.person_colors = {} # dictionary for colors of data points for different person labels
        self.person_indices = {} # dictionary holding arrays of indices in feature data for people
        self.person_initials = [] # array holding all initials for labels in legend

def visualize_data():
    feature_data = FeatureData()
    create_indices_for_features(feature_data)

    feature_data.features = feature_loading(feature_data)
    # extract_eigenvalues(feature_data)
    MDS_algorithm(feature_data)
    PCA_algorithm(feature_data)
    plot_pca_data(feature_data)

    # accuracy measuring part
    # SVM
    if NO_OF_FEATURES_AFTER_ALG == 2:
        SVM_validation(feature_data=feature_data)

    # heatmap
    plot_heatmap(feature_data=feature_data)

def plot_heatmap(feature_data):
    scaled_features = MinMaxScaler().fit_transform(feature_data.features)
    print("test", scaled_features.shape)
    i = 1
    for person in feature_data.person_initials:
        print(person, feature_data.person_indices[person])
        records = scaled_features[feature_data.person_indices[person]]
        labels = [feature_data.feature_keys[i] for i in range(feature_data.feature_count)]
        plt.subplot(len(feature_data.person_initials),1,i)
        plt.title(person)
        sns.heatmap(records[0:10], xticklabels=labels, vmin=0.0, vmax=1.0)
        i+=1
    plt.show()

def SVM_validation(feature_data):
    # print(feature_data.features_pca)
    for pair in list(itertools.combinations(feature_data.person_initials, 2)):
        person1, person2 = pair
        records1 = feature_data.features_pca[feature_data.person_indices[person1]]
        labels1 = [person1 for record in records1]
        records2 = feature_data.features_pca[feature_data.person_indices[person2]]
        labels2 = [person2 for record in records2]
        # columns are the 2 reduced features
        X = np.concatenate((records1, records2), axis=0)
        y = np.concatenate((labels1, labels2), axis=0)

        svm = SVC(kernel="linear", C=1)
        svm.fit(X,y)

        DecisionBoundaryDisplay.from_estimator(
            svm,
            X,
            response_method="predict",
            alpha=0.8,
            cmap="Pastel1",
            xlabel="x",
            ylabel="y"
        )

        W=svm.coef_[0]
        I=svm.intercept_
        a = -W[0]/W[1]
        b = I[0]/W[1]

        line_x = np.linspace(np.min(X[:,0])-1, np.max(X[:,0])+1, num=10)
        line_y = [a*x - b for x in line_x]
        plt.plot(line_x, line_y, "--", c="k")
        plt.xlim(np.min(X[:,0])-0.5, np.max(X[:,0])+0.5)
        plt.ylim(np.min(X[:,1])-0.5, np.max(X[:,1])+0.5)

        y_pred = svm.predict(X)
        length_y = len(y_pred)
        counter = 0
        for i in range(length_y):
            if y[i] == y_pred[i]:
                counter += 1
        
        print(f"Accuracy for {person1} and {person2}: {(counter/length_y)*100:.2f}%")

        plt.title(f"{person1} and {person2} divided by linear SVM")
        plt.scatter(records1[:, 0], records1[:, 1], c="red")
        plt.scatter(records2[:, 0], records2[:, 1], c="blue")
        plt.legend(["Dividing line", person1, person2])

        plt.show()

def create_indices_for_features(feature_data):
    record = None
    with open("./features/extracted_features.jsonl", "r") as file:
        record = file.readline()
    i = 0
    json_line = json.loads(record)
    for key in json_line:
        if isinstance(json_line[key], list):
            index = 1
            for val in json_line[key]:
                feature_data.feature_keys[i] = key + str(index)
                index += 1
                i += 1
        else:
            feature_data.feature_keys[i] = key
            i += 1

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
            json_record = json.loads(record)
            feature_vector = parse_features_line(json_record, feature_data)
            features.append(feature_vector)
            record = file.readline()
    feature_data.feature_count = len(features[0])
    feature_data.person_colors = {"JD_sit": "red", "MJ_sit": "green", "MK_sit": "blue"} ###########TODO############
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

def MDS_algorithm(feature_data):
    X = feature_data.features
    X_scaled = StandardScaler().fit_transform(X)

    mds = MDS(
        n_components=NO_OF_FEATURES_AFTER_ALG,
        random_state=42,
        n_init=4
    )

    feature_data.features_mds = mds.fit_transform(X_scaled)
    print("MDS result")
    plt.title(f"Representing {feature_data.feature_count} features with {NO_OF_FEATURES_AFTER_ALG} using MDS")
    for person in feature_data.person_initials:
        records = feature_data.features_mds[feature_data.person_indices[person]]
        plt.scatter(records[:,0], records[:,1], c=feature_data.person_colors[person])
    plt.legend(feature_data.person_initials)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    print()

def PCA_algorithm(feature_data):
    # standarize_data(feature_data)
    X = feature_data.features
    X_centered = X - np.mean(X, axis=0)
    cov = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = eigvals[::-1]
    print(eigvals)
    scaled_features = StandardScaler().fit_transform(feature_data.features)
    pca = PCA(n_components=NO_OF_FEATURES_AFTER_ALG)
    feature_data.features_pca = pca.fit_transform(scaled_features)
    i = 1
    for feature_direction in pca.components_:
        print(f"Feature {i} direction values:", feature_direction)
        sorted_indices = np.argsort(np.abs(feature_direction))[::-1]
        # print(feature_data.feature_keys)
        print(f"Feature {i} importance in each direction:", [feature_data.feature_keys[k] for k in sorted_indices])
        i += 1
        print()
    
    eigenvalues = pca.explained_variance_
    print("Eigenvalues:", eigenvalues)
    sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
    # print("Eigenvalues sorted:", [feature_data.feature_keys[k] for k in sorted_indices])

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
        # ax.set_zlabel('Feature 3')
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