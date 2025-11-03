import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA

from main import ADC_COUNT
NO_OF_FEATURES_AFTER_ALG = 2
NO_OF_SCALARS = 3
NO_OF_LINES = 5
FEATURES_PATH = './features'

class FeatureData():
    def __init__(self):
        self.feature_files = []
        self.feature_files_count = 0
        self.features = None
        self.feature_count = None
        self.features_pca = None

def visualize_data():
    feature_data = FeatureData()
    feature_data.features = feature_loading(feature_data)
    PCA_algorithm(feature_data)
    plot_pca_data(feature_data)

def feature_loading(feature_data):
    features = []
    file = None
    with os.scandir(FEATURES_PATH) as es:
        for e in es:
            if e.is_file() and e.name.endswith('.txt'):
                feature_data.feature_files.append(e.name)
                feature_data.feature_files_count += 1
                feature_vector = []
                with open(e.path, encoding='utf-8') as f:
                    file = f.read().split("\n")
                for feature in range(NO_OF_SCALARS):
                    feature_val = float(file[feature].split()[-1])
                    feature_vector.append(feature_val)
                for feature in range(NO_OF_SCALARS, NO_OF_LINES):
                    for i in range(ADC_COUNT):
                        feature_val = float(file[feature].split()[i+1])
                        feature_vector.append(feature_val)
                features.append(feature_vector)
    return np.array(features)

def standarize_data(feature_data):
    feature_data.feature_count = len(feature_data.features[0])
    for i in range(feature_data.feature_count):
        feature_data.features[:, i] = (feature_data.features[:, i] - np.mean(feature_data.features[:, i])) / np.std(feature_data.features[:, i])

def PCA_algorithm(feature_data):
    standarize_data(feature_data)
    pca = PCA(n_components=NO_OF_FEATURES_AFTER_ALG)
    feature_data.features_pca = pca.fit_transform(feature_data.features)

def plot_pca_data(feature_data):
    plt.title(f"Representing {feature_data.feature_count} features with {NO_OF_FEATURES_AFTER_ALG} using PCA")
    plt.scatter(feature_data.features_pca[:, 0], feature_data.features_pca[:, 1])
    for i in range(feature_data.feature_files_count):
        plt.text(feature_data.features_pca[i, 0], feature_data.features_pca[i, 1], feature_data.feature_files[i])
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def main():
    visualize_data()

if __name__ == "__main__":
    main()