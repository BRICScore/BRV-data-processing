import numpy as np

def extract_eigenvalues(feature_data):
    A = calculate_covariance_matrix(feature_data)
    eigenvalues = np.linalg.eig(A)[0]
    print("eigenvalues:", np.argsort(np.abs(eigenvalues))[::-1])

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