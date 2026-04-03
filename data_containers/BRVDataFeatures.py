from dataclasses import dataclass, field
from numpy import ndarray

@dataclass
class BRVDataFeatures:
    def __init__(self):
        pass

    feature_files: list 
    feature_colors: list
    feature_count: list
    #TODO: define ndarrays and convert lists to ndarrays, if possible
    features: ndarray
    features_pca: ndarray

    person_colors: dict # dictionary for colors of data points for different person labels
    person_indices: dict # dictionary holding arrays of indices in feature data for people
    person_initials: list # array holding all initials for labels in legend
    feature_index: int = 0