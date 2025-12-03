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