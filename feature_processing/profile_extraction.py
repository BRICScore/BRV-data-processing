import matplotlib.pyplot as plt
import numpy as np
import math

def extract_data_profiles(feature_data):
    # just for comparison purposes here is a plot for everything in feature data
    # for every person we get mean bpm and breath_depth and plot it on one plot

    for person in feature_data.person_initials:
        records = feature_data.features[feature_data.person_indices[person]]
        mean_bpm, mean_breath_depth = np.mean(records[:,0]), np.mean(records[:,1])
        bps = mean_bpm/60
        spb = 1/bps
        x = np.linspace(0,spb,num=50)
        y = []
        for point in x:
            y.append(math.sin((2*math.pi/spb)*point)*mean_breath_depth)
        plt.plot(x,y,c=feature_data.person_colors[person])
    plt.legend(feature_data.person_initials)
    plt.show()

    #calculate personal profile here: