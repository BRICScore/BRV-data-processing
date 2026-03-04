import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.interpolate as spi
import copy
import json
import sys
import math

RECORD_COUNT = 3000
MAX_24B = 2**23 - 1
MIN_24B = -2**23
VOLTAGE_RANGE = 0.4  # -0.2V to 0.2V
ADC_COUNT = 5
TARGET_ADC = 3  # ADC to analyze for functions

SEGMENT_LENGTH_MS = 120_000 # 2-minute segments
ACCEPTABLE_DATA_LOSS = 0.2 # fraction threshold for the data to be analyzed

TARGET_ADC = 2 # indexed from ZERO "0" (0-4)
INDEX = 0

PERSON_ID = 3
ACTIVITY_ID = 4

PERCENTILE_THRESHOLD = 10  # % for both lower and upper bounds

#feature vector discard criteria
MIN_BPM = 5.0
MAX_BPM = 25.0

STD_DEV_CONST = 0.1 # when finding local maxima in breath separation we calculate mean + std_dev * STD_DEV_CONST as a threshold for peaks
MIN_DISTANCE = 30 # minimum distance between peaks in breath separation

INHALE_INDEX = 0
EXHALE_INDEX = 2
MIN_INHALE_OR_EXHALE_LENGTH = 500.0