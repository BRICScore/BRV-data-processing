RECORD_COUNT = 3000
MAX_24B = 2**23 - 1
MIN_24B = -2**23
VOLTAGE_RANGE = 0.4  # -0.2V to 0.2V
ADC_COUNT = 5
TARGET_ADC = 3  # ADC to analyze for functions

SEGMENT_LENGTH_MS = 120_000 # 2-minute segments
ACCEPTABLE_DATA_LOSS = 0.2 # fraction threshold for the data to be analyzed

TARGET_ADC = 3
INDEX = 0

PERCENTILE_THRESHOLD = 1  # % for both lower and upper bounds

#feature vector discard criteria
MIN_BPM = 5.0
MAX_BPM = 20.0

MIN_PEAK_SPREAD = 30

INHALE_INDEX = 0
EXHALE_INDEX = 2
MIN_INHALE_OR_EXHALE_LENGTH = 1000.0