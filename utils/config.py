RECORD_COUNT = 3000
MAX_24B = 2**23 - 1
MIN_24B = -2**23
VOLTAGE_RANGE = 0.4  # -0.2V to 0.2V
ADC_COUNT = 5
TARGET_ADC = 3  # ADC to analyze for functions

SEGMENT_LENGTH_MS = 120_000 # 2-minute segments
ACCEPTABLE_DATA_LOSS = 20 # percentage threshold for the data to be analyzed

TARGET_ADC = 3
INDEX = 0

PERCENTILE_THRESHOLD = 15  # % for both lower and upper bounds

RESAMPLE_NODE_COUNT = 200