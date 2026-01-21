import numpy as np
from config import *

class ADCdata:
    def __init__(self):
        self.timestamps = np.array([])
        self.adc_output_data = [np.array([]) for _ in range(ADC_COUNT)]
        self.adc_normalized_data = [np.array([]) for _ in range(ADC_COUNT)]
        self.adc_voltage_means = []
        self.smoothed_signal = None
        self.signal_minima = None
        self.signal_maxima = None
        self.non_time_outlier_adc_data = None
        self.time_outlier_adc_data = None
        self.non_amplitude_outlier_adc_data = None
        self.amplitude_outlier_adc_data = None
        self.breath_count = 0
        self.avg_breath_depth = 0
        self.avg_breath_depth_std_dev = 0
        self.final_adc_data = None
        self.final_adc_timestamps = None

        self.plot_enabled = False