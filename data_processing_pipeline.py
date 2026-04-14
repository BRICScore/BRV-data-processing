import sys
sys.path.append("utils")

from config import *
import argparse
from data_containers import BRVDataClean
from data_containers import MeasurementData, MeasurementMetadata
from data_processing.initial_data_processing import initial_data_processing
from extract_features import extract_features
from database_access.database_handler import DatabaseHandler

def split_data_into_segments(input_file : Path, BRV_data_clean : BRVDataClean):
    """
        Split the resampled ADC data into segments that contain values from a specific time window,
        and save each segment into a separate JSONL file.

        Parameters
        ----------
        input_file: Path
            The path to the raw input file containing the ADC data.
        BRV_data_clean : BRVDataClean
            The BRVDataClean object containing the cleaned and resampled ADC data and timestamps further
            defined in project's DTP.

        Returns
        -------
        None          
        
        Side Effects
        ------------
        This function creates multiple JSONL files in the "./results" directory, each containing a segment of the ADC data.
    """

    segment_index = 0
    total_segments = int(np.ceil(BRV_data_clean.timestamps[-1] / SEGMENT_LENGTH_MS))
    filename = str(input_file).split("_")
    time = filename[1]
    person = filename[2]
    condition = filename[3]
    no_of_sample = filename[4]

    for segment_index in range(total_segments):
        segment_start = segment_index * SEGMENT_LENGTH_MS
        segment_end = segment_start + SEGMENT_LENGTH_MS
        with open(f"./results/clean_{time}_{segment_index}_{person}_{condition}_{no_of_sample.split('.')[0]}.jsonl", 'w') as o_f:
            for i in range(len(BRV_data_clean.timestamps)):
                if segment_start <= BRV_data_clean.timestamps[i] < segment_end:
                    record = {
                        "timestamp": int(BRV_data_clean.timestamps[i]),
                        "adc_outputs": [BRV_data_clean.adc_data[a][i] for a in range(ADC_COUNT)]
                    }
                    o_f.write(json.dumps(record) + "\n")

def save_clean_data(BRV_data_clean : BRVDataClean, input_file : Path):
    os.makedirs("results/clean", exist_ok=True)
    results_path = f"results/clean/clean_{input_file}"
    with open(results_path, 'w') as f:
        for i in range(len(BRV_data_clean.timestamps)):
            record = {
                "timestamp": int(BRV_data_clean.timestamps[i]),
                "adc_outputs": [BRV_data_clean.adc_data[a][i] for a in range(ADC_COUNT)]
            }
            f.write(json.dumps(record) + "\n")
    return Path(results_path)

def clear_results_folder():
    # remove all everythong in results directory
    with os.scandir('results') as results:
        for result in results:
            if result.is_file():
                os.remove(result.path)


def parser_setup():
    parser = argparse.ArgumentParser(description="Data parser and feature extractor")

    parser.add_argument('input_file', type=str,
                    help='A required argument containing input file for the programme')

    parser.add_argument('--plot', action='store_true',
                    help='A boolean switch for plotting transformations')
    
    parser.add_argument('--debugplot', action='store_true',
                    help='A boolean switch for plotting while debugging')

    return parser

def main():
    parser = parser_setup()
    args = parser.parse_args()
    input_file = args.input_file
    plot_enabled = args.plot
    debug_plot = args.debugplot

    measurement_data = MeasurementData()
    measurement_metadata = MeasurementMetadata()
    measurement_metadata.filepath_raw = Path(input_file)
    measurement_data.metadata = measurement_metadata
    
    initial_data_processing(BRV_measurement_data = measurement_data, target_adc = TARGET_ADC, plot_enabled = plot_enabled)
    measurement_data.metadata.filepath_clean = save_clean_data(measurement_data.data_clean, input_file)
    split_data_into_segments(input_file, measurement_data.data_clean)
    extract_features(measurement_data=measurement_data)

    db_handler = DatabaseHandler()
    fr = measurement_data.metadata.filepath_raw
    fc = measurement_data.metadata.filepath_clean
    ff = measurement_data.metadata.filepath_features
    db_handler.uploadMeasurement(filepath_raw=fr.name, filepath_clean=fc.name, filepath_features=ff.name)
    clear_results_folder()


"""
    BRV_measurement_data = poprawna iniclajizacja PUSTEGO obiektu MeasurementData
    BRV_measurement_data.BRV_data_intermediate = process_raw_file(input_file, plot_enabled=plot_enabled)
    BRV_measurement_data.BRV_data_clean = outlier_detection(BRV_measurement_data.BRV_data_intermediate, target_adc=TARGET_ADC, plot_enabled=plot_enabled)
    split_data_into_segments(input_file, BRV_measurement_data.BRV_data_clean)
    extract_features(BRV_measurement_data, target_adc=TARGET_ADC, plot_enabled=plot_enabled)
    remove_data_segments(input_file)
    input_measurement_metadata() -> ostatecznie wypełniamy measurement_metadata i chyba też measurement_data przed uploadem
    upload_measurement()
"""

if __name__ == "__main__":
    main()