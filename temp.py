from utils.file_processing import MeasurementDataBuilder
from data_containers import MeasurementData

data = MeasurementData()

mdb = MeasurementDataBuilder(data)

mdb.build_data("results/clean/clean_input_30_JD_sit_1.jsonl", "clean")

print(data)