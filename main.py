def main():
    action = input("1 - create data profiles (nie działa) \n2 - data processing pipeline \n3 - db download \n4 - visualize data\n")
    match action:
        case "1":
            from create_data_profiles import main as create_data_profiles_main
            create_data_profiles_main()
        case "2":
            from data_processing_pipeline import main as data_processing_pipeline_main
            data_processing_pipeline_main()
        case "3":
            from db_download import main as db_download_main
            db_download_main()
        case "4":
            from visualize_data import main as visualize_data_main
            visualize_data_main()


if __name__ == "__main__":
        main()