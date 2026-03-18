import argparse
import sys
import dotenv
sys.path.append("data_processing")
sys.path.append("feature_processing")
sys.path.append("utils")

dotenv.load_dotenv(".env")

from initial_data_processing import process_file
from config import *

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


    process_file(parser)

if __name__ == "__main__":
    main()