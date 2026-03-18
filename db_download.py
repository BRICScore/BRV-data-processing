import sys
sys.path.append("utils")
from config import *
from database_access.requests import *

def main():
    dotenv.load_dotenv()
    downloadMeasurement()

if __name__ == "__main__":
    main()