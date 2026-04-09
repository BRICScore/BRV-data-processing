import sys
sys.path.append("utils")
from config import *
from database_access.database_handler import *

def main():
    dotenv.load_dotenv()
    dbh = DatabaseHandler()
    dbh.downloadMeasurement()

if __name__ == "__main__":
    main()