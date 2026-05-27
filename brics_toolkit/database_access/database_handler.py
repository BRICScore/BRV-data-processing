import requests
import os
from pathlib import Path
from urllib3 import Retry
from requests.adapters import HTTPAdapter
import dotenv
from ..utils.input_measurement_metadata import input_measurement_metadata
from ..utils.build_model_metadata import build_model_metadata
import tempfile
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

class DatabaseHandler:
    """
            Class for handling usage of the database functions

            Attributes
            ----------
            session : requests.Session
                Session object for accessing 
            
            Methods
            ------------
            uploadMeasurement(filepath_raw: str, filepath_clean: str):
                Uploads chosen measurement into the remote database.
            
            downloadMeasurement():
                Downloads measurements fitting the criteria from the database into a configured zip archive.

            deleteMeasurement():
                Deletes chosen measurement from database using its ID.
        """

    def __init__(self):
        dotenv.load_dotenv(".env")
        self._session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[500,502,503,504])
        adapter = HTTPAdapter(max_retries=retries)
        headers = {"CF-Access-Client-Id": os.getenv("ACCESS_CLIENT_ID"), "CF-Access-Client-Secret": os.getenv("ACCESS_CLIENT_SECRET")}
        self._session.headers.update(headers) # type: ignore
        self._session.mount("https://", adapter)

    def uploadMeasurement(self, filepath_raw: str, filepath_clean: str, filepath_features: str):
        """
            Upload measurement files into the database for future use.

            Parameters
            ----------
            filepath_raw : str
                String containing the path (absolute or relative) to a file containing raw BRV data.
            filepath_clean : str
                String containing the path (absolute or relative) to a file containing corresponding cleaned BRV data.
            
            Returns
            -------
            none

            Side Effects
            ------------
            This function will result in a files being uploaded into the database and creation of a record corresponding to them.
        """

        measurement_metadata_dict = (input_measurement_metadata([filepath_raw, filepath_clean, filepath_features])).model_dump_json(by_alias=True)
        
        form_data = {"measurement_metadata": measurement_metadata_dict}
        try:
            tmp_dir = Path(tempfile.mkdtemp())
            (tmp_dir / "dataset").mkdir()
            shutil.copy(filepath_raw, tmp_dir / "dataset" / "raw")
            shutil.copy(filepath_clean, tmp_dir / "dataset" / "clean")
            shutil.copy(filepath_features, tmp_dir / "dataset" / "features")
            with ZipFile(tmp_dir / "zipped.zip", "w", compression=ZIP_DEFLATED, compresslevel=9) as zf:
                for file in (tmp_dir / "dataset").rglob("*"):
                    if file.is_file():
                        zf.write(
                            file,
                            arcname=file.relative_to(tmp_dir / "dataset")
                        )
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        with open(tmp_dir / "zipped.zip", "rb") as file_zip:
            files = {"measurement_file_zip": file_zip}
            r = self._session.put('https://brics-api.electimore.xyz/measurements/upload', files=files, data=form_data)

        shutil.rmtree(tmp_dir, ignore_errors=True)
        
        r.raise_for_status()
        print(r.status_code)
        print(r.text)
        return
        
    def downloadMeasurement(self):
        """
            Download measurement files from the database with accordance to the query.

            Parameters
            ----------
            none
            
            Returns
            -------
            none
            
            Side Effects
            ------------
            This function will result in a zip archive being created in the user's Downloads directory, if it exists.
        """

        
        #TODO: MOVE UI TO AN APP/WEBPAGE
        print("Fill measurement download query parameters")

        person_id = input("Person ID (currently email): ").strip() or None
        
        length_min = int(input("Minimum length in minutes: ").strip() or 0)
        length_max = int(input("Maximum length in minutes: ").strip() or 24*60) # default value - 1 day
        
        age_min = int(input("Minimum age of subject: ").strip() or 0)
        age_max = int(input("Maximum age of subject: ").strip() or 100)
        
        level = input("Enter acceptable levels (raw/clean/features) seperated by space or press enter for all").strip().split() or None
        gender = input("Enter genders (male/female) seperated by space or skip for all").strip().split() or None
        activity = input("Enter activities seperated by space or skip or all").strip().split() or None
        condition = input("Enter conditions seperated by space or skip or all").strip().split() or None
        health = input("Enter health statuses seperated by space or skip or all").strip().split() or None

        weight_min = int(input("Minimum weight in kgs: ").strip() or 0)
        weight_max = int(input("Maximum weight in kgs: ").strip() or 200)

        height_min = int(input("Minimum height in cm: ").strip() or 0)
        height_max = int(input("Maximum height in cm: ").strip() or 250)

        query_data = {
            "person_id": person_id,
            "length_min": length_min * 60000, #length kept in db in milliseconds
            "length_max": length_max * 60000,
            "age_min": age_min,
            "age_max": age_max,
            "level": level,
            "gender": gender,
            "activity": activity,
            "condition": condition,
            "health": health,
            "weight_min" : weight_min,
            "weight_max" : weight_max,
            "height_min" : height_min,
            "height_max" : height_max
        }
        
        r = self._session.get('https://brics-api.electimore.xyz/measurements/download', params=query_data)

        path = Path.home() / "Downloads" / "measurements_dataset.zip"
        if r.status_code == 200:
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        r.raise_for_status()
        print(r.status_code)
        return

    def deleteMeasurement(self):
        """
            Delete measurement record and files from database using it's ID.

            Parameters
            ----------
            none
            
            Returns
            -------
            none
            
            Side Effects
            ------------
            This function will result in a measurement record and files being permamently deleted, if they exist.
        """
        query_data = {}
        measurement_id = input("Fill id of measurement to delete: ")

        query_data["measurement_id"] = measurement_id 
        r = self._session.delete('https://brics-api.electimore.xyz/measurements/delete', params=query_data, timeout=30)

        r.raise_for_status()
        print(r.status_code)
        print(r.text)

    def upload_model(self, filepath_weights: str, filepath_pth: str, filepath_scaler: str):
        """
            Upload model files to the database.

            Parameters
            ----------
            filepath_weights: str
                Path to file with weights
            filepath_pth: str
                Path to pth file
            filepath_scaler: str
                Path to file with scaler
            
            Returns
            -------
            none
            
            Side Effects
            ------------
            This function will result in a model record and files being uploaded to database, if they are correct.
        """

        form_data = build_model_metadata([filepath_weights, filepath_pth, filepath_scaler]).model_dump_json(by_alias=True)
        try:
            tmp_dir = Path(tempfile.mkdtemp())
            (tmp_dir / "dataset").mkdir()
            shutil.copy(filepath_weights, tmp_dir / "model" / "raw")
            shutil.copy(filepath_pth, tmp_dir / "model" / "pth")
            shutil.copy(filepath_scaler, tmp_dir / "model" / "scaler")
            with ZipFile(tmp_dir / "zipped.zip", "w", compression=ZIP_DEFLATED, compresslevel=9) as zf:
                for file in (tmp_dir / "dataset").rglob("*"):
                    if file.is_file():
                        zf.write(
                            file,
                            arcname=file.relative_to(tmp_dir / "dataset")
                        )
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        with open(tmp_dir / "zipped.zip", "rb") as file_zip:
            files = {"model_zip": file_zip}
            r = self._session.put('https://brics-api.electimore.xyz/models/upload', files=files, data=form_data)

        shutil.rmtree(tmp_dir, ignore_errors=True)

        r.raise_for_status()
        print(r.status_code)
        print(r.text)
        return

    def download_model(self):
        """
            Download model files from the database.

            Parameters
            ----------
            none
            
            Returns
            -------
            none
            
            Side Effects
            ------------
            This function will result in a zip archive with model files being created in the user's Downloads directory, if it exists.
        """
        
        r = self._session.get('https://brics-api.electimore.xyz/models/download')

        r.raise_for_status()
        print(r.status_code)
        print(r.text)

        path = Path.home() / "Downloads" / "model.zip"
        if r.status_code == 200:
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        return
    
    def delete_model(self):
        """
            Delete model record and files from database.

            Parameters
            ----------
            none
            
            Returns
            -------
            none
            
            Side Effects
            ------------
            This function will result in a model record and files being permamently deleted, if they exist.
        """
        r = self._session.delete('https://brics-api.electimore.xyz/models/delete')

        r.raise_for_status()
        print(r.status_code)
        print(r.text)
        
        return