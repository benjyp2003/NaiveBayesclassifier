import requests
import time


class Client:
    def __init__(self, url: str = 'http://127.0.0.1:8000'):
        self.url = url


    def load_hardcoded_data(self):
        """ Load hardcoded data from the server."""
        try:
            response = requests.get(f"{self.url}/load_hardcoded_data")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"message": f"An error occurred while client data load request: {e}", "status": "error"}


    def get_model(self):
        """
        Get the model from the server.
        """
        try:
            response = requests.get(f"{self.url}/get_model")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"message": f"An error occurred: {e}", "status": "error"}


    def send_for_training(self, data: dict):
        """
        Send data to the server for training a model, and return the model.
        """
        try:
            response = requests.post(f"{self.url}/train", json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"message": f"An error occurred: {e}", "status": "error"}


    def test_model(self, data: dict):
        """
        Test a model's accuracy with a given dataset.
        """
        try:
            response = requests.post(f"{self.url}/test_model", json={"data": data})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"message": f"An error occurred: {e}", "status": "error"}




    def cache_model(self, model: dict):
        """
        Cache the model temporarily on the server and return the path to the cached model.
        """
        try:
            response = requests.post(f"{self.url}/cache_model", json={"model": model})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"message": f"An error occurred: {e}", "status": "error"}


    def load_data_from_file(self, path: str):
        """
        Load data from a file and return it as a dictionary.
        The file should be in CSV format.
        """
        try:
            response = requests.post(f"{self.url}/load_data", json={"path": path})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"message": f"An error occurred: {e}", "status": "error"}


    def clean_csv_file(self, file_path: str, **cleaning_options):
        """
        Clean a CSV file using the server's cleaning endpoint.

        Returns:
            dict: Response from the cleaning endpoint
        """
        try:
            request_body = {"file_path": file_path, **cleaning_options}
            response = requests.post(f"{self.url}/clean_csv", json=request_body)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"message": f"An error occurred: {e}", "status": "error"}




