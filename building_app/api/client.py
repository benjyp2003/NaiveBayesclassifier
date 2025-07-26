import requests


class Client:
    def __init__(self, url: str = 'http://127.0.0.1:8000'):
        self.url = url

    def load_hardcoded_data(self):
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


    def get_model_by_name(self, model_name: str):
        """
        Get a model by its name.
        """
        try:
            response = requests.post(f"{self.url}/get_model_by_name", json={"model_name": model_name})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"message": f"An error occurred: {e}", "status": "error"}


    def get_all_available_models(self):
        """
        Get all available models from the server.
        """
        try:
            response = requests.get(f"{self.url}/get_all_available_models")
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            return {"message": f"An error occurred while getting available models: {e}", "status": "error"}


