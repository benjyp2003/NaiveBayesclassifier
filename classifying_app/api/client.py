import requests


class Client:
    def __init__(self, api_url: str = 'http://127.0.0.1:8080', builder_url: str = 'http://building_app_container:8000'):
        self.api_url = api_url
        self.builder_url = builder_url

    def get_model_from_trainer(self):
        """
        Get the model from the trainer.
        """
        try:
            response = requests.get(f"{self.builder_url}/get_model")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"message": f"An error occurred: {e}", "status": "error"}


    def classify_new_example(self, new_example: dict):
        """
        Classify a new example using the specified model.
        """
        try:
            response = requests.post(f"{self.api_url}/classify", json={"new_example": new_example})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"message": f"An error occurred: {e}", "status": "error"}
