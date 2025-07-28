import requests
import time


class Client:
    def __init__(self, api_url: str = 'http://127.0.0.1:8001', builder_url: str = 'http://building_app_container:8000'):
        self.api_url = api_url
        self.builder_url = builder_url


    def check_builder_server_health(self, max_retries: int = 3, retry_delay: float = 2.0):
        """
        Check if the builder server is up and running.

        Returns:
            dict: {"status": "success"|"error", "message": str}
        """
        for attempt in range(max_retries):
            try:
                # Try to connect to the root endpoint of the builder server
                response = requests.get(f"{self.builder_url}/", timeout=5)
                if response.status_code == 200:
                    return {"status": "success", "message": "Builder server is up and running"}
                else:
                    return {"status": "error", "message": f"Builder server returned status code: {response.status_code}"}
            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    return {"status": "error", "message": f"Builder server is not reachable after {max_retries} attempts"}
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    return {"status": "error", "message": f"Builder server request timed out after {max_retries} attempts"}
            except Exception as e:
                return {"status": "error", "message": f"Unexpected error checking builder server: {e}"}
        
        return {"status": "error", "message": "Failed to connect to builder server"}

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
