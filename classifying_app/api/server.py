import logging
from fastapi import FastAPI, Request
from requests import RequestException

from classifying_app.api.client import Client
from classifying_app.core.classifier import Classifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

MODEL = None

@app.on_event("startup")
async def fetch_model_on_startup():
    """
    Fetch the model from building_app when the server starts.
    """
    global MODEL
    logger.info("Startup: Fetching model from builder...")
    try:
        client = Client()
        result = client.get_model_from_trainer()
        if result.get("status") == "success":
            MODEL = result.get("model")
            logger.info("Startup: Model fetched and loaded successfully.")
        else:
            logger.warning(f"Startup: Failed to fetch model: {result.get('message', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Startup: Error fetching model: {e}")

@app.get('/get_model')
async def get_model():
    """
    Fetches the model from the builder (building_app) and stores it locally.
    """
    global MODEL
    logger.info("/get_model endpoint called.")
    try:
        # Use the Docker network hostname for building_app
        client = Client(api_url='http://building_app:8000')
        response = client.get_model_from_trainer()
        if response.get("status") == "success":
            MODEL = response.get("model")
            logger.info("Model fetched from builder and loaded successfully.")
            return {"message": "Model fetched and loaded successfully.", "status": "success"}
        else:
            logger.warning(f"Failed to fetch model: {response.get('message', 'Unknown error')}")
            return {"message": response.get("message", "Failed to fetch model."), "status": "error"}
    except RequestException as e:
        logger.error(f"Error in /get_model: {e}")
        return {"message": str(e), "status": "error"}
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return {"message": f"An error occurred: {e}", "status": "error"}

@app.post('/classify')
async def classify_data(request: Request):
    """
    Classifies a new data example using the given model name.
    """
    global MODEL
    logger.info("/classify endpoint called.")
    try:
        body = await request.json()
        new_example = body.get("new_example")
        logger.info(f"Received classification request: {body}")
        # check that there are no missing data
        if not new_example:
            logger.warning(f"Missing new_example in request body {body}")
            return {"message": f"Missing new_example in request body {body}", "status": "error"}
        model = MODEL
        if MODEL is None:
            logger.warning("Model not loaded. Please call /get_model first.")
            return {"message": "Model not loaded. Please call /get_model first.", "status": "error"}

        # read model from file

        # Classify the new example
        predicted_class = Classifier.classify_record(new_example, model)

        # return the predicted class (if classified successfully)
        if predicted_class:
            logger.info(f"Classification successful. Predicted class: {predicted_class}")
            return {"predicted_class": predicted_class, "status": "success"}
        else:
            logger.warning("Classification failed.")
            return {"message": "Classification failed.", "status": "error"}

    except Exception as e:
        logger.error(f"An error occurred during classification: {e}")
        return {"message": f"An error occurred during classification: {e}", "status": "error"}
