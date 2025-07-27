import logging
from fastapi import FastAPI, Request
from requests import RequestException

from classifying_app.api.client import Client
from app.core.classifier import Classifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

MODEL = None

@app.on_event("startup")
async def fetch_model_on_startup():
    """
    Fetch the model from app when the server starts.
    """
    global MODEL
    logger.info("Startup: Checking builder server health...")
    try:
        client = Client()
        
        if not builder_is_healthy():
            logger.warning("Startup: Builder server is not healthy. Cannot fetch model.")
            return
        
        logger.info("Startup: Builder server is healthy, fetching model...")
        result = client.get_model_from_trainer()
        if result.get("status") == "success":
            MODEL = result.get("model")
            logger.info("Startup: Model fetched and loaded successfully.")
        else:
            logger.warning(f"Startup: Failed to fetch model: {result.get('message', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Startup: Error fetching model: {e}")

@app.get('/health')
async def health_check():
    """
    Health check endpoint that verifies both classifier and builder server status.
    """
    logger.info("/health endpoint called.")
    try:
        client = Client()
        builder_health = client.check_builder_server_health(max_retries=1, retry_delay=1.0)
        
        health_status = {
            "classifier_server": "healthy",
            "model_loaded": MODEL is not None,
            "builder_server": builder_health.get("status"),
            "builder_message": builder_health.get("message")
        }
        
        # Overall status is healthy if classifier is up and model is loaded
        overall_status = "healthy" if MODEL is not None else "degraded"
        
        return {
            "status": overall_status,
            "details": health_status
        }
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "unhealthy",
            "details": {
                "classifier_server": "unhealthy",
                "error": str(e)
            }
        }

@app.get('/get_model')
async def get_model():
    """
    Fetches the model from the builder (app) and stores it locally.
    """
    global MODEL
    logger.info("/get_model endpoint called.")
    try:
        # Use the Docker network hostname for app
        client = Client(api_url='http://building_app:8000')
        
        # First check if the builder server is up
        logger.info("Checking builder server health...")
        health_check = client.check_builder_server_health()
        if health_check.get("status") != "success":
            logger.warning(f"Builder server health check failed: {health_check.get('message')}")
            return {"message": f"Builder server is not available: {health_check.get('message')}", "status": "error"}
        
        logger.info("Builder server is healthy, fetching model...")
        response = client.get_model_from_trainer()
        if response.get("status") == "success":
            MODEL = response.get("model")
            logger.info("Model fetched from builder and loaded successfully.")
            return {"message": "Model fetched and loaded successfully.", "model": MODEL, "status": "success"}
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
        
        # If model is not loaded, try to fetch it from builder
        if MODEL is None:
            logger.info("Model not loaded, attempting to fetch from builder...")
            client = Client()
            
            # Check if builder server is up
            health_check = client.check_builder_server_health()
            if health_check.get("status") != "success":
                logger.warning(f"Builder server not available: {health_check.get('message')}")
                return {"message": f"Model not loaded and builder server unavailable: {health_check.get('message')}", "status": "error"}
            
            # Try to fetch model
            result = client.get_model_from_trainer()
            if result.get("status") == "success":
                MODEL = result.get("model")
                logger.info("Model fetched successfully during classification request.")
            else:
                logger.warning(f"Failed to fetch model: {result.get('message', 'Unknown error')}")
                return {"message": "Model not loaded and failed to fetch from builder.", "status": "error"}
        
        model = MODEL
        if MODEL is None:
            logger.warning("Model not loaded. Please call /get_model first.")
            return {"message": "Model not loaded. Please call /get_model first.", "status": "error"}

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


def builder_is_healthy():
    """
    Check the health of the builder server.
    """
    client = Client()
    # First check if the builder server is up
    health_check = client.check_builder_server_health()
    if health_check.get("status") != "success":
        logger.warning(f"Startup: Builder server health check failed: {health_check.get('message')}")
        return False
    else:
        return True