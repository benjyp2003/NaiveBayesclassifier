from fastapi import FastAPI, Request
import tempfile
import pickle
import threading
import pandas as pd
import os
import logging

from building_app.core.trainer import Trainer
from building_app.core.validator import Validator
from building_app.main import run_automated_client_task

app = FastAPI()

temp_files = []
MODEL = None
_data_dir = "building_app/data"

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def on_startup():
    """When the server starts, run the client logic in a background thread."""
    logger.info("Server startup: running automated client task in background thread.")
    thread = threading.Thread(target=run_automated_client_task)
    thread.start()


@app.get('/load_hardcoded_data')
async def load_hardcoded_data():
    """
    Returns a hardcoded dataset in the container..
    """
    logger.info("/load_hardcoded_data endpoint called.")
    try:
        file_name = 'PlayTennis.csv'
        df = pd.read_csv(os.path.join(_data_dir, file_name))

        data = df.to_dict(orient='records')

        if not data:
            logger.warning(f"No data found in the {file_name} dataset.")
            return {"message": f"No data found in the {file_name} dataset.", "status": "error"}
        logger.info(f"Loaded hardcoded data from {file_name}.")
        return {"data": data, "status": "success"}
    except Exception as e:
        logger.error(f"Error in /load_hardcoded_data: {e}")
        return {"message": f"An error occurred: {e}", "status": "error"}


@app.on_event("shutdown")
async def cleanup_temp_files():
    logger.info("Server shutdown: cleaning up temp files.")
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)
            logger.info(f"Deleted temp file: {f}")


@app.get("/get_model")
async def get_model():
    """"Returns the saved model."""
    logger.info("/get_model endpoint called.")
    try:
        if not MODEL:
            logger.warning("No model has been trained.")
            return {"message": "No model has been trained.", "status": "error"}
        logger.info("Model returned successfully.")
        return {"model": MODEL, "status": "success"}

    except Exception as e:
        logger.error(f"Error in /get_model: {e}")
        return {"message": f"An error occurred: {e}", "status": "error"}




@app.post('/train')
async def train_model(request: Request):
    """
    Trains a Naive Bayes model using JSON data.
    Also saves the model as a JSON file in the models/directory, if it doesn't already exist.
    """
    logger.info("/train endpoint called.")
    try:
        body = await request.json()
        data = body["data"]
        df = pd.DataFrame(data)

        # Initialize and build the model
        trainer = Trainer()
        trained_model = trainer.build_model(df)

        with tempfile.NamedTemporaryFile(mode='wb+', delete=False) as tmp:
            pickle.dump(trained_model, tmp)
            temp_files.append(tmp.name)

        global MODEL
        MODEL = trained_model
        logger.info("Model trained and saved to temporary file.")
        return {"message": "model was trained and saved to a temporary file.", "status": 'success'}
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return {"message": f"An error occurred during training: {e}", "status": "error"}


@app.post('/test_model')
async def test_model(request: Request):
    """Test a given models accuracy with a given data set tester"""
    logger.info("/test_model endpoint called.")
    try:
        body = await request.json()
        json_data = body.get('data')
        data = pd.DataFrame(json_data)

        # get model accuracy
        accuracy = Validator.validate_model_accuracy(MODEL, data)
        logger.info(f"Model accuracy tested: {accuracy}")

        return {"Model_Accuracy": f"{accuracy}", 'status': 'success'}

    except OSError as e:
        logger.error(f"OSError in /test_model: {e}")
        return {"message": f"An error occurred trying to get model paths {e}", 'status': 'error'}

    except Exception as e:
        logger.error(f"Exception in /test_model: {e}")
        return {"message": f"An error occurred: {e}", "status": "error"}


@app.post('/load_data')
async def load_data(request: Request):
    logger.info("/load_data endpoint called.")
    try:
        body = await request.json()
        path = body.get('path')

        # check if the path given is not none
        if not path:
            logger.warning("No path was given in /load_data.")
            return {'message': "no path was given", "status": 'error'}

        # check if the path exists
        if not os.path.exists(path):
            logger.warning(f"Path '{path}' not found in /load_data.")
            return {"message": f"Path '{path}' not found.", "status": "error"}

        data = pd.read_csv(path)

        # convert from df to dict
        data = data.to_dict(orient='records')
        logger.info(f"Loaded data from {path}.")

        return {"data": data, "status": "success"}

    except Exception as e:
        logger.error(f"Exception in /load_data: {e}")
        return {"message": f"An error occurred: {e}", "status": "error"}

@app.post("/cache_model")
async def save_model(request: Request):
    """
    Save a model to a temporary file and return its path.
    """
    logger.info("/cache_model endpoint called.")
    try:
        body = await request.json()
        if not body:
            logger.warning("Request body is empty in /cache_model.")
            return {"message": "Request body is empty.", "status": "error"}

        model_data = body.get("model")

        if not model_data:
            logger.warning("Model data is missing in /cache_model.")
            return {"message": "Model data is missing.", "status": "error"}

        with tempfile.NamedTemporaryFile(mode='wb+', delete=False) as tmp:
            pickle.dump(model_data, tmp)
            temp_files.append(tmp.name)
            logger.info(f"Model cached to temp file: {tmp.name}")
            return {"path": tmp.name, "status": "success"}

    except Exception as e:
        logger.error(f"Error in /cache_model: {e}")
        return {"message": f"An error occurred: {e}", "status": "error"}

@app.get('/')
async def root():
    """Basic message for an empty api call"""
    logger.info("Root endpoint called.")
    return {"message": "Naive Bayes Classifier API. Use /train to train a model and /classify to classify data."}