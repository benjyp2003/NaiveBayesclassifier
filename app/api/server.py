from fastapi import FastAPI, Request
import tempfile
import pickle
import threading
import pandas as pd
import os
import logging

from app.core.trainer import Trainer
from app.core.validator import Validator
from app.core.cleaner import Cleaner
from app.main import run_automated_client_task

app = FastAPI()

temp_files = []
MODEL = None
_data_dir = "app/data"

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def on_startup():
    """When the server starts, automatically run the client logic in a background thread."""
    logger.info("Server startup: running automated client task in background thread.")
    thread = threading.Thread(target=run_automated_client_task)
    thread.start()


@app.get('/health')
async def health_check():
    """
    Health check endpoint for the builder server, that checks that the server is up.
    """
    logger.info("/health endpoint called.")
    try:
        health_status = {
            "server": "healthy",
            "model_available": MODEL is not None,
        }
        
        return {
            "status": "healthy",
            "details": health_status
        }
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "unhealthy",
            "details": {
                "server": "unhealthy",
                "error": str(e)
            }
        }


@app.get('/load_hardcoded_data')
async def load_hardcoded_data():
    """
    Returns a hardcoded dataset in the container..
    """
    logger.info("/load_hardcoded_data endpoint called.")
    try:
        file_name = 'PlayTennis.csv'
        # Check if the file exists in the data directory
        if not os.path.exists(os.path.join(_data_dir, file_name)):
            logger.warning(f"File {file_name} not found in {_data_dir}.")
            return {"message": f"File {file_name} not found.", "status": "error"}

        # Load the dataset
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
    """Clean up temporary files when the server shuts down."""
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
        # check that the model is not None
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
    """ Trains a Naive Bayes model using JSON data. """

    logger.info("/train endpoint called.")
    try:
        body = await request.json()
        data = body["data"]
        df = pd.DataFrame(data)

        # Initialize and build the model
        trainer = Trainer()
        trained_model = trainer.build_model(df)

        # check if the model was trained successfully
        if not trained_model:
            logger.warning("Model training failed, no model returned.")
            return {"message": "Model training failed, no model returned.", "status": "error"}

        # Save the model to a temporary file
        with tempfile.NamedTemporaryFile(mode='wb+', delete=False) as tmp:
            pickle.dump(trained_model, tmp)
            temp_files.append(tmp.name)

        # initialize the global MODEL variable with the trained model
        global MODEL
        MODEL = trained_model

        logger.info("Model trained and saved to temporary file.")
        return {"message": "model was trained and saved to a temporary file.", "status": 'success'}
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return {"message": f"An error occurred during training: {e}", "status": "error"}


@app.post('/test_model')
async def test_model(request: Request):
    """Test the saved model accuracy with a given data set tester"""
    logger.info("/test_model endpoint called.")
    try:
        # get the data from the request body
        body = await request.json()
        if not body:
            logger.warning("Request body is empty in /test_model.")
            return {"message": "Request body is empty.", "status": "error"}
        json_data = body.get('data')

        # convert the json data to a DataFrame
        data = pd.DataFrame(json_data)
        if data.empty:
            logger.warning("No data provided for testing the model in /test_model.")
            return {"message": "No data provided for testing the model.", "status": "error"}

        # get model accuracy
        accuracy = Validator.validate_model_accuracy(MODEL, data)

        # check if the accuracy is None
        if accuracy is None:
            logger.warning("Model accuracy was not found.")
            return {"message": "Model accuracy was not found.", "status": "error"}

        logger.info(f"Model accuracy tested: {accuracy}")
        return {"Model_Accuracy": f"{accuracy}", 'status': 'success'}

    except OSError as e:
        logger.error(f"OSError in /test_model: {e}")
        return {"message": f"An error occurred trying to get model paths {e}", 'status': 'error'}

    except Exception as e:
        logger.error(f"Exception in /test_model: {e}")
        return {"message": f"An error occurred: {e}", "status": "error"}




@app.post('/clean_csv')
async def clean_csv_file(request: Request):
    """
    Clean a CSV file and return the cleaned data.
    """
    logger.info("/clean_csv endpoint called.")
    try:
        body = await request.json()
        file_path = body.get('file_path')
        remove_duplicates = body.get('remove_duplicates', True)
        handle_missing = body.get('handle_missing', 'drop')  # 'drop', 'fill', 'interpolate'
        fill_value = body.get('fill_value', None)
        normalize_columns = body.get('normalize_columns', True)
        remove_special_chars = body.get('remove_special_chars', True)
        strip_whitespace = body.get('strip_whitespace', True)
        
        if not file_path:
            logger.warning("No file_path provided in /clean_csv.")
            return {"message": "No file_path provided", "status": "error"}
        
        if not os.path.exists(file_path):
            logger.warning(f"File path '{file_path}' not found in /clean_csv.")
            return {"message": f"File path '{file_path}' not found.", "status": "error"}
        
        # Initialize cleaner and clean the CSV
        cleaner = Cleaner()
        cleaned_df = cleaner.clean_csv_to_df(
            file_path=file_path,
            remove_duplicates=remove_duplicates,
            handle_missing=handle_missing,
            fill_value=fill_value,
            normalize_columns=normalize_columns,
            remove_special_chars=remove_special_chars,
            strip_whitespace=strip_whitespace
        )
        
        # Convert to dictionary for JSON response
        cleaned_data = cleaned_df.to_dict(orient='records')
        cleaning_summary = cleaner.get_cleaning_summary()
        
        logger.info(f"Successfully cleaned CSV file: {file_path}")
        return {
            "data": cleaned_data,
            "cleaning_summary": cleaning_summary,
            "shape": cleaned_df.shape,
            "columns": list(cleaned_df.columns),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Exception in /clean_csv: {e}")
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