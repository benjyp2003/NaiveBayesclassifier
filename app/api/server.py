from fastapi import FastAPI, Request
import logging
import threading
import tempfile
import pickle
import pandas as pd
import os

from app.main import run_automated_client_task
from app.core.trainer import Trainer
from app.core.validator import Validator
from app.core.cleaner import Cleaner
from app.core.classifier import Classifier
app = FastAPI()

temp_files = []
MODEL = None
_data_dir = "app/data"

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def on_startup():
    logger.info("Server startup: running automated client task in background thread.")
    thread = threading.Thread(target=run_automated_client_task)
    thread.start()


@app.get('/load_hardcoded_data')
async def load_hardcoded_data():
    """Load hardcoded data from a CSV file."""
    logger.info("/load_hardcoded_data endpoint called.")
    try:
        file_name = 'PlayTennis.csv'
        if not os.path.exists(os.path.join(_data_dir, file_name)):
            logger.warning(f"File {file_name} not found in {_data_dir}.")
            return {"message": f"File {file_name} not found.", "status": "error"}

        # load the CSV file
        df = pd.read_csv(os.path.join(_data_dir, file_name))
        # convert DataFrame to a list of dictionaries
        data_dict = df.to_dict(orient='records')
        if not data_dict:
            logger.warning(f"No data found in the {file_name} dataset.")
            return {"message": f"No data found in the {file_name} dataset.", "status": "error"}

        logger.info(f"Loaded hardcoded data from {file_name}.")
        return {"data": data_dict, "status": "success"}
    except Exception as e:
        logger.error(f"Error in /load_hardcoded_data: {e}")
        return {"message": f"An error occurred: {e}", "status": "error"}

@app.post('/train')
async def train_model(request: Request):
    """Train a model using the provided data."""
    logger.info("/train endpoint called.")
    try:
        body = await request.json()
        data = body["data"]
        # check if data is provided
        if not data:
            logger.warning("No data provided for training in /train.")
            return {"message": "No data provided for training.", "status": "error"}

        # convert the data to a DataFrame
        df = pd.DataFrame(data)

        # send the DataFrame to the Trainer for building
        trainer = Trainer()
        trained_model = trainer.build_model(df)

        # check if the model was trained successfully
        if not trained_model:
            logger.warning("Model training failed, no model returned.")
            return {"message": "Model training failed, no model returned.", "status": "error"}

        # save the trained model to a temporary file
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
    """Test the model's accuracy with a provided testing dataset."""
    logger.info("/test_model endpoint called.")
    try:
        body = await request.json()
        # check if the request body is empty
        if not body:
            logger.warning("Request body is empty in /test_model.")
            return {"message": "Request body is empty.", "status": "error"}

        json_data = body.get('data')
        data = pd.DataFrame(json_data)
        if data.empty:
            logger.warning("No data provided for testing the model in /test_model.")
            return {"message": "No data provided for testing the model.", "status": "error"}

        # get the model's accuracy using the Validator
        accuracy = Validator.validate_model_accuracy(MODEL, data)
        if accuracy is None:
            logger.warning("Model accuracy was not found.")
            return {"message": "Model accuracy was not found.", "status": "error"}

        logger.info(f"Model accuracy tested: {accuracy}")
        return {"Model_Accuracy": f"{accuracy}", 'status': 'success'}
    except Exception as e:
        logger.error(f"Exception in /test_model: {e}")
        return {"message": f"An error occurred: {e}", "status": "error"}


@app.get('/get_model')
async def get_model():
    """Return the currently trained model."""
    logger.info("/get_model endpoint called.")
    try:
        # check if the model is loaded
        if not MODEL:
            logger.warning("No model has been trained.")
            return {"message": "No model has been trained.", "status": "error"}
        logger.info("Model returned successfully.")
        return {"model": MODEL, "status": "success"}
    except Exception as e:
        logger.error(f"Error in /get_model: {e}")
        return {"message": f"An error occurred: {e}", "status": "error"}

@app.post('/classify')
async def classify_data(request: Request):
    """Classify a new example using the trained model."""
    logger.info("/classify endpoint called.")
    try:
        body = await request.json()
        new_example = body.get("new_example")
        logger.info(f"Received classification request: {body}")
        if not new_example:
            logger.warning(f"Missing new_example in request body {body}")
            return {"message": f"Missing new_example in request body {body}", "status": "error"}

        model = MODEL
        # check if the model is loaded
        if MODEL is None:
            logger.warning("Model not loaded. Please call /get_model first.")
            return {"message": "Model not loaded. Please call /get_model first.", "status": "error"}

        # get the predicted class using the Classifier
        predicted_class = Classifier.classify_record(new_example, model)
        if predicted_class:
            logger.info(f"Classification successful. Predicted class: {predicted_class}")
            return {"predicted_class": predicted_class, "status": "success"}
        else:
            logger.warning("Classification failed.")
            return {"message": "Classification failed.", "status": "error"}
    except Exception as e:
        logger.error(f"An error occurred during classification: {e}")
        return {"message": f"An error occurred during classification: {e}", "status": "error"}


@app.post('/clean_csv')
async def clean_csv_file(request: Request):
    """Clean a CSV file and return the cleaned data."""
    logger.info("/clean_csv endpoint called.")
    try:
        body = await request.json()
        file_path = body.get('file_path')
        remove_duplicates = body.get('remove_duplicates', True)
        handle_missing = body.get('handle_missing', 'drop')
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
    """Load data from a given path and return it as a list of dictionaries."""
    logger.info("/load_data endpoint called.")
    try:
        body = await request.json()
        path = body.get('path')
        if not path:
            logger.warning("No path was given in /load_data.")
            return {'message': "no path was given", "status": 'error'}

        # check if the path exists
        if not os.path.exists(path):
            logger.warning(f"Path '{path}' not found in /load_data.")
            return {"message": f"Path '{path}' not found.", "status": "error"}

        # load the CSV file
        data = pd.read_csv(path)
        data = data.to_dict(orient='records')
        logger.info(f"Loaded data from {path}.")
        return {"data": data, "status": "success"}
    except Exception as e:
        logger.error(f"Exception in /load_data: {e}")
        return {"message": f"An error occurred: {e}", "status": "error"}


@app.post('/cache_model')
async def save_model(request: Request):
    """Cache the model temporarily on the server and return the path to the cached model."""
    logger.info("/cache_model endpoint called.")
    try:
        body = await request.json()
        # check if the request body is empty
        if not body:
            logger.warning("Request body is empty in /cache_model.")
            return {"message": "Request body is empty.", "status": "error"}

        model_data = body.get("model")
        # check if model data is provided
        if not model_data:
            logger.warning("Model data is missing in /cache_model.")
            return {"message": "Model data is missing.", "status": "error"}

        # save the model data to a temporary file
        with tempfile.NamedTemporaryFile(mode='wb+', delete=False) as tmp:
            pickle.dump(model_data, tmp)
            temp_files.append(tmp.name)
            logger.info(f"Model cached to temp file: {tmp.name}")
            return {"path": tmp.name, "status": "success"}
    except Exception as e:
        logger.error(f"Error in /cache_model: {e}")
        return {"message": f"An error occurred: {e}", "status": "error"}


@app.get('/list_datasets')
async def list_datasets():
    """List all available CSV datasets in the /data directory."""
    logger.info("/list_datasets endpoint called.")
    try:
        data_dir = _data_dir
        if not os.path.exists(data_dir):
            logger.warning(f"Data directory {data_dir} does not exist.")
            return {"datasets": [], "status": "error", "message": f"Data directory {data_dir} does not exist."}
        files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        logger.info(f"Found datasets: {files}")
        return {"datasets": files, "status": "success"}
    except Exception as e:
        logger.error(f"Error in /list_datasets: {e}")
        return {"datasets": [], "status": "error", "message": str(e)}

@app.get('/get_dataset_columns')
async def get_dataset_columns(dataset: str):
    """Return the columns/features of a given CSV dataset in /data."""
    logger.info(f"/get_dataset_columns endpoint called for dataset: {dataset}")
    try:
        if not _data_dir:
            logger.warning("Data directory is not set.")
            return {"columns": [], "status": "error", "message": "Data directory is not set."}
        data_dir = _data_dir

        # initialize the dataset path
        file_path = os.path.join(data_dir, dataset)
        if not os.path.exists(file_path):
            logger.warning(f"Dataset file {file_path} does not exist.")
            return {"columns": [], "status": "error", "message": f"Dataset file {file_path} does not exist."}

        # convert the dataset to a DataFrame and get the columns
        df = pd.read_csv(file_path, nrows=1)
        columns = list(df.columns)
        logger.info(f"Columns for {dataset}: {columns}")
        return {"columns": columns, "status": "success"}
    except Exception as e:
        logger.error(f"Error in /get_dataset_columns: {e}")
        return {"columns": [], "status": "error", "message": str(e)}


@app.on_event("shutdown")
async def cleanup_temp_files():
    """Clean up temporary files created during the server's lifetime."""
    logger.info("Server shutdown: cleaning up temp files.")
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)
            logger.info(f"Deleted temp file: {f}")

@app.get('/')
async def root():
    logger.info("Root endpoint called.")
    return {"message": "Naive Bayes Classifier API. Use /train to train a model and /classify to classify data."}