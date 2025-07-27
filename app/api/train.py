from fastapi import APIRouter, Request
from app.core.trainer import Trainer
from app.core.validator import Validator
from app.core.cleaner import Cleaner
import pandas as pd
import os
import tempfile
import pickle
import logging
from app.api.server import MODEL, temp_files, logger, _data_dir

router = APIRouter()

@router.get('/load_hardcoded_data')
async def load_hardcoded_data():
    logger.info("/load_hardcoded_data endpoint called.")
    try:
        file_name = 'PlayTennis.csv'
        if not os.path.exists(os.path.join(_data_dir, file_name)):
            logger.warning(f"File {file_name} not found in {_data_dir}.")
            return {"message": f"File {file_name} not found.", "status": "error"}
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

@router.post('/train')
async def train_model(request: Request):
    logger.info("/train endpoint called.")
    try:
        body = await request.json()
        data = body["data"]
        df = pd.DataFrame(data)
        trainer = Trainer()
        trained_model = trainer.build_model(df)
        if not trained_model:
            logger.warning("Model training failed, no model returned.")
            return {"message": "Model training failed, no model returned.", "status": "error"}
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

@router.post('/test_model')
async def test_model(request: Request):
    logger.info("/test_model endpoint called.")
    try:
        body = await request.json()
        if not body:
            logger.warning("Request body is empty in /test_model.")
            return {"message": "Request body is empty.", "status": "error"}
        json_data = body.get('data')
        data = pd.DataFrame(json_data)
        if data.empty:
            logger.warning("No data provided for testing the model in /test_model.")
            return {"message": "No data provided for testing the model.", "status": "error"}
        accuracy = Validator.validate_model_accuracy(MODEL, data)
        if accuracy is None:
            logger.warning("Model accuracy was not found.")
            return {"message": "Model accuracy was not found.", "status": "error"}
        logger.info(f"Model accuracy tested: {accuracy}")
        return {"Model_Accuracy": f"{accuracy}", 'status': 'success'}
    except Exception as e:
        logger.error(f"Exception in /test_model: {e}")
        return {"message": f"An error occurred: {e}", "status": "error"}

@router.post('/clean_csv')
async def clean_csv_file(request: Request):
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

@router.post('/load_data')
async def load_data(request: Request):
    logger.info("/load_data endpoint called.")
    try:
        body = await request.json()
        path = body.get('path')
        if not path:
            logger.warning("No path was given in /load_data.")
            return {'message': "no path was given", "status": 'error'}
        if not os.path.exists(path):
            logger.warning(f"Path '{path}' not found in /load_data.")
            return {"message": f"Path '{path}' not found.", "status": "error"}
        data = pd.read_csv(path)
        data = data.to_dict(orient='records')
        logger.info(f"Loaded data from {path}.")
        return {"data": data, "status": "success"}
    except Exception as e:
        logger.error(f"Exception in /load_data: {e}")
        return {"message": f"An error occurred: {e}", "status": "error"}

@router.post('/cache_model')
async def save_model(request: Request):
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