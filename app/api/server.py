from fastapi import FastAPI, Request
import pandas as pd
import os
import json

from app.core.trainer import Trainer
from app.core.classifier import Classifier


app = FastAPI()


@app.post('/load_data')
async def load_data(request: Request):
    try:
        body = await request.json()
        path = body.get('path')

        # check if the path given is not none
        if not path:
            return {'message': "no path was given", "status": 'error'}

        # check if the path exists
        if not os.path.exists(path):
            return {"message": f"Path '{path}' not found.", "status": "error"}

        with open(path, 'r') as f:
            data = pd.read_csv(path)

        # convert from df to dict
        data = data.to_dict()

        return {"data": data, "status": "success"}

    except Exception as e:
        print(f"Exception: {e}")
        return {"message": f"An error occurred: {e}", "status": "error"}


@app.post('/train')
async def train_model(request: Request):
    """
    Trains a Naive Bayes model using JSON data.
    Also saves the model as a JSON file in the models/directory, if it doesn't already exist.
    """
    try:
        body = await request.json()
        model_name = body["model_name"]
        data = body["data"]
        df = pd.DataFrame(data)

        # Initialize and build the model
        trainer = Trainer()
        trained_model = trainer.build_model(df, model_name)

        # Prepare file path
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        model_file_path = os.path.join(models_dir, f"{model_name}.json")

        # Check if a model with the same name already exists
        if os.path.exists(model_file_path):
            return {"model": trained_model, "status": 'returned existing model.'}

        # Save the model to a JSON file
        with open(model_file_path, 'w', encoding='utf-8') as f:
            json.dump(trained_model, f, ensure_ascii=False, indent=4)

        return {"model": trained_model, "status": 'success'}
    except Exception as e:
        return {"message": f"An error occurred during training: {e}", "status": "error"}


@app.post('/classify')
async def classify_data(request: Request):
        """
        Classifies a new data example using the given model name.
        """
        try:
            body = await request.json()
            new_example = body.get("new_example")
            model_name = body.get("model_name")

            # check that there are no missing data
            if not new_example or not model_name:
                return {"message": "Missing new_example or model_name in request body", "status": "error"}

            # read model from file
            model = read_file_from_model_folder(model_name)

            # Classify the new example
            predicted_class = Classifier.classify_record(new_example, model_name, model)

            # return the predicted class (if classified successfully)
            if predicted_class:
                return {"predicted_class": predicted_class, "status": "success"}
            else:
                return {"message": "Classification failed.", "status": "error"}

        except Exception as e:
            return {"message": f"An error occurred during classification: {e}", "status": "error"}


@app.post('/test_model')
async def test_model(request: Request):
    """Test a given models accuracy with a given data set tester"""
    try:
        body = await request.json()
        model_name = body.get('model_name')
        json_data = body.get('data')
        data = pd.DataFrame(json_data)

        # read model from file
        model = read_file_from_model_folder(model_name)

        if model:
            correct = 0
            total = len(data)
            label_col = data.columns[-1]  # assuming last column is the target label
            label_col = str(label_col)  # Ensure label_col is a string for drop

            for _, row in data.iterrows():
                features = row.drop(label_col).to_dict()
                actual = row[label_col]  # get the actual target class
                predicted_result = Classifier.classify_record(features, model_name, model)
                if predicted_result:
                    predicted_class = predicted_result.get('class')  # Get the predicted class name
                    # compare the actual target class and the predicted one
                    if predicted_class == actual:
                        correct += 1

            # calculate in percent the percent the model predicted a right answer
            accuracy = (correct / total) * 100 if total > 0 else 0
            return {"Model_Accuracy": f"{accuracy}", 'status': 'success'}

    except OSError as e:
        print(f"OSError: {e}")
        return {"message": f"An error occurred trying to get model paths {e}", 'status': 'error'}

    except Exception as e:
        print(f"Exception: {e}")
        return {"message": f"An error occurred: {e}", "status": "error"}


@app.get('/get_models_file_paths')
async def get_all_models_file_paths():
    """Returns all the saved models file paths.
       In format {model_name: model_path}"""
    try:
        # the folder that contains all the saved models
        folder_path = r"/models"

        # return all the existing model names and paths
        return {
            os.path.splitext(file)[0]: os.path.join(folder_path, file)
            for file in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, file))
        }
    except OSError as e:
        return {"message": f"An error occurred trying to get model paths {e}", 'status': 'error'}


@app.post('/get_model_by_name')
async def get_model_by_name(request: Request):
    """Return the model by its given name."""
    try:
        body = await request.json()
        model_name = body.get("model_name")

        model = read_file_from_model_folder(model_name)

        return {'model': model, 'status': 'success'}

    except Exception as e:
        return {"message": f"An error occurred: {e}", "status": "error"}


def read_file_from_model_folder(model_name):
    """Read a file from the 'models' folder according to a given name"""
    try:
        # initialize the model file path
        model_file_path = f"models/{model_name}.json"
        # check that the path exists
        if not os.path.exists(model_file_path):
            return {"message": f"Model '{model_name}' not found.", "status": "error"}
        # read the model from the file
        with open(model_file_path, 'r') as f:
            model = json.load(f)

        return model

    except OSError as e:
        return {"message": f"An error occurred trying to get model paths {e}", 'status': 'error'}
    except Exception as e:
        return {"message": f"An error occurred: {e}", "status": "error"}

@app.get('/')
async def root():
    """Basic message for an empty api call"""
    return {"message": "Naive Bayes Classifier API. Use /train to train a model and /classify to classify data."}