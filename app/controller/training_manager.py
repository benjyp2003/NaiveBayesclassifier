import requests
import pandas

from app.core.trainer import Trainer


class TrainingManager:

    def __init__(self):
        self.model_builder = Trainer()


    def process_new_model(self, training_df, testing_df):
        """Send the data for training, and save the model."""
        # get a name for the model
        model_name = self.get_name_for_model()
        # convert the training data from df to dict
        training_data = training_df.to_dict(orient="records")
        # send data for training
        model =  self.train_model_via_server(training_data, model_name)

        if model is None:
            print("Failed to train model. Cannot check accuracy.")
            return

        # get model accuracy
        accuracy = self.check_model_accuracy(model_name, testing_df)
        if accuracy is not None:
            print(f"Model Accuracy: {float(accuracy):.2f}%\n")
        else:
            print("Model Accuracy: N/A (test failed)\n")


    @staticmethod
    def train_model_via_server(training_data, model_name):
        """
        Send training data and model name to the server for training and return the trained model.
        """
        try:
            # send data and model name to the server
            response = requests.post("http://127.0.0.1:8000/train",
                                     json={"model_name": model_name, "data": training_data},
                                     timeout=20)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success":
                model = data.get("model")
                print("\nModel trained successfully via server.")
                return model
            elif data.get("status") == 'returned existing model.':
                model = data.get("model")
                print(f"\nGot existing model named '{model_name}'.")
                return model

        except requests.exceptions.Timeout as e:
            print(f"API request timed out: {e}")
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
        except OSError as e:
            print(f"File error: {e}")
        return None


    @staticmethod
    def get_name_for_model():
        model_name = input('Enter the name you want to call the model: ')
        return model_name


    @staticmethod
    def check_model_accuracy(model_name, test_df: pandas.DataFrame):
        """Evaluate the core's accuracy on the test set and print the result."""
        try:
            # convert test_data from df to dict
            test_data = test_df.to_dict(orient="records")
            response = requests.post(
                "http://127.0.0.1:8000/test_model",
                json={"model_name": model_name, "data": test_data},
                timeout=15
            )
            response.raise_for_status()
            accuracy = response.json()
            if accuracy.get("status") == "success":
                return accuracy.get("Model_Accuracy")
            else:
                return 0
        except requests.exceptions.Timeout as e:
            print(f"API request timed out: {e}")
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
        except OSError as e:
            print(f"File error: {e}")
        return None





    # @staticmethod
    # def save_model_to_file(model_name, model):
    #     """Save the model to a json file, and return the path of the file."""
    #     try:
    #         path = f"models/{model_name}.json"
    #         with open(path, "w", encoding="utf-8") as f:
    #             json.dump(model, f, indent=4)
    #
    #         return path
    #     except OSError as e:
    #         print('An error occurred while saving model to file: \n', e)


    # def check_if_model_exists(self, test_model):
    #     """Returns the existing model if model on the given data already exists., else False"""
    #     try:
    #         for file_path in list(self.__all_models_file_paths.values()):
    #             with open(file_path, 'r') as f:
    #                 model = json.load(f)
    #             vals_model = model.values()
    #             vals_test_model = test_model.values()
    #             if list(vals_model) == list(vals_test_model):
    #                 return model
    #         return False
    #     except OSError as e:
    #        print('An error occurred while reading the models from file: \n', e)


    # def train_locally(self, model_name, training_df, testing_df):
    #     model = self.train_new_model(training_df, model_name)
    #     check = self.check_if_model_exists(model)
    #     if check:
    #         path =  self.save_model_to_file(model_name, model)
    #         self.__all_models_file_paths.append(path)
    #         print(f"\nModel '{model_name}' was built and saved successfully.")
    #         self.check_model_accuracy(model_name, model, testing_df)
    #
    #     else:
    #         print('Model on this data already exists.')
    #         return
    #
    # def train_new_model(self, data, name):
    #     model = self.model_builder.build_model(data, name)
    #     return model

