import requests

from app.api.client import Client
from app.core.trainer import Trainer


class TrainingManager:

    def __init__(self):
        self.model_builder = Trainer()
        self.client = Client()


    def process_new_model(self, training_df, testing_df):
        """Send the data for training, and save the model."""

        # convert the training and testing data from df to dict
        training_data = training_df.to_dict(orient="records")
        testing_df = testing_df.to_dict(orient="records")

        # send data for training
        if self.train_model_via_server(training_data):
            # get model accuracy
            body = self.client.test_model(testing_df)
            # check if the response is successful
            if body.get("status") == "success":
                accuracy = body.get("Model_Accuracy")
                if accuracy is not None:
                    print(f"Model Accuracy: {float(accuracy):.2f}%\n")
                    return accuracy
                else:
                    print("Model Accuracy: N/A (test failed)\n")
            else:
                print("Error testing model:", body.get("message"))
                return None

    def train_model_via_server(self, training_data):
        """
        Send training data to the server for training.
        """
        try:
            # initialize the request body with training data
            req_body = {"data": training_data}

            # send the request to the server for training
            res_body = self.client.send_for_training(req_body)
            # check if the response is successful
            if res_body.get("status") == "success":
                print("\nModel trained successfully via server.")
                return True
            else:
                print("Failed to train model via server:", res_body.get("message"))
                return False

        except Exception as e:
            print(f"An error occurred while training the model via server: {e}")
        return False




