import pandas as pd

from building_app.controller.training_manager import TrainingManager
from building_app.api.client import Client


class Manager:
    def __init__(self):
        self.training_manager = TrainingManager()
        self.client = Client()


    def start(self):
        """Start model process handling."""
        self.model_handling()


    def model_handling(self):
        # Load hardcoded data from the server.
        body = self.client.load_hardcoded_data()
        # Check if the response is successful
        if body.get("status") == "success":
            # Convert the data to a DataFrame
            df = pd.DataFrame.from_dict(body.get("data"))
            if not df.empty:
                # Split the data set for training and testing
                training_df, testing_df = self.split_the_data_set(df)
                # Send the data for training and testing
                accuracy = self.training_manager.process_new_model(training_df, testing_df)
            else:
                print("No data found in the hardcoded dataset.")
                return

        else:
            print("Error loading hardcoded data:", body.get("message"))
            return



    @staticmethod
    def split_the_data_set(df):
        """
        Randomly split the DataFrame into 70% training and 30% testing sets.
        """
        df_shuffled = df.sample(frac=1, random_state=420).reset_index(drop=True)  # Shuffle the DataFrame
        split_index = int(len(df_shuffled) * 0.70)
        training_df = df_shuffled.iloc[:split_index]
        checking_df = df_shuffled.iloc[split_index:]
        return training_df, checking_df

