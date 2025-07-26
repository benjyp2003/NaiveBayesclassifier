import requests

from app.controller.classifying_manager import ClassifyingManager
from app.controller.training_manager import TrainingManager
from app.data.loader import DataLoader
from app.ui.cli import CLI


class Manager:
    def __init__(self):
        self.training_manager = TrainingManager()
        self.classifying_manager = ClassifyingManager()


    def start(self):
        """Start the main menu loop."""
        self.handle_menu_choice()


    def handle_menu_choice(self):
        """Display menu and handle user choices for training, classifying, or exiting."""
        exit = False
        while not exit:
            CLI.show_menu()
            choice = input('>>> ')
            match choice:
                # User chose to trian a new model
                case '1':
                    # Load a given data set
                    data_set = DataLoader.load_data_by_file_path()
                    # Split the data set for training and testing
                    training_df, testing_df = self.split_the_data_set(data_set)
                    # Send the data for training and testing
                    self.training_manager.process_new_model(training_df, testing_df)

                # User chose to classify an example
                case '2':
                    self.classifying_manager.classify_new_example(self.get_all_models_file_paths())

                #User chose to exit
                case '3':
                    exit = True

                case _:
                    print('Invalid choice.\n')


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


    @staticmethod
    def get_all_models_file_paths():
        """Get all paths for the saved models JSON files"""
        try:
            response = requests.get('http://127.0.0.1:8000/get_models_file_paths')
            response.raise_for_status()
            model_names_and_paths = response.json()
            return model_names_and_paths

        except requests.exceptions.RequestException as e:
            print(f'API request error: {e}')
            return None
        except OSError as e:
            print("Error getting models file path:", e)
            return None

