import pandas as pd

class DataLoader:

    @staticmethod
    def load_data_by_file_path():
        while True:
            try:
                path = DataLoader.get_path()
                df = pd.read_csv(path)
                return df
            except:
                print("Path does not exist. \n")

    @staticmethod
    def get_path():
        """Prompt the user to enter a file path and return it."""
        path = input('Enter file path: ')
        return path