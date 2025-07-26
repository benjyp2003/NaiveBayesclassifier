import requests


class ClassifyingManager:

    def classify_new_example(self, model_file_paths):
        # get all available models names
        all_model_names = self.get_all_available_models_names(model_file_paths)

        # let the user choose a model from the available models.
        model_name = self.choose_model(all_model_names)
        if model_name is None:
            print("\nNo model selected, returning..")
            return

        # Get data to classify from user
        new_data = self.get_data_from_user_for_classifying(model_name)
        if new_data is None:
            print("Failed to get data from user. \n")
            return

        # Send the data for classifying
        self.send_new_data_for_classifying(new_data, model_name)


    def send_new_data_for_classifying(self, new_data: dict, model_name):
        try:
            # send the example to the server for classifying
            response = requests.post("http://127.0.0.1:8000/classify", json={"new_example": new_data, "model_name": model_name})
            response.raise_for_status()  # Raise an exception for HTTP errors
            predicted = response.json()
            if predicted.get("status") == "success":
                # show the predicted class and the percentage
                print(f"Predicted Class: {predicted.get('predicted_class')}")
            else:
                print(f"Error: {predicted.get('message')}")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")


    def get_data_from_user_for_classifying(self, model_name):
        """Get from user values for each feature column and return as a dict."""
        user_data = {}
        try:
            response = requests.post("http://127.0.0.1:8000/get_model_by_name", json={'model_name': model_name}, timeout=10)
            response.raise_for_status()
            model = response.json()
            if model.get("status") == "success":
                model_data = model.get('model').get(model_name)
                likelihoods: dict[str: dict[str: any]] = list(model_data.get('likelihoods').values())[0]
            else:
                print(model.get("message"))
                return None
            if not likelihoods:
                return None

            # run on all the columns in the likelihoods
            for column_name, features in likelihoods.items():
                print(f"\nSelect a value for '{column_name}':")

                for idx, feature in enumerate(features, 1):
                    print(f"  {idx}. {feature}")  # print every columns values.

                while True:
                    choice = input(f"Enter the number (1-{len(features)}) or 'q' to quit: ")
                    if choice.lower() == 'q':
                        return None
                    
                    try:
                        choice_num = int(choice)
                        if 1 <= choice_num <= len(features):
                            user_data[column_name] = str(list(features.keys())[choice_num - 1])
                            break
                        else:
                            print("Invalid choice. Please try again.")
                    except ValueError:
                        print("Please enter a valid number or 'q' to quit.")

            print("\nYou entered:", user_data)
            return user_data

        except requests.exceptions.RequestException as e:
            print(f'API request error: {e}')
            return None
        except ValueError as e:
            print('An error accord', e)


    def choose_model(self, available_model_names: list[str]):
            # check if there is available models and print them
            if self.print_all_available_models(available_model_names):
                while True:
                    choice = input(f"Choose model (1 - {len(available_model_names)}) or 'q' to quit: \n")
                    if choice.lower() == 'q':
                        return None

                    try:
                        choice_num = int(choice)
                        if 1 <= choice_num <= len(available_model_names):
                            selected_model = available_model_names[choice_num-1]
                            return selected_model
                        else:
                            print(f'Invalid choice. Please enter a number between 1 and {len(available_model_names)}.')
                    except ValueError:
                        print('Please enter a valid number or "q" to quit.')
            else:
                print("DEBUG: No models available")
                return None


    @staticmethod
    def print_all_available_models(available_model_names):
        # check if there are no available models
        if available_model_names == []:
            print('No models available.')
            return False
        else:
            count = 1
            print("\nAvailable models:")
            for name in available_model_names:
                print(f"{count}: {name}")
                count += 1
            return True


    @staticmethod
    def get_all_available_models_names(model_file_paths):
        return list(model_file_paths.keys())