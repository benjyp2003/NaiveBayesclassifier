class Validator:

    @staticmethod
    def validate_model_accuracy(model, test_data):
        if model:
            correct = 0
            total = len(test_data)
            label_col = test_data.columns[-1]  # assuming last column is the target label
            label_col = str(label_col)  # Ensure label_col is a string for drop

            for _, row in test_data.iterrows():
                features = row.drop(label_col).to_dict()
                actual = row[label_col]  # get the actual target class
                predicted_result = Validator.classify_record(features, model)
                if predicted_result:
                    predicted_class = predicted_result.get('class')  # Get the predicted class name
                    # compare the actual target class and the predicted one
                    if predicted_class == actual:
                        correct += 1

            # calculate in percent, the percent the model predicted a right answer
            accuracy = (correct / total) * 100 if total > 0 else 0
            return accuracy


    @staticmethod
    def classify_record(new_example, model):
        """Calculate the posterior probabilities for each class
        This involves multiplying the prior probability by the likelihood of the new example
        given each class."""
        try:
            # Extract the priors and likelihoods from the model
            priors = model["model"]['priors']
            likelihoods = model["model"]['likelihoods']
            mult_results = {}

            for class_label in likelihoods:
                # initialize the priors relations in the mult_results for multiplying later with the feature relations
                mult_results[class_label] = priors[class_label]

                for feature, feature_value in new_example.items():
                    feature_probs = likelihoods[class_label].get(feature, {})
                    prob = feature_probs.get(feature_value, 1)  # default to 1 if unseen
                    mult_results[class_label] *= prob  # multiply P(X|C)

            if mult_results:
                # find the max percent in the mult_result
                max_class = max(mult_results, key=lambda k: mult_results[k])
                predicted = {'class': max_class, 'percentage': mult_results[max_class]}
                return predicted
            else:
                print("No results to classify.")
                return None

        except Exception as e:
            print('An error occurred during classification: ', e)
            return None