
class Classifier:

    @staticmethod
    def classify_record(new_example, model_name, model):
        """Calculate the posterior probabilities for each class
        This involves multiplying the prior probability by the likelihood of the new example
        given each class."""
        try:
            # Extract the priors and likelihoods from the model
            priors = model[model_name]['priors']
            likelihoods = model[model_name]['likelihoods']
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
        except TypeError as e:
            print('Model or model name is None. ', e)
            return None

