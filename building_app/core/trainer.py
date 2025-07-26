import pandas as pd


class Trainer:

    def build_model(self, df):
        """
        Builds a Naive Bayes core from the input DataFrame or list of dicts.
        This function calculates prior probabilities and conditional probabilities
        for each class and feature..
        """
        # Accept both DataFrame and list of dicts
        df = self.ensure_dataframe(df)

        # Extract the features and the target classes from the df
        features = self.get_features(df)
        target_class = self.get_target_class(df)

        # Calculate prior probabilities P(C)
        class_priors = target_class.value_counts(normalize=True).to_dict()

        # Calculate conditional probabilities P(features|C)
        likelihoods = self.calculate_likelihoods(features, target_class)
        if likelihoods:
            return {"model": {'priors': class_priors, 'likelihoods': likelihoods}}
        else:
            return None


    @staticmethod
    def calculate_likelihoods(features, target_class):
        """Get the likelihoods relations"""
        try:
            likelihoods = {}
            for class_label in target_class.unique():
                features_mask = features[target_class == class_label]  # Subset where class = class_label
                class_label_str = str(class_label)
                likelihoods[class_label_str] = {}

                for col in features.columns:
                    feature_values = features[col].unique()  # All possible values of the feature
                    value_counts = features_mask[col].value_counts()  # Raw counts (not normalized)
                    total_count = value_counts.sum()
                    num_unique_values = len(feature_values)

                    # Initialize inner dict for this column
                    likelihoods[class_label_str][col] = {}

                    for val in feature_values:
                        str_val = str(val)
                        count = value_counts.get(val, 0)  # Get count or 0 if not found
                        # Laplace smoothing formula:
                        smoothed_prob = (count + 1) / (total_count + num_unique_values)
                        likelihoods[class_label_str][col][str_val] = smoothed_prob

            return likelihoods

        except Exception as e:
            print("A error accord: ", e)
            return None


    @staticmethod
    def ensure_dataframe(df):
        """Convert list of dicts to DataFrame if needed."""
        if isinstance(df, list):
            return pd.DataFrame(df)
        return df


    @staticmethod
    def get_target_class(df):
        """
        Extracts the target class column from the DataFrame.
        The target class is assumed to be the last column.
        """
        return df.iloc[:, -1]


    @staticmethod
    def get_features(df):
        """
        Extracts all feature columns from the DataFrame, excluding the target class.
        """
        return df.iloc[:, :-1]


