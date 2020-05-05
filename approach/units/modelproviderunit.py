# This file contains all functionality related to the machine learning and modeling procedure.
from approach import util
import numpy as np
from sklearn.model_selection import cross_val_score


# This class takes care of training and evaluating performance models of a specified type.
class PerformanceModelProvider:

    # Initialize this object and specify the model to use for modeling.
    def __init__(self, model_type):
        self.model = model_type
        self.order = None

    # Create a performance model using all data available in the given measurements.
    # Returns the built model and an internal accuracy estimate.
    def create_model(self, measurements):
        labels = []
        measurement_set = measurements.get_available_feature_set()
        features = []
        for feat in measurement_set:
            labels.append(measurements.get_one_measurement_point(feat))
            features.append(self.get_feature_vector(feat))
            np_features = np.concatenate(features)
            #print(self.get_feature_vector(feat), measurements.get_one_measurement_point(feat))
        instance = PerformanceModel(model_to_use=self.model, training_features=np_features, training_labels=labels)
        return instance.get_trained_model(), instance.get_internal_accuracy_score()

    # Transforms the feature values given by a dict into an ordered feature vector.
    def get_feature_vector(self, features):
        if not self.order:
            self.order = []
            # This is the first call, therefore we initialize the order
            for key, value in features.items():
                self.order.append(key)
        vector = []
        # extract respective feature values in order order of self.order
        for feat in self.order:
            if feat not in features:
                raise ValueError("The feature {0} is not present in the features vector {1}".format(feat, features))
            val = features[feat]
            if feat == "vmsize":
                core, mem = util.get_core_and_memory(val)
                vector.append(core)
                vector.append(mem)
            else:
                vector.append(val)
        return np.asarray(vector).reshape(1, -1)




# Instances of this class represent our performance model.
class PerformanceModel:

    def __init__(self, model_to_use, training_features, training_labels):
        self.features = training_features
        self.labels = training_labels
        model, score = self.__train_model(model_to_use)
        self.score = score
        self.model = model

    def get_trained_model(self):
        return self.model

    def get_internal_accuracy_score(self):
        return self.score

    def __train_model(self, model_to_use):
        score = cross_val_score(estimator=model_to_use, X=self.features, y=self.labels, cv=3, n_jobs=-1, scoring="neg_mean_absolute_error").mean()
        full_model = model_to_use.fit(X=self.features, y=self.labels)
        return full_model, score
