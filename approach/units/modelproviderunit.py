"""
This units contains all functionality related to the machine learning and performance modeling procedure.
"""
from approach import util
import numpy as np
from sklearn.model_selection import cross_val_score


class PerformanceModelProvider:
    """
    This class takes care of training and evaluating performance models of a specified type.
    """

    def __init__(self, model_type):
        """
        Initialize this object and specify the model to use for modeling.
        :param model_type: The modeling formalism to train.
        """
        self.model = model_type
        self.order = None

    # Create a performance model using all data available in the given measurements.
    # Returns the built model and an internal accuracy estimate.
    def create_model(self, measurements):
        """
        Create a performance model using all data available in the given measurements. Returns the built model and an
        internal scoring estimate.
        :param measurements: An instance of measurementunit.MeasurementSet containing the measurements for training.
        :return: The final model model and the internal error score.
        """
        labels = []
        measurement_set = measurements.get_available_feature_set()
        features = []
        for feat in measurement_set:
            labels.append(measurements.get_one_measurement_point(feat))
            features.append(self.get_feature_vector(feat))
            np_features = np.concatenate(features)
        instance = PerformanceModel(model_to_use=self.model, training_features=np_features, training_labels=labels)
        return instance.get_trained_model(), instance.get_internal_accuracy_score()

    def get_feature_vector(self, features):
        """
        Transforms the feature values given by a dict into an ordered feature vector. The order of the features is
        guaranteed to be the same.
        :param features: Feature representation to transform.
        :return: Numpy array containing a numerical representation of the features.
        """
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


class PerformanceModel:
    """
    Instances of this class represent one performance model.
    """

    def __init__(self, model_to_use, training_features, training_labels):
        """
        Creates and trains a performance model instance.
        :param model_to_use: The modeling formalism to use.
        :param training_features: The features of the training set.
        :param training_labels: The labels of the training set. Must be of same size and order as the features.
        """
        self.features = training_features
        self.labels = training_labels
        model, score = self.__train_model(model_to_use)
        self.score = score
        self.model = model

    def get_trained_model(self):
        """
        Returns the trained model.
        :return: The trained model of this instance.
        """
        return self.model

    def get_internal_accuracy_score(self):
        """
        The internal score that the respective model received.
        :return: The internal score.
        """
        return self.score

    def __train_model(self, model_to_use):
        """
        Trains the model with the available data.
        :param model_to_use: The model type to use as estimator. Should implement the fit(X,y) function.
        :return: Returns the trained model, and the internal score that this model achieved.
        """
        score = cross_val_score(estimator=model_to_use, X=self.features, y=self.labels, cv=3, n_jobs=-1, scoring=util.negative_mape_scorer).mean()
        full_model = model_to_use.fit(X=self.features, y=self.labels)
        return full_model, score
