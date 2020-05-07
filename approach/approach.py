"""
Main module of the implementation of our approach, as submitted in our scientific paper.
Our approach consists of three main steps:
0. Choose appropriate robust metric for analyzing the data (Done offline by metricAnalyzer.py)
1. Dynamically select the right amount of measurement repetitions of each data point until a certain threshold of measurement accuracy is met
2. Dynamically select the next point of the measurement space, required to be sampled in order to increase the accuracy of the model
3. Model the whole search space with the available points using different ML algorithms
"""
import math

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

import approach.dataprovider as dp
from approach.units.measurementunit import MeasurementSet
from approach.units.modelproviderunit import PerformanceModelProvider
from approach.units.samplingunit import ConfigurationPointProvider


class PerformancePredictior:
    """
    This is the main class of the performance prediction step. This class starts the workflow, stores all required
    constants and coordinates the individual units.
    """
    # Robust metric to be used in order to summarize one benchmark run
    ROBUST_METRIC = lambda x: np.percentile(x, 95)
    # Metric to summarize multiplie repetitions of a benchmark run
    MEASUREMENT_POINT_AGGREGATOR = np.median
    # Metric to quantify the variation of the aggregation function, i.e., the confidence in its stability
    CONFIDENCE_QUANTIFIER = stats.variation
    # Threshold quantifying which values of the CONFIDENCE_QUANTIFIER are deemed acceptable
    COV_THRESHOLD = 0.02
    # Target accuracy threshold for the performance models internal validation until it is deemed acceptable (negative values for error scores)
    ACC_THRESHOLD = -0.1
    # Target accuracy threshold for the performance models internal validation until it is deemed acceptable
    MAX_MEASUREMENTS = 10

    # For the model construction
    # Ratio of measurement points (in relation to the total number of points) that are taken
    INITIAL_MEASUREMENT_RATIO = 0.05
    # Ratio of measurement points (in relation to the total number of points) that are taken
    INCREMENT_MEASUREMENT_RATIO = 0.01
    # Maximum ratio of measurement points (in relation to the total number of points) that are allowed to be taken
    UPPER_BOUND_MEASUREMENT_RATIO = 0.9
    # ML model to use for learning and prediction
    MODEL_TYPE = RandomForestRegressor()

    def __init__(self, datafolder):
        """
        Initializing the approach objects, with all required sub-units.

        Keyword arguments:
        datafolder -- the datafolder is forwarded to the DataProvider in order to locate the measurement files.
        """

        # the provider of the measurement data
        self.dataprovider = dp.DataProvider(datafolder, robust_metric=PerformancePredictior.ROBUST_METRIC)
        # storing  measurement data
        self.measurements = MeasurementSet(dataprovider=self.dataprovider, target_metric="target/throughput",
                                           threshold=PerformancePredictior.COV_THRESHOLD,
                                           max_measurments=PerformancePredictior.MAX_MEASUREMENTS,
                                           aggregation_function=PerformancePredictior.MEASUREMENT_POINT_AGGREGATOR,
                                           confidence_function=PerformancePredictior.CONFIDENCE_QUANTIFIER)
        self.modelprovider = PerformanceModelProvider(model_type=PerformancePredictior.MODEL_TYPE)
        self.configuration_provider = ConfigurationPointProvider(self.dataprovider.get_all_possible_values())

    def start_training_workflow(self):
        """
        Main entry point of the performance prediction workflow. Executes the measurment-modelling loop until a
        sufficient accuracy is achieved. Then returns the final model.
        """
        # print("Started model workflow.")
        # print("Conducting initial set of measurements.")
        self.__get_initial_measurements()
        model, accuracy = self.modelprovider.create_model(self.measurements)
        # print("Initial internal model accuracy using " + (str(len(self.measurements.get_available_feature_set()))) + " measurements: " + str(
        #     accuracy))
        while accuracy < PerformancePredictior.ACC_THRESHOLD:
            curr_points = len(self.measurements.get_available_feature_set())
            if (curr_points >= len(
                    self.configuration_provider.get_feature_space()) * PerformancePredictior.UPPER_BOUND_MEASUREMENT_RATIO):
                # We already took too many points
                print(
                    "Breaking iterative model improvement as too many measurement points have been demanded (Measured: {0}, total points: {1}, applied max ratio: {2}.".format(
                        curr_points, len(self.configuration_provider.get_feature_space()),
                        PerformancePredictior.UPPER_BOUND_MEASUREMENT_RATIO))
                break
            self.__add_one_increment()
            model, accuracy = self.modelprovider.create_model(self.measurements)
            # print("Improved internal model accuracy using " + (str(len(self.measurements.get_available_feature_set()))) + " measurements: " + str(
            #     accuracy))
        print("Final internal model accuracy using " + (
            str(len(self.measurements.get_available_feature_set()))) + " measurements: " + str(
            accuracy) + ". Returning model.")
        self.model = model
        return self

    def get_prediction(self, features):
        """
        Returns a prediction for the given feature set. Raises a ValueError if the feature is unknown, and returns the
        original measurement, if the point is already part of the measurement set.

        Keyword arguments:
        features -- the feature for the prediction
        """

        # Check if feature is valid
        if features not in self.configuration_provider.get_feature_space():
            raise ValueError("The feature combination {0} is not known to this model.".format(features))
        # Check if feature is already present in measurement set
        if features in self.measurements.get_available_feature_set():
            return self.measurements.get_one_measurement_point(features)
        # Return model prediction of feature set
        return self.model.predict(self.modelprovider.get_feature_vector(features))

    def __get_initial_measurements(self):
        """Defines and collects the set of initial measurements to conduct"""
        # Determine number of points to be measured based on the size of the feature set
        feat_len = len(self.configuration_provider.feature_space)
        points = int(feat_len * PerformancePredictior.INITIAL_MEASUREMENT_RATIO)
        # print(
        #   "We have a total number of {0} configuration points in the space and apply a ratio of {1}, resulting in a total of {2} initial measurements.".format(
        #        feat_len, PerformancePredictior.INITIAL_MEASUREMENT_RATIO, points))
        for i in range(0, points):
            self.__add_one_measurement(i)

    def __add_one_increment(self):
        """Increases the current number of measurements by the defined incremental amount (rounded up)"""
        current = len(self.measurements.get_available_feature_set())
        points = math.ceil(
            len(self.configuration_provider.get_feature_space()) * PerformancePredictior.INCREMENT_MEASUREMENT_RATIO)
        for i in range(points):
            self.__add_one_measurement(current + i)

    def __add_one_measurement(self, index):
        """
        Selects a new configuration point and adds it to the measurement set.

        Keyword arguments:
        index -- the number of already measured points.
        """
        feats = self.configuration_provider.get_next_measurement_features(index)
        self.measurements.get_one_measurement_point(feats)
