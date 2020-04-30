# Our approach consists of three main steps.
# 0. Choose appropriate robust metric for analyzing the data (Done offline by metricAnalyzer.py)
# 1. Dynamically select the right amount of measurement repetitions of each data point until a certain threshold of measurement accuracy is met
# 2. Dynamically select the next point of the measurement space, required to be sampled in order to increase the accuracy of the model
# 3. Model the whole search space with the available points using different ML algorithms
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

import approach.dataprovider as dp
from approach.units.measurementunit import MeasurementSet
from approach.units.modelproviderunit import PerformanceModelProvider
from approach.units.samplingunit import ConfigurationPointProvider


# This is the main class of the performance prediction step. This class starts the workflow, stores all required constants and coordinates the individual units.
class PerformancePredictior:
    # Robust metric to be used in order to summarize one benchmark run
    ROBUST_METRIC = lambda x: np.percentile(x, 95)
    # Metric to summarize multiplie repetitions of a benchmark run
    MEASUREMENT_POINT_AGGREGATOR = np.median
    # Metric to quantify the variation of the aggregation function, i.e., the confidence in its stability
    CONFIDENCE_QUANTIFIER = stats.variation
    # Threshold quantifying which values of the CONFIDENCE_QUANTIFIER are deemed acceptable
    COV_THRESHOLD = 0.02
    # Target accuracy threshold for the performance models internal validation until it is deemed acceptable
    ACC_THRESHOLD = 0.6
    # Target accuracy threshold for the performance models internal validation until it is deemed acceptable
    MAX_MEASUREMENTS = 10
    # Ratio of measurement points (in relation to the total number of points) that are taken
    INITIAL_MEASUREMENT_RATIO = 0.1
    # Maximum ratio of measurement points (in relation to the total number of points) that are allowed to be taken
    UPPER_BOUND_MEASUREMENT_RATIO = 0.8
    # ML model to use for learning and prediction
    MODEL_TYPE = RandomForestRegressor()

    # Initializing the approach objects, with all required sub-units. The datafolder is forwarded to the DataProvider in order to locate the measurement files.
    def __init__(self, datafolder):
        # the provider of the measurement data
        self.dataprovider = dp.DataProvider(datafolder, robust_metric=PerformancePredictior.ROBUST_METRIC)
        # storing  measurement data
        self.measurements = MeasurementSet(dataprovider=self.dataprovider, target_metric="target/throughput",
                                           threshold=PerformancePredictior.COV_THRESHOLD, max_measurments=PerformancePredictior.MAX_MEASUREMENTS,
                                           aggregation_function=PerformancePredictior.MEASUREMENT_POINT_AGGREGATOR,
                                           confidence_function=PerformancePredictior.CONFIDENCE_QUANTIFIER)
        self.modelprovider = PerformanceModelProvider(model_type=PerformancePredictior.MODEL_TYPE)
        self.configuration_provider = ConfigurationPointProvider(self.dataprovider.get_all_possible_values())

    # Main entry point of the performance prediction workflow. Executes the measurment-modelling loop until a sufficient accuracy is achieved. Then returns the final model.
    def start_training_workflow(self):
        #print("Started model workflow.")
        #print("Conducting initial set of measurements.")
        self.__get_initial_measurements()
        model, accuracy = self.modelprovider.create_model(self.measurements)
        # print("Initial internal model accuracy using " + (str(len(self.measurements.get_available_feature_set()))) + " measurements: " + str(
        #     accuracy))
        while accuracy < PerformancePredictior.ACC_THRESHOLD:
            curr_points = len(self.measurements.get_available_feature_set())
            if (curr_points >= len(self.measurements.get_available_feature_set()) * PerformancePredictior.UPPER_BOUND_MEASUREMENT_RATIO):
                # We already took too many points
                print("Breaking iterative model improvement as too many measurement points have been demanded.")
                break
            self.__add_one_measurement(curr_points)
            model, accuracy = self.modelprovider.create_model(self.measurements)
            # print("Improved internal model accuracy using " + (str(len(self.measurements.get_available_feature_set()))) + " measurements: " + str(
            #     accuracy))
        print("Final internal model accuracy using " + (str(len(self.measurements.get_available_feature_set()))) + " measurements: " + str(
            accuracy) + ". Returning model.")
        self.model = model
        return self

    # Returns a prediction for the given feature set. Raises a ValueError if the feature is unknown, and returns the
    # original measurement, if the point is already part of the measurement set.
    def get_prediction(self, features):
        # Check if feature is valid
        if features not in self.configuration_provider.get_feature_space():
            raise ValueError("The feature combination {0} is not known to this model.".format(features))
        # Check if feature is already present in measurement set
        if features in self.measurements.get_available_feature_set():
            return self.measurements.get_one_measurement_point(features)
        # Return model prediction of feature set
        return self.model.predict(self.modelprovider.get_feature_vector(features))

    # Defines and collects the set of initial measurements to conduct
    def __get_initial_measurements(self):
        # Determine number of points to be measured based on the size of the feature set
        feat_len = len(self.configuration_provider.feature_space)
        points = int(feat_len * PerformancePredictior.INITIAL_MEASUREMENT_RATIO)
        print(
            "We have a total number of {0} configuration points in the space and apply a ratio of {1}, resulting in a total of {2} initial measurements.".format(
                feat_len, PerformancePredictior.INITIAL_MEASUREMENT_RATIO, points))
        for i in range(0, points):
            self.__add_one_measurement(i)

    # Selects a new configuration point and adds it to the measurement set.
    # Index is the number of already measured points.
    def __add_one_measurement(self, index):
        feats = self.configuration_provider.get_next_measurement_features(index)
        self.measurements.get_one_measurement_point(feats)
