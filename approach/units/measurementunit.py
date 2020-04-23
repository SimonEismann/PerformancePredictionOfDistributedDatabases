# This unit contains all functionality to repeat and quantify measured values with variability.
import numpy as np
from sklearn.ensemble import IsolationForest


# This class takes care of measuring requested feature combinations using a dataprovider.
# It determines the required amount of measurement repetitions and also stores prior measurements.
class MeasurementSet:

    # Initialize an instance. Parameters:
    # dataprovider: References to a DataProvider instance used for obtaining the raw measurements
    # target_metric: String of the target metric that is searched for. Forwarded directly to the dataprovider.
    # threshold: The threshold until a value of confidence_function is deemed to be sufficiently confident.
    # max_measurement: The maximum amount of measurements to be conducted.
    # aggregation_function: The function used for aggregating a set of measurement repetitions
    # confidence_function: The function used to evaluate the confidence into a given set of measurement repetitions.
    def __init__(self, dataprovider, target_metric, threshold, max_measurments, aggregation_function, confidence_function):
        # storing actual measurement data
        self.raw_measurements = {}
        # storing aggreagted measurement data. This storage is required as the aggregation step (incl. outlier filtering) might work non-determenistic.
        # By caching the output, consistent answers are guaranteed.
        self.permanent_value_store = {}
        # already measured features
        self.measured_features = []

        # used configuration
        self.measurement_point_aggregator = aggregation_function
        self.confidence_quantifier = confidence_function
        self.cov_threshold = threshold
        self.n_max = max_measurments
        self.dataprovider = dataprovider
        self.metric = target_metric

    # Returns the set of feature combinations that were added to the measurement set and are available up to this point.
    def get_available_feature_set(self):
        return self.measured_features

    # Returns the total number of raw measurement repetitions that were requested from the dataprovider for statistics.
    def get_total_number_of_measurements(self):
        sum = 0
        for key in self.raw_measurements:
            sum += len(self.raw_measurements[key])
        return sum

    # Returns the aggregated value (i.e., the target value) of one feature combination.
    def get_one_measurement_point(self, features):
        if features not in self.measured_features:
            # If not yet measured, obtain measurement
            self.__obtain_measurement(features)
        return self.permanent_value_store[hash(frozenset(features.items()))]

    # Obtains a requested feature combination, executes the required amount of repetitions and stores them in the data store.
    def __obtain_measurement(self, features):
        values = [self.dataprovider.get_measurement_point(index=0, metric=self.metric, features=features),
                  self.dataprovider.get_measurement_point(index=1, metric=self.metric, features=features)]
        i = 2
        while not (self.__accuracy_sufficient(values) or len(values) >= self.n_max):
            values.append(
                self.dataprovider.get_measurement_point(index=i, metric=self.metric, features=features))
            i = i + 1
        self.__add_entry(features, values)

    # Adds a set of measurement repetitions and its corresponding feature combintation to the raw_measurements store and the permanent_value_store. Raises an error, if the requested feature combintation is already stored.
    def __add_entry(self, features, value):
        if hash(frozenset(features.items())) in self.permanent_value_store:
            raise ValueError("Can not add entry " + str(features) + " as it is already stored.")
        else:
            self.raw_measurements[hash(frozenset(features.items()))] = value
            val, cov = self.__quantify_measurement_point(value)
            self.permanent_value_store[hash(frozenset(features.items()))] = val
            self.measured_features.append(features)

    # Returns True, if the values have a satisfactory confidence values, False if more repetitions are required.
    def __accuracy_sufficient(self, values):
        val, cov = self.__quantify_measurement_point(values)
        if cov > self.cov_threshold:
            return False
        return True

    # This function filters first all outliers, then calculates both the aggregation metric and the confidence metric of the remaining values and returns both.
    def __quantify_measurement_point(self, values):
        # 1. perform outlier detection
        core_values = self.__filter_outliers(values)
        if len(core_values) == 0:
            # if all get removed -> use original set
            core_values = values
        # 2. then report median
        val = self.measurement_point_aggregator(core_values)
        val2 = np.median(core_values)
        # 3. then report coefficient of variation
        cov = self.confidence_quantifier(core_values)
        return val, cov

    # This function takes a set of values as input, filters all outliers, and return the same set of values without the respective outliers.
    def __filter_outliers(self, values):
        vals = np.asarray(values)
        isolation_forest = IsolationForest(n_estimators=2)
        scores = isolation_forest.fit_predict(vals.reshape(-1, 1))
        mask = scores > 0
        # if not mask.all():
        #    print("Filtered "+str(len(mask) - np.sum(mask))+" anomalies for values "+str(values) + ".")
        return list(vals[mask])
