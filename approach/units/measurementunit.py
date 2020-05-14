"""
This unit contains all functionality to repeat and quantify measured values with variability.
"""
import numpy as np
from sklearn.ensemble import IsolationForest


class MeasurementSet:
    """
    This class takes care of measuring requested feature combinations using a dataproviderunit.DataProvider. It
    determines the required amount of measurement repetitions and also stores prior measurements.
    """

    def __init__(self, dataprovider, target_metric, threshold, max_measurments, aggregation_function, confidence_function):
        """
        Initialize an instance.
        :param dataprovider: References to a DataProvider instance used for obtaining the raw measurements.
        :param target_metric: String of the target metric that is searched for. Forwarded directly to the dataprovider.
        :param threshold: The threshold until a value of confidence_function is deemed to be sufficiently confident.
        :param max_measurments: The maximum amount of measurements to be conducted.
        :param aggregation_function: The function used for aggregating a set of measurement repetitions.
        :param confidence_function: The function used to evaluate the confidence into a given set of measurement
        repetitions.
        """
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

    def get_available_feature_set(self):
        """
        Returns the set of feature combinations that were added to the measurement set and are available up to this
        point.
        :return: List of feature combinations already measured.
        """
        return self.measured_features

    def get_total_number_of_measurements(self):
        """
        Returns the total number of raw measurement repetitions that were requested from the dataprovider for statistics.
        :return: Total number of measurements stored.
        """
        sum = 0
        for key in self.raw_measurements:
            sum += len(self.raw_measurements[key])
        return sum

    def get_one_measurement_point(self, features):
        """
        Returns the aggregated value (i.e., the target value) of one feature combination.
        :param features: The desired feature value.
        :return: The stored value for that feature.
        """
        if features not in self.measured_features:
            # If not yet measured, obtain measurement
            self.__obtain_measurement(features)
        return self.permanent_value_store[hash(frozenset(features.items()))]

    def __obtain_measurement(self, features):
        """
        Obtains a requested feature combination, executes the required amount of repetitions and stores them in the
        data store.
        :param features: The feature combination to measure.
        :return: None.
        """
        values = [self.dataprovider.get_measurement_point(index=0, metric=self.metric, features=features),
                  self.dataprovider.get_measurement_point(index=1, metric=self.metric, features=features)]
        i = 2
        while not (self.__accuracy_sufficient(values) or len(values) >= self.n_max):
            values.append(
                self.dataprovider.get_measurement_point(index=i, metric=self.metric, features=features))
            i = i + 1
        self.__add_entry(features, values)

    def __add_entry(self, features, value):
        """
        Adds a set of measurement repetitions and its corresponding feature combination to the raw_measurements store
        and the permanent_value_store. Raises an error, if the requested feature combination is already stored.
        :param features: The feature combination to add.
        :param value: The value to add.
        :return: None.
        :raises: ValueError if the requested feature combination was added before.
        """
        if hash(frozenset(features.items())) in self.permanent_value_store:
            raise ValueError("Can not add entry " + str(features) + " as it is already stored.")
        else:
            self.raw_measurements[hash(frozenset(features.items()))] = value
            val, cov = self.__quantify_measurement_point(value)
            self.permanent_value_store[hash(frozenset(features.items()))] = val
            self.measured_features.append(features)

    def __accuracy_sufficient(self, values):
        """
        Returns True, if the values have a satisfactory confidence values, False if more repetitions are required.
        :param values: A list of values to analyze.
        :return: True, if the values have a satisfactory confidence values, False otherwise.
        """
        val, cov = self.__quantify_measurement_point(values)
        if cov > self.cov_threshold:
            return False
        return True

    def __quantify_measurement_point(self, values):
        """
        This function filters first all outliers, then calculates both the aggregation metric and the confidence metric
        of the remaining values and returns both.
        :param values: A list of values to analyze.
        :return: The aggregated value, together with an internal confidence.
        """
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

    def __filter_outliers(self, values):
        """
        This function takes a set of values as input, filters all outliers, and return the same set of values without
        the respective outliers.
        :param values: A list of values to filter.
        :return: A filtered list of values.
        """
        vals = np.asarray(values)
        isolation_forest = IsolationForest(n_estimators=2)
        scores = isolation_forest.fit_predict(vals.reshape(-1, 1))
        mask = scores > 0
        # if not mask.all():
        #    print("Filtered "+str(len(mask) - np.sum(mask))+" anomalies for values "+str(values) + ".")
        return list(vals[mask])
