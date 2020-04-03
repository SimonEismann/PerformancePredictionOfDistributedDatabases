import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor

import approach.ModelProvider as mp
import approach.dataprovider as dp
import approach.util as util


# Our approach consists of three main steps.
# 0. Choose appropriate robust metric for analyzing the data (Done offline by metricAnalyzer.py)
# 1. Dynamically select the right amount of measurement repetitions of each data point until a certain threshold of measurement accuracy is met
# 2. Dynamically select the next point of the measurement space, required to be sampled in order to increase the accuracy of the model
# 3. Model the whole search space with the available points using different ML algorithms
class PerformancePredictior:
    applied_robust_metric = lambda x: np.percentile(x, 95)

    measurement_point_aggregator = np.median

    confidence_quantifier = stats.variation

    COV_THRESHOLD = 0.05

    ACC_THRESHOLD = 0.80

    INITIAL_MEASUREMENT_RATIO = 0.1

    def __init__(self, datafolder):
        # the provider of the measurement data
        self.dataprovider = dp.DataProvider(datafolder, robust_metric=PerformancePredictior.applied_robust_metric)
        # storing actual measurement data
        self.measurements = {}
        # already measured features
        self.measured_features = []
        # calculate possible feature space
        self.feature_space = util.get_cartesian_feature_product(self.dataprovider.get_all_possible_values())
        # get random walk permutation (order in which to traverse the points)
        self.permutation = np.random.permutation(len(self.feature_space))

    def get_entry(self, features):
        if hash(frozenset(features.items())) in self.measurements:
            return self.measurements[hash(frozenset(features.items()))]
        else:
            return None

    def add_entry(self, features, value):
        if hash(frozenset(features.items())) in self.measurements:
            raise ValueError("Can not add entry " + str(features) + " as it is already stored.")
        else:
            self.measurements[hash(frozenset(features.items()))] = value

    def get_total_number_of_measurements(self):
        sum = 0
        for key in self.measurements:
            sum += len(self.measurements[key])
        return sum

    def start_workflow(self):
        print("Started model workflow.")
        modelprovider = mp.PerformanceModelProvider(model_type=MLPRegressor(max_iter=1000000))
        print("Conducting initial set of measurements.")
        self.get_initial_measurements()
        model, accuracy = modelprovider.create_model(self.measurements)
        print("Initial internal model accuracy using " + (str(len(self.measurements))) + " measurements: " + str(
            accuracy))
        while accuracy < PerformancePredictior.ACC_THRESHOLD:
            self.add_one_measurement()
            model, accuracy = modelprovider.create_model(self.measurements)
            print("Improved internal model accuracy using " + (str(len(self.measurements))) + " measurements: " + str(
                accuracy))
        print("Final internal model accuracy using " + (str(len(self.measurements))) + " measurements: " + str(
            accuracy) + ". Returning model.")
        return model, accuracy

    def get_initial_measurements(self):
        # Determine number of points to be measured based on the size of the feature set
        points = int(len(self.feature_space) * PerformancePredictior.INITIAL_MEASUREMENT_RATIO)
        print(
            "We have a total number of {0} features in the space and apply a ratio of {1}, resulting in a total of {2} initial measurements.".format(
                len(self.feature_space), PerformancePredictior.INITIAL_MEASUREMENT_RATIO, points))
        for i in range(0, points):
            features = self.get_next_measurement_features(i)
            self.get_one_measurement_point(features)

    def get_next_measurement_features(self, index):
        return self.feature_space[self.permutation[index]]

    def filter_outliers(self, values):
        vals = np.asarray(values)
        isolation_forest = IsolationForest(n_estimators=1)
        scores = isolation_forest.fit_predict(vals.reshape(-1, 1))
        mask = scores > 0
        # if not mask.all():
        #    print("Filtered "+str(len(mask) - np.sum(mask))+" anomalies for values "+str(values) + ".")
        return list(vals[mask])

    def quantify_measurement_point(self, values):
        # 1. perform outlier detection
        core_values = self.filter_outliers(values)
        # 2. then report median
        val = PerformancePredictior.measurement_point_aggregator(core_values)
        # 3. then report coefficient of variation
        cov = PerformancePredictior.confidence_quantifier(core_values)
        return val, cov

    def get_one_measurement_point(self, features):
        if not self.get_entry(features):
            # If not yet measured, obtain measurement
            self.obtain_measurement(features)
        val, cov = self.quantify_measurement_point(self.get_entry(features))
        return val

    def obtain_measurement(self, features):
        if self.get_entry(features):
            raise ValueError("Measurement point with features " + str(features) + " is already stored.")
        values = [self.dataprovider.get_measurement_point(index=0, metric="target/throughput", features=features),
                  self.dataprovider.get_measurement_point(index=1, metric="target/throughput", features=features)]
        i = 2
        while not self.accuracy_sufficient(values):
            values.append(
                self.dataprovider.get_measurement_point(index=i, metric="target/throughput", features=features))
            i = i + 1
        self.add_entry(features, values)

    def accuracy_sufficient(self, values):
        val, cov = self.quantify_measurement_point(values)
        if cov > self.COV_THRESHOLD:
            return False
        return True
