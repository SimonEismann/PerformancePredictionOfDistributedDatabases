import approach.dataprovider as dp
import numpy as np
from scipy import stats


# Our approach consists of three main steps.
# 0. Choose appropriate robust metric for analyzing the data (Done offline by metricAnalyzer.py)
# 1. Dynamically select the right amount of measurement repetitions of each data point until a certain threshold of measurement accuracy is met
# 2. Dynamically select the next point of the measurement space, required to be sampled in order to increase the accuracy of the model
# 3. Model the whole search space with the available points using different ML algorithms
class PerformancePredictior:

    applied_metric = lambda x: np.percentile(x, 0.95)

    def __init__(self, datafolder):
        self.dataprovider = dp.DataProvider(datafolder, robust_metric=PerformancePredictior.applied_metric)
        self.measurements = {}

    def get_entry(self, features):
        if hash(frozenset(features.items())) in self.measurements:
            return self.measurements[hash(frozenset(features.items()))]
        else:
            return None

    def add_entry(self, features, value):
        if hash(frozenset(features.items())) in self.measurements:
            raise ValueError("Can not add entry "+str(features)+" as it is already stored.")
        else:
            self.measurements[hash(frozenset(features.items()))] = value


    def get_one_measurement_point(self, features):
        if self.get_entry(features):
            raise ValueError("Measurement point with features "+str(features)+" is already stored.")
        values = [self.dataprovider.get_measurement_point(index=0, metric="target/throughput", features=features),
                  self.dataprovider.get_measurement_point(index=1, metric="target/throughput", features=features)]
        i = 2
        while not has_sufficient_accuracy(values):
            values.append(
                self.dataprovider.get_measurement_point(index=i, metric="target/throughput", features=features))
            i = i + 1
        self.add_entry(features, values)


def has_sufficient_accuracy(values):
    if stats.variation(values) > 0.1:
        return False
    return True
