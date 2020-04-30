# This unit contains all functionality to select the next sample, i.e., measurement point
import numpy as np
from approach import util

# This class determines the next feature combination to be sampled
class ConfigurationPointProvider:

    # Initializes an object, which takes as input a dict of the possible values
    def __init__(self, possible_values):
        self.values = possible_values
        # calculate possible feature space
        self.feature_space = util.get_cartesian_feature_product(self.values)
        # get random walk permutation (order in which to traverse the points)
        self.permutation = np.random.permutation(len(self.feature_space))

    # Decides on the next feature combination that it measured in order to improve the model.
    # The parameter "index" is a counter specifying how many measurements have been conducted already.
    def get_next_measurement_features(self, index):
        return self.feature_space[self.permutation[index]]