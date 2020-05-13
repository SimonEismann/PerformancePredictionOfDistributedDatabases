"""
This unit contains all functionality to select the next configuration point.
"""
import numpy as np
from approach import util

class ConfigurationPointProvider:
    """
    This class determines the next feature combination to be sampled.
    """

    def __init__(self, possible_values):
        """
        Initializes an object, which takes as input a dict of the possible values
        :param possible_values: List of all metrics with their possible values.
        """
        self.values = possible_values
        # calculate possible feature space
        self.feature_space = util.get_cartesian_feature_product(self.values)
        # get random walk permutation (order in which to traverse the points)
        self.permutation = np.random.permutation(len(self.feature_space))

    def get_feature_space(self):
        """
        Returns the cartesian products of all available features, as used by this configuration provider. The value is
        stored and just computed once.
        :return: A list of all possible feature combinations.
        """
        return self.feature_space

    def get_next_measurement_features(self, index):
        """
        Decides on the next feature combination that it measured in order to improve the model. The parameter "index"
        is a counter specifying how many measurements have been conducted already.
        :param index: How many measurements have been conducted already.
        :return: The feature combination that should be measured next.
        """
        return self.feature_space[self.permutation[index]]