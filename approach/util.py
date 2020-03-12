# Set of util methods
import numpy as np
import itertools


# Calculated the Hodges-Lehmann metric for a given input vector.
def calculate_hodges_lehmann(l_input):
    l_avgs = []
    k = 0
    j = 0
    while k < len(l_input):
        while j < len(l_input):
            l_avgs.append(np.mean([l_input[k], l_input[j]]))
            j = j + 1
        k = k + 1
        j = k
    return np.median(l_avgs)


# Calculates and returns the cartesian product as a list of dicts of the feature ranges, given as a dict of sets.
def get_cartesian_feature_product(feature_values):
    dicts = []
    lists = []
    for key in feature_values:
        lists.append(list(feature_values[key]))
    combinations = itertools.product(*lists)
    for c in combinations:
        d = {}
        i = 0
        # ensure that we keep the order
        for key in feature_values:
            d[key] = c[i]
            i = i + 1
        dicts.append(d)
    return dicts
