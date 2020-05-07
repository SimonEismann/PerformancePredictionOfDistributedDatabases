"""
Set of util methods.
"""
import numpy as np
import itertools

# Map assigning VM sizes with core and memory allocations
vmsize_map = {"tiny": [1, 2], "small": [2, 4], "medium": [4, 8], "large-memory": [4, 12], "large-cpu": [6, 8]}


def calculate_hodges_lehmann(l_input):
    """
   Calculates the Hodges-Lehmann metric for a given input vector.

    Keyword arguments:
    l_input -- the input vector to process
    """
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


def get_cartesian_feature_product(feature_values):
    """
   Calculates and returns the cartesian product as a list of dicts of the feature ranges, given as a dict of sets.

    Keyword arguments:
    feature_values -- the features values as a dict of sets
    """
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
    return filter_combinations(dicts)


def filter_combinations(combinations):
    """
   Filters impossible feature combinations based on DB domain knowledge.

    Keyword arguments:
    combinations -- the combination list to filter
    """
    final_combinations = []
    for c in combinations:
        if is_valid_combination(c):
            final_combinations.append(c)
    return final_combinations


def is_valid_combination(combination):
    """
    Decides if a combination is valid based on domain knowledge from the area of databases.

    Keyword arguments:
    combination -- a single combintation to analyze
    """
    if combination["clientconsistency"] > combination["replicationfactor"]:
        # this is an unvalid configuration
        return False
    # if you make it until here, you are probably valid
    return True


def get_vm_size(cores, memory):
    """
    Returns the vmsize based on the core and memory counts as defined in the vmsize_map.

    Keyword arguments:
    cores -- the number of cores
    memory -- the amount of memory
    """
    sizes = [cores, memory]
    for size in vmsize_map:
        if vmsize_map[size] == sizes:
            vmsize=size
    if not vmsize:
        raise ValueError("The VM size associated with sizes of "+str(sizes)+" could not be mapped...")
    return vmsize


def get_core_and_memory(vmsize):
    """
    Returns the number of memory and cores based on the vmsize string as defined in the vmsize_map.

    Keyword arguments:
    vmsize -- the vmsize description to convert
    """
    if vmsize not in vmsize_map:
        raise ValueError("The VM size associated with the name " + str(vmsize) + " could not be mapped...")
    return vmsize_map[vmsize]

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculates the mean absolute percentage error (MAPE), returns as percentage.

    Keyword arguments:
    y_true -- vector of expected values
    y_pred -- vector of actual predicted values
    """
    y_true, y_pred = np.array(y_true).reshape(1,-1), np.array(y_pred).reshape(1,-1)
    return np.mean(np.abs((y_true - y_pred)) / y_true) * 100


def negative_mape_scorer(estimator, X, y):
    """
    Calculates the negative mean absolute percentage error (NMAPE) of the given estimator, as a negative ratio

    Keyword arguments:
    estimator -- the trained estimator to use. Should implement the "predict(X)" function
    X -- vector-matrix of the features to evaluate by predicting them via the estimator
    y -- vector of the expected labels for the given features
    """
    y_pred = estimator.predict(X)
    return - mean_absolute_percentage_error(y, y_pred)/100