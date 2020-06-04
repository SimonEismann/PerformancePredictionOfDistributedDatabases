"""
Set of util methods.
"""
import numpy as np
import itertools

# Map assigning VM sizes with core and memory allocations
vmsize_map = {"tiny": [1, 2], "small": [2, 4], "medium": [4, 8], "large-memory": [4, 12], "large-cpu": [6, 8], "large": [6, 10]}


def calculate_hodges_lehmann(l_input):
    """
    Calculates the Hodges-Lehmann metric for a given input vector.
    :param l_input: The input vector to process.
    :return: The calculated Hodges-Lehmann metric.
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
    :param feature_values: The features values as a dict of sets.
    :return: A filtered list of all feature combinations.
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
    :param combinations: The combination list to filter.
    :return: The filtered list.
    """
    final_combinations = []
    for c in combinations:
        if is_valid_combination(c):
            final_combinations.append(c)
    return final_combinations


def is_valid_combination(combination):
    """
    Decides if a combination is valid based on domain knowledge from the area of databases.
    :param combination: A single combination to analyze.
    :return: True, if combination is valid, False otherwise.
    """
    if combination["clientconsistency"] > combination["replicationfactor"]:
        # this is an unvalid configuration
        return False
    # if you make it until here, you are probably valid
    return True


def get_vm_size(cores, memory):
    """
    Returns the vmsize based on the core and memory counts as defined in the vmsize_map.
    :param cores: The number of cores.
    :param memory: The amount of memory.
    :return: The string representation of the vm size.
    """
    sizes = [cores, memory]
    for size in vmsize_map:
        if vmsize_map[size] == sizes:
            vmsize = size
    if not vmsize:
        raise ValueError("The VM size associated with sizes of "+str(sizes)+" could not be mapped...")
    return vmsize


def get_core_and_memory(vmsize):
    """
    Returns the number of memory and cores based on the vmsize string as defined in the vmsize_map.
    :param vmsize: the vmsize description to convert
    :return: A list of cores and memory.
    """
    if vmsize not in vmsize_map:
        raise ValueError("The VM size associated with the name " + str(vmsize) + " could not be mapped...")
    return vmsize_map[vmsize]


def format_client_consistency(client_consistency):
    """
    Extracts the client consistency as an integer based on its string representation.
    :param client_consistency: A string representation of the client consistency.
    :return: The client consistency as an integer.
    """
    if client_consistency == "any" or client_consistency == "ANY":
        return 0
    if client_consistency == "one" or client_consistency == "ONE":
        return 1
    if client_consistency == "two" or client_consistency == "TWO":
        return 2
    if client_consistency == "three" or client_consistency == "THREE":
        return 3
    raise ValueError("Unknown client consistency: " + client_consistency)


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculates the mean absolute percentage error (MAPE).
    :param y_true: The real values to compare with.
    :param y_pred: The predicted values to compare.
    :return: The calculated error.
    """
    y_true, y_pred = np.array(y_true).reshape(1,-1), np.array(y_pred).reshape(1,-1)
    return np.mean(np.abs((y_true - y_pred)) / y_true) * 100


def negative_mape_scorer(estimator, X, y):
    """
    Calculates the negative mean absolute percentage error (NMAPE) of the given estimator, as a negative ratio.
    :param estimator: The trained estimator to use. Should implement the "predict(X)" function.
    :param X: Vector-matrix of the features to evaluate by predicting them via the estimator.
    :param y: Vector of the expected labels for the given features.
    :return: The NMAPE score of the given estimator.
    """
    y_pred = estimator.predict(X)
    return - mean_absolute_percentage_error(y, y_pred)/100