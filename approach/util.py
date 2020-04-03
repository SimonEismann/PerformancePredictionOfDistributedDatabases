# Set of util methods
import numpy as np
import itertools

# Map assigning VM sizes with core and memory allocations
vmsize_map = {"tiny": [1, 2], "small": [2, 4], "medium": [4, 8], "large-memory": [4, 12], "large-cpu": [6, 8]}

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
    return filter_combinations(dicts)


# Filters impossible feature combinations based on DB domain knowledge.
def filter_combinations(combinations):
    final_combinations = []
    for c in combinations:
        if is_valid_combination(c):
            final_combinations.append(c)
    return final_combinations


# Decides if a combination is valid based on domain knowledge from the area of databases.
def is_valid_combination(combination):
    if combination["clientconsistency"] > combination["replicationfactor"]:
        # this is an unvalid configuration
        return False
    # if you make it until here, you are probably valid
    return True


# Returns the vmsize based on the core and memory counts as defined in the vmsize_map.
def get_vm_size(cores, memory):
    sizes = [cores, memory]
    for size in vmsize_map:
        if vmsize_map[size] == sizes:
            vmsize=size
    if not vmsize:
        raise ValueError("The VM size associated with sizes of "+str(sizes)+" could not be mapped...")
    return vmsize


# Returns the number of memory and cores based on the vmsize string as defined in the vmsize_map.
def get_core_and_memory(vmsize):
    if vmsize not in vmsize_map:
        raise ValueError("The VM size associated with the name " + str(vmsize) + " could not be mapped...")
    return vmsize_map[vmsize]