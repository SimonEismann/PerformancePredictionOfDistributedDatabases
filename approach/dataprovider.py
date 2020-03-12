import Data
import numpy as np


# Class responsible of holding and delivering individual measurements upon request
class DataProvider:

    vmsize_map = {"tiny": [1,2], "small": [2,4], "medium": [4,8], "large-memory": [4,12], "large-cpu": [6,8]}

    # Constructor takes the base folder (basefolder) to use to detect all available metrics, and the metric to use
    # (robust_metric) to aggregate the individual measurement intervals for each experiment run. Default: Mean
    def __init__(self, basefolder, robust_metric=np.mean):
        Data.basefolder = basefolder
        self.ds = Data.load_data_set().calculate_robust_metric(robust_metric)

    # Returns the i-th (index-th) measurement point of the given features.
    def get_measurement_point(self, index, metric, features):
        if index >= 10:
            raise ValueError("No more than ten measurements available.")
        experiment = self.get_exp(metric, features)
        return experiment[index]

    # Returns all available experiment measurements for the given features and the given metrics.
    def get_exp(self, metric, features):
        for index, row in self.ds.iterrows():
            match = True
            for key in features:
                if row["feature/"+key] != features[key]:
                    match = False
                    break
            if match:
                return row[metric]
        raise ValueError("No experiment with features "+str(features)+" was found.")

    # Returns a dict with all possible features and assigning all possible values of that feature to that feature.
    def get_all_possible_values(self):
        feature_values = {}
        for key in self.ds:
            if str(key).startswith("feature/"):
                feature_values[str(key).replace("feature/", "")] = set()
        # delete cores and memory and replace them by vm size
        del feature_values["cores"]
        del feature_values["memory"]
        feature_values["vmsize"] = set()
        for index, row in self.ds.iterrows():
            line = self.replace_vm_sizes(row)
            for key in feature_values:
                feature_values[key].add(row["feature/"+key])

        return feature_values


    def replace_vm_sizes(self, row):
        sizes = [row["feature/cores"], row["feature/memory"]]
        for size in DataProvider.vmsize_map:
            if DataProvider.vmsize_map[size] == sizes:
                vmsize=size
        if not vmsize:
            raise ValueError("The VM size associated with sizes of "+str(sizes)+" could not be mapped...")
        row["feature/vmsize"] = vmsize
        return row



