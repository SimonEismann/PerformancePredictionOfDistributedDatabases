import Data
import numpy as np


# Class responsible of holding and delivering individual measurements upon request
class DataProvider:

    vmsize_map = {"tiny": [1, 2], "small": [2, 4], "medium": [4, 8], "large-memory": [4, 12], "large-cpu": [6, 8]}

    # Constructor takes the base folder (basefolder) to use to detect all available metrics, and the metric to use
    # (robust_metric) to aggregate the individual measurement intervals for each experiment run. Default: Mean
    def __init__(self, basefolder, robust_metric):
        Data.basefolder = basefolder
        self.ds = Data.load_data_set().calculate_robust_metric(robust_metric)
        self.derive_vm_sizes()

    # Returns the i-th (index-th) measurement point of the given features.
    def get_measurement_point(self, index, metric, features):
        if index >= 10:
            raise ValueError("No more than ten measurements available for features "+str(features)+".")
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
        # delete cores and memory as we have already replaced them with vmsize
        del feature_values["cores"]
        del feature_values["memory"]
        for index, row in self.ds.iterrows():
            for key in feature_values:
                feature_values[key].add(row["feature/"+key])
        return feature_values

    # Returns the vmsize based on the core and memory counts as defined in the global vmsize_map.
    def get_vm_size(self, row):
        sizes = [row["feature/cores"], row["feature/memory"]]
        for size in self.vmsize_map:
            if self.vmsize_map[size] == sizes:
                vmsize=size
        if not vmsize:
            raise ValueError("The VM size associated with sizes of "+str(sizes)+" could not be mapped...")
        return vmsize

    # Adds vmsize enum column to the dataframe as defined in the global vmsize_map based on core and memory counts.
    def derive_vm_sizes(self):
        for index, row in self.ds.iterrows():
            self.ds.loc[index, 'feature/vmsize'] = self.get_vm_size(row)
