# This file contains implementations and interfaces of data providers
import random
import numpy as np
from approach import util
from data import Datasetloader


# Class responsible of holding and delivering individual measurements upon request
class DataProvider:

    # Constructor takes the base folder (basefolder) to use to detect all available metrics, and the metric to use
    # (robust_metric) to aggregate the individual measurement intervals for each experiment run. Default: Mean
    def __init__(self, basefolder, robust_metric, export=False):
        self.ds = Datasetloader.load_data_set(basefolder).calculate_robust_metric(robust_metric)
        self.derive_vm_sizes()
        if export:
            with open("csvexport.csv", "w+") as file:
                self.ds["tp"] = self.ds["target/throughput"].apply(np.median)
                self.ds.to_csv(file)
        # shuffle data points
        for index, row in self.ds.iterrows():
            ltps = list(self.ds.loc[index, "target/throughput"])
            random.shuffle(ltps)
            self.ds.at[index, "target/throughput"] = ltps
            llat = list(self.ds.loc[index, "target/latency"])
            random.shuffle(llat)
            self.ds.at[index, "target/latency"] = llat


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

    # Adds vmsize enum column to the dataframe as defined in the global vmsize_map based on core and memory counts.
    def derive_vm_sizes(self):
        for index, row in self.ds.iterrows():
            self.ds.loc[index, 'feature/vmsize'] = util.get_vm_size(row["feature/cores"], row["feature/memory"])
