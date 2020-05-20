"""
This units contains implementations and interfaces to provide measurement data.
"""
import os
import random
import numpy as np
import pandas as pd
import re
from io import StringIO

from approach import util
from dataload import Datasetloader


class DataProvider:
    """
    Class responsible of holding and delivering individual measurements upon request.
    """

    def __init__(self, basefolder, robust_metric, export=False):
        """
        Constructor takes the base folder to use to detect all available metrics, and the metric to use to aggregate the
        individual measurement intervals for each experiment run.
        :param basefolder: The base folder to scan.
        :param robust_metric: The metric used to aggregate individual measurements.
        :param export: Boolean, indicating whether or not the data-frame should additionally be stored into a .csv file.
        """
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


    def get_measurement_point(self, index, metric, features):
        """
        Returns the i-th (index-th) measurement point of the given features.
        :param index: The i-th measurement to obtain (i-th index in the list).
        :param metric: The target metric to retrieve.
        :param features: The feature combination to measure.
        :return: The index-th measurement point of the given features.
        :raises ValueError, index is higher than the number of available measurements.
        """
        if index >= 10:
            raise ValueError("No more than ten measurements available for features "+str(features)+".")
        experiment = self.get_exp(metric, features)
        return experiment[index]

    def get_exp(self, metric, features):
        """
        Returns all available experiment measurements for the given features and the given metrics.
        :param metric: The target metric to retrieve.
        :param features: The feature combination to measure.
        :return: A list of measurements available for this feature combination and the corresponding target metric.
        """
        for index, row in self.ds.iterrows():
            match = True
            for key in features:
                if row["feature/"+key] != features[key]:
                    match = False
                    break
            if match:
                return row[metric]
        raise ValueError("No experiment with features "+str(features)+" was found.")

    def get_all_possible_values(self):
        """
        Returns a dict with all possible features and assigning all possible values of that feature to that feature.
        :return: A dictionary assigning each possible feature a list of possible values.
        """
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

    def derive_vm_sizes(self):
        """
        Adds a "feature/vmsize" column to the data frame as defined in the util.vmsize_map based on core and memory
        counts.
        :return: None.
        """
        for index, row in self.ds.iterrows():
            self.ds.loc[index, 'feature/vmsize'] = util.get_vm_size(row["feature/cores"], row["feature/memory"])

class Dataset:
    """
    This class represents the total data set, i.e., a collection of Experiment instances.
    """
    exps = False

    def __init__(self, exps):
        """
         Creates a new instance of this class.
        :param exps: The list of experiments to collect.
        """
        self.exps = exps

    def calculate_robust_metric(self, calculate_robust_metric_function):
        """
        Applies the given robust metric function to all contained experiments.
        :param calculate_robust_metric_function: The function to apply calculating the desired metric.
        :return: A dataframe with one row per experiment.
        """
        data_set = []
        for exp in self.exps:
            data_set.append(exp.calculate_robust_metric(calculate_robust_metric_function))
        df = pd.DataFrame(data_set)
        df = df.apply(pd.to_numeric, errors='ignore')
        return df


class Experiment:
    """
    This class contains one Experiment, i.e., a set of measurement values for a given configuration of parameters.
    """

    def __init__(self, configuration, basefolder):
        """
        Creates a new experiment instance for the given configuration by scanning all files given by basefolder.
        :param configuration: The parameter configuration to create.
        :param basefolder: The basefolder to use.
        """
        self.configuration = configuration
        folders = os.listdir(basefolder + "\\" + configuration)
        if "plots" in folders:
            folders.remove("plots")
        if len(folders) > 1:
            self.features = self.extract_features(configuration)
            path = basefolder + "\\" + configuration
            folders = os.listdir(path)
            folders = [path + "\\" + fol for fol in folders]
            self.throughput_values = []
            self.latency_values = []
            for repPath in folders:
                if not (repPath.__contains__("plots") | repPath.__contains__("archiv")):
                    if os.path.isfile(repPath + "\\data\\load.txt"):
                        with open(repPath + "\\data\\load.txt", encoding='utf8') as f:
                            text = f.read().strip()
                            text = re.sub(r'^\[OVERALL\].*\n?', '', text, flags=re.MULTILINE)
                            text = re.sub(r'^\[INSERT\].*\n?', '', text, flags=re.MULTILINE)
                            resps = pd.read_csv(StringIO(text), sep=";",
                                                names=['Time', 'Throughput', 'Latency', 'Garbage'])
                            metrics = {'throughputdata': np.asarray(resps['Throughput']),
                                       'latencydata': np.asarray(resps['Latency'])}
                            if len(metrics['throughputdata']) == 0:
                                print("[WARN] Empty load.txt file: ", repPath)
                            else:
                                self.throughput_values.append(metrics['throughputdata'])
                            if len(metrics['latencydata']) == 0:
                                print("[WARN] Empty latency data: ", repPath)
                            else:
                                self.latency_values.append(metrics['latencydata'])
                    else:
                        print("[WARN] No load.txt file:", repPath)
        else:
            raise Exception('Invalid Experiment: ' + configuration)

    def calculate_robust_metric(self, calculate_robust_metric_function):
        """
        Applies the robust metric function to all measurements of this experiment.
        :param calculate_robust_metric_function: The function to apply calculating the desired metric.
        :return: A dict containing the configuration, the features, the robust throughput, and the robust latency
        """
        robust_metrics_tp = []
        robust_metrics_lat = []
        for raw_measurements in self.throughput_values:
            robust_metrics_tp.append(calculate_robust_metric_function(raw_measurements))
        for raw_measurements in self.latency_values:
            robust_metrics_lat.append(calculate_robust_metric_function(raw_measurements))
        return {**{'configuration': self.configuration}, **self.features, **{'target/throughput': robust_metrics_tp},
                **{'target/latency': robust_metrics_lat}}

    def extract_features(self, folder_name):
        """
        Extract the feature representation as a dictionary, based on the folder name given as a string.
        :param folder_name: The string representation of the configuration
        :return: A dictionary containing all possible feature values and its value.
        """
        list_of_features = folder_name.split("_")
        vm_size = list_of_features[0][3:]
        res = util.get_core_and_memory(vm_size)
        cores, memory = res[0], res[1]
        cluster_size = list_of_features[1].split("-")[1]
        client_consistency = util.format_client_consistency(list_of_features[3].split("-")[1])
        replication_factor = list_of_features[2].split("-")[1]
        feats = {'feature/clustersize': cluster_size, 'feature/replicationfactor': replication_factor,
                 'feature/clientconsistency': client_consistency, 'feature/cores': cores, 'feature/memory': memory}
        return feats
