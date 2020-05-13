"""
This module contains the functionality to load a pre-recorded data set by Mowgli.
"""
import os
import re
from io import StringIO

import numpy as np
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.options.display.width = None


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
        cores, memory = self.extract_vm_specs(vm_size)
        cluster_size = list_of_features[1].split("-")[1]
        client_consistency = self.format_client_consistency(list_of_features[3].split("-")[1])
        replication_factor = list_of_features[2].split("-")[1]
        feats = {'feature/clustersize': cluster_size, 'feature/replicationfactor': replication_factor,
                 'feature/clientconsistency': client_consistency, 'feature/cores': cores, 'feature/memory': memory}
        return feats

    def extract_vm_specs(self, vm_size):
        """
        Extracts a tuple of core and memory configuration based on the VM size as string representation.
        :param vm_size: A string representation of the VM size.
        :return: The number of cores, and the amount of memory, separated by a comma.
        """
        if vm_size == "tiny":
            return 1, 2
        if vm_size == "small":
            return 2, 4
        if vm_size == "medium":
            return 4, 8
        if vm_size == "large-memory":
            return 4, 12
        if vm_size == "large-cpu":
            return 6, 8
        raise ValueError("Unknown VM size: " + vm_size)

    def format_client_consistency(self, client_consistency):
        """
        Extracts the client consistency as an integer based on its string representation.
        :param client_consistency: A string representation of the client consistency.
        :return: The client consistency as an integer.
        """
        if client_consistency == "one":
            return 1
        if client_consistency == "two":
            return 2
        if client_consistency == "three":
            return 3
        raise ValueError("Unknown client consistency: " + client_consistency)


def is_valid_exp(configuration, basefolder):
    """
    Checks, if there is at least 1 configuration for the given experiment folder.
    Also, deleted the "plots" and "archiv" folders.
    Prints a warning, if there are not 10 measurements in this folder.

    :param configuration: The specific configuration folder.
    :param basefolder: The corresponding base folder, in which the configuration folder is located.
    :return: True, if at least 1 measurement is contained in the given folder.
    """
    folders = os.listdir(basefolder + "\\" + configuration)
    if "plots" in folders:
        folders.remove("plots")
    if "archiv" in folders:
        folders.remove("archiv")
    if len(folders) > 1:
        if len(folders) != 10:
            print("[WARN] Unexpected number of experiment repetitions detected for " + configuration + ": ",
                  len(folders))
    return len(folders) > 1


def load_data_set(basefolder):
    """
    Loads all experiments found in the given basefolder, and returns an instance of Dataset containing all experiments.
    :param basefolder: The basefolder to search through.
    :return: An instance of dataset containing all experiments that were found.
    """
    exps = []
    for config in os.listdir(basefolder):
        if is_valid_exp(config):
            if config != "vm-large-memory_cs-7_rf-3_cc-two" and config != "vm-medium_cs-3_rf-3_cc-one":
                exp = Experiment(config, basefolder)
                if len(exp.throughput_values) != 0:
                    exps.append(exp)
    return Dataset(exps)


def load_tiny_vm_data_set(basefolder):
    """
    Loads all experiments of the "tiny" class found in the given basefolder, and returns an instance of Dataset
    containing all those Experiment instances.
    :param basefolder: The basefolder to search through.
    :return: An instance of dataset containing a filtered list of all experiments that were found.
    """
    exps = []
    for config in os.listdir(basefolder):
        if is_valid_exp(config):
            if config != "vm-large-memory_cs-7_rf-3_cc-two" and config != "vm-medium_cs-3_rf-3_cc-one":
                exp = Experiment(config, basefolder)
                if exp.features['feature/cores'] == 1:
                    if len(exp.throughput_values) != 0:
                        exps.append(exp)
    return Dataset(exps)


def load_small_vm_data_set(basefolder):
    """
    Loads all experiments of the "small" class found in the given basefolder, and returns an instance of Dataset
    containing all those Experiment instances.
    :param basefolder: The basefolder to search through.
    :return: An instance of dataset containing a filtered list of all experiments that were found.
    """
    exps = []
    for config in os.listdir(basefolder):
        if is_valid_exp(config):
            if config != "vm-large-memory_cs-7_rf-3_cc-two" and config != "vm-medium_cs-3_rf-3_cc-one":
                exp = Experiment(config, basefolder)
                if exp.features['feature/cores'] == 2:
                    if len(exp.throughput_values) != 0:
                        exps.append(exp)
    return Dataset(exps)


def load_large_vm_data_set(basefolder):
    """
    Loads all experiments of the "large" class found in the given basefolder, and returns an instance of Dataset
    containing all those Experiment instances.
    :param basefolder: The basefolder to search through.
    :return: An instance of dataset containing a filtered list of all experiments that were found.
    """
    exps = []
    for config in os.listdir(basefolder):
        if is_valid_exp(config):
            if config != "vm-large-memory_cs-7_rf-3_cc-two" and config != "vm-medium_cs-3_rf-3_cc-one":
                exp = Experiment(config, basefolder)
                if exp.features['feature/cores'] == 6:
                    if len(exp.throughput_values) != 0:
                        exps.append(exp)
    return Dataset(exps)


def load_tiny_small_vm_data_set(basefolder):
    """
    Loads all experiments of the "tiny" and the "small" class found in the given basefolder, and returns an instance of
    Dataset containing all those Experiment instances.
    :param basefolder: The basefolder to search through.
    :return: An instance of dataset containing a filtered list of all experiments that were found.
    """
    exps = []
    for config in os.listdir(basefolder):
        if is_valid_exp(config):
            if config != "vm-large-memory_cs-7_rf-3_cc-two" and config != "vm-medium_cs-3_rf-3_cc-one":
                exp = Experiment(config, basefolder)
                if exp.features['feature/cores'] == 1 or exp.features['feature/cores'] == 2:
                    if len(exp.throughput_values) != 0:
                        exps.append(exp)
    return Dataset(exps)
