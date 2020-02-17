import os
import numpy as np
import re
from io import StringIO
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)
pd.options.display.width = None
basefolder = "mowgli-ml-data\\results\\scalability-ycsb-write\\openstack\\cassandra"


class Dataset:
    exps = False

    def __init__(self, exps):
        self.exps = exps

    def calculate_robust_metric(self, calculate_robust_metric_function):
        data_set = []
        for exp in self.exps:
            data_set.append(exp.calculate_robust_metric(calculate_robust_metric_function))
        df = pd.DataFrame(data_set)
        df = df.apply(pd.to_numeric, errors='ignore')
        #df['feature/clustersize'] = df['feature/clustersize'].astype(int)
        #df['feature/replicationfactor'] = df['feature/replicationfactor'].astype(int)
        #df['target/throughput'] = df['target/throughput'].astype(int)
        return df


class Experiment:
    features = False
    measurement_values = False
    configuration = False

    def __init__(self, configuration):
        self.configuration = configuration
        folders = os.listdir(basefolder + "\\" + configuration)
        if "plots" in folders:
            folders.remove("plots")
        if len(folders) > 1:
            self.features = self.extract_features(configuration)
            path = basefolder + "\\" + configuration
            folders = os.listdir(path)
            folders = [path + "\\" + fol for fol in folders]
            self.measurement_values = []
            for repPath in folders:
                if not (repPath.__contains__("plots") | repPath.__contains__("archiv")):
                    if os.path.isfile(repPath + "\\data\\load.txt"):
                        with open(repPath + "\\data\\load.txt", encoding='utf8') as f:
                            text = f.read().strip()
                            text = re.sub(r'^\[OVERALL\].*\n?', '', text, flags=re.MULTILINE)
                            text = re.sub(r'^\[INSERT\].*\n?', '', text, flags=re.MULTILINE)
                            resps = pd.read_csv(StringIO(text), sep=";", names=['Time', 'Throughput', 'Latency', 'Garbage'])
                            metrics = {'throughputdata': np.asarray(resps['Throughput']), 'latencydata': np.asarray(resps['Latency'])}
                            if len(metrics['throughputdata']) == 0:
                                print("[WARN] Empty load.txt file: ", repPath)
                            else:
                                self.measurement_values.append(metrics['throughputdata'])
                    else:
                        print("[WARN] No load.txt file:", repPath)
        else:
            raise Exception('Invalid Experiment: ' + configuration)

    def calculate_robust_metric(self, calculate_robust_metric_function):
        robust_metrics = []
        for raw_measurements in self.measurement_values:
            robust_metrics.append(calculate_robust_metric_function(raw_measurements))
        return {**{'configuration': self.configuration}, **self.features, **{'target/throughput': robust_metrics}}

    def extract_features(self, folder_name):
        list_of_features = folder_name.split("_")
        vm_size = list_of_features[0][3:]
        cores, memory = self.extract_vm_specs(vm_size)
        cluster_size = list_of_features[1].split("-")[1]
        client_consistency = self.format_client_consistency(list_of_features[3].split("-")[1])
        replication_factor = list_of_features[2].split("-")[1]
        feats = {'feature/clustersize': cluster_size, 'feature/replicationfactor': replication_factor, 'feature/clientconsistency': client_consistency, 'feature/cores': cores, 'feature/memory': memory}
        return feats

    def extract_vm_specs(self, vm_size):
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
        if client_consistency == "one":
            return 1
        if client_consistency == "two":
            return 2
        if client_consistency == "three":
            return 3
        raise ValueError("Unknown client consistency: " + client_consistency)


def is_valid_exp(configuration):
    folders = os.listdir(basefolder + "\\" + configuration)
    if "plots" in folders:
        folders.remove("plots")
    if "archiv" in folders:
        folders.remove("archiv")
    if len(folders) > 1:
        if len(folders) != 10:
            print("[WARN] Unexpected number of experiment repetitions detected for " + configuration + ": ", len(folders))
    return len(folders) > 1


def load_data_set():
    exps = []
    for config in os.listdir(basefolder):
        if is_valid_exp(config):
            if config != "vm-large-memory_cs-7_rf-3_cc-two" and config != "vm-medium_cs-3_rf-3_cc-one" and config != "vm-tiny_cs-9_rf-2_cc-two":
                exp = Experiment(config)
                # Remove me later
                if len(exp.measurement_values) != 0:
                    exps.append(exp)
    return Dataset(exps)


def load_small_vm_data_set():
    exps = []
    for config in os.listdir(basefolder):
        if is_valid_exp(config):
            if config != "vm-large-memory_cs-7_rf-3_cc-two" and config != "vm-medium_cs-3_rf-3_cc-one" and config != "vm-tiny_cs-9_rf-2_cc-two":
                exp = Experiment(config)
                # Remove me later
                if exp.features['feature/cores'] == 2:
                    if len(exp.measurement_values) != 0:
                        exps.append(exp)
    return Dataset(exps)


def load_large_vm_data_set():
    exps = []
    for config in os.listdir(basefolder):
        if is_valid_exp(config):
            if config != "vm-large-memory_cs-7_rf-3_cc-two" and config != "vm-medium_cs-3_rf-3_cc-one" and config != "vm-tiny_cs-9_rf-2_cc-two":
                exp = Experiment(config)
                # Remove me later
                if exp.features['feature/cores'] == 6:
                    if len(exp.measurement_values) != 0:
                        exps.append(exp)
    return Dataset(exps)

if __name__ == "__main__":
    ds = load_data_set()
    print(ds.calculate_robust_metric(np.mean).dtypes)