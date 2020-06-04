"""
This module contains the functionality to load a pre-recorded data set by Mowgli.
"""
import os

import pandas as pd

import approach.units.dataproviderunit as Provider

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.options.display.width = None


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
        if is_valid_exp(config, basefolder):
            # these need to be excluded as they are the only ones in their size that have more than 1 config (and therefore are not filtered by "is_valid_exp"
            if config != "vm-large-memory_cs-7_rf-3_cc-two" and config != "vm-medium_cs-3_rf-3_cc-one":
                exp = Provider.Experiment(config, basefolder)
                if len(exp.throughput_values) != 0:
                    exps.append(exp)
        else:
            #print("Ignoring this folder: ", config)
            pass
    return Provider.Dataset(exps)


def load_tiny_vm_data_set(basefolder):
    """
    Loads all experiments of the "tiny" class found in the given basefolder, and returns an instance of Dataset
    containing all those Experiment instances.
    :param basefolder: The basefolder to search through.
    :return: An instance of dataset containing a filtered list of all experiments that were found.
    """
    exps = []
    for config in os.listdir(basefolder):
        if is_valid_exp(config, basefolder):
            if config != "vm-large-memory_cs-7_rf-3_cc-two" and config != "vm-medium_cs-3_rf-3_cc-one":
                exp = Provider.Experiment(config, basefolder)
                if exp.features['feature/cores'] == 1:
                    if len(exp.throughput_values) != 0:
                        exps.append(exp)
    return Provider.Dataset(exps)


def load_small_vm_data_set(basefolder):
    """
    Loads all experiments of the "small" class found in the given basefolder, and returns an instance of Dataset
    containing all those Experiment instances.
    :param basefolder: The basefolder to search through.
    :return: An instance of dataset containing a filtered list of all experiments that were found.
    """
    exps = []
    for config in os.listdir(basefolder):
        if is_valid_exp(config, basefolder):
            if config != "vm-large-memory_cs-7_rf-3_cc-two" and config != "vm-medium_cs-3_rf-3_cc-one":
                exp = Provider.Experiment(config, basefolder)
                if exp.features['feature/cores'] == 2:
                    if len(exp.throughput_values) != 0:
                        exps.append(exp)
    return Provider.Dataset(exps)


def load_large_vm_data_set(basefolder):
    """
    Loads all experiments of the "large" class found in the given basefolder, and returns an instance of Dataset
    containing all those Experiment instances.
    :param basefolder: The basefolder to search through.
    :return: An instance of dataset containing a filtered list of all experiments that were found.
    """
    exps = []
    for config in os.listdir(basefolder):
        if is_valid_exp(config, basefolder):
            if config != "vm-large-memory_cs-7_rf-3_cc-two" and config != "vm-medium_cs-3_rf-3_cc-one":
                exp = Provider.Experiment(config, basefolder)
                if exp.features['feature/cores'] == 6:
                    if len(exp.throughput_values) != 0:
                        exps.append(exp)
    return Provider.Dataset(exps)


def load_tiny_small_vm_data_set(basefolder):
    """
    Loads all experiments of the "tiny" and the "small" class found in the given basefolder, and returns an instance of
    Dataset containing all those Experiment instances.
    :param basefolder: The basefolder to search through.
    :return: An instance of dataset containing a filtered list of all experiments that were found.
    """
    exps = []
    for config in os.listdir(basefolder):
        if is_valid_exp(config, basefolder):
            if config != "vm-large-memory_cs-7_rf-3_cc-two" and config != "vm-medium_cs-3_rf-3_cc-one":
                exp = Provider.Experiment(config, basefolder)
                if exp.features['feature/cores'] == 1 or exp.features['feature/cores'] == 2:
                    if len(exp.throughput_values) != 0:
                        exps.append(exp)
    return Provider.Dataset(exps)
