# Main script for evaluation purposes
from approach import metricanalyzer
import Data
from approach import util
import boltons.statsutils as su

import numpy as np
import matplotlib
from scipy import stats


# Configurable base folder for the experiments
my_basefolder = "..\\mowgli-ml-data\\results\\scalability-ycsb-write\\openstack\\cassandra"

def calculate_and_plot_robustness_metrics():
    # List of metrics to be analyzed
    metrics = [('Mean', np.mean),
               ('Median', np.median),
               ('Max', np.max),
               ('Min', np.min),
               ('95th percentile', lambda x: np.percentile(x, 95)),
               ('90th percentile', lambda x: np.percentile(x, 90)),
               ('80th percentile', lambda x: np.percentile(x, 80)),
               ('70th percentile', lambda x: np.percentile(x, 70)),
               ('20th percentile', lambda x: np.percentile(x, 20)),
               ('10th percentile', lambda x: np.percentile(x, 10)),
               ('5th percentile', lambda x: np.percentile(x, 5)),
               ('Trimmed(5%) mean', lambda x: stats.trim_mean(x, 0.05)),
               ('Trimmed(10%) mean', lambda x: stats.trim_mean(x, 0.1)),
               ('Trimmed(20%) mean', lambda x: stats.trim_mean(x, 0.2)),
               ('Trimmed(30%) mean', lambda x: stats.trim_mean(x, 0.3)),
               ('Winzorized(5%) mean', lambda x: np.mean(stats.mstats.winsorize(x, [0.05, 0.05]))),
               ('Winzorized(10%) mean', lambda x: np.mean(stats.mstats.winsorize(x, [0.1, 0.1]))),
               ('Winzorized(20%) mean', lambda x: np.mean(stats.mstats.winsorize(x, [0.2, 0.2]))),
               ('Winzorized(30%) mean', lambda x: np.mean(stats.mstats.winsorize(x, [0.3, 0.3]))),
               ('Trimean', lambda x: su.Stats(x).trimean),
               ('Hodges-Lehmann', util.calculate_hodges_lehmann)
               ]
    Data.basefolder = my_basefolder
    ds = Data.load_data_set()
    performance = metricanalyzer.analyze_metrics(ds, metrics)
    for performance:
    print(performance)
    min = np.argmin(performance)
    print(min)

def plot_robustness_barchart(name, metrics):


if __name__ == "__main__":
    calculate_and_plot_robustness_metrics()