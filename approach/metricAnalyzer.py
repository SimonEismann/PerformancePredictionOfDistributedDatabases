# This script evaluates the available data and calculates the best metrics in terms of robustness.
import numpy as np
from scipy import stats

# Analyzes the performance of the given metrics and applies all of them to the given data set.
# Returns a dictionary, assigning every measurement metric (i.e., throughput or latency) a dictionary, assigning every
# robust metric the average as well as the variance of all COVs (coefficient of variation) over all measurement points
# of each experiment.
def analyze_metrics(dataset, metriclist):
    all_vars_tp = {}
    all_vars_lat = {}
    # iterate through all metrics to be analyzed
    for metric in metriclist:
        df = dataset.calculate_robust_metric(metric[1])
        tps = []
        lats = []
        # iterate through all experiment of that dataset
        for index, row in df.iterrows():
            # calculate coefficient of variance of the specific experiment
            tps.append(stats.variation(row['target/throughput']))
            lats.append(stats.variation(row['target/latency']))
        all_vars_tp[metric[0]] = (np.average(tps), np.var(tps))
        all_vars_lat[metric[0]] = (np.average(lats), np.var(lats))
    return {"throuhput": all_vars_tp, "latency": all_vars_lat}


def find_optimal_metric(dataset, metriclist):
    results = analyze_metrics(dataset, metriclist)
    for value in results:
        print(value)
        print(min(results[value]))