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
        all_vars_tp[metric[0]] = (np.average(tps), np.std(tps))
        all_vars_lat[metric[0]] = (np.average(lats), np.std(lats))
    return {"throughput": all_vars_tp, "latency": all_vars_lat}


# Returns a dictionary with for each given measurement metric, a metric with the lowest average coefficient of variation
# , together with the respective value is returned.
def find_optimal_metric(dataset, metriclist):
    results = analyze_metrics(dataset, metriclist)
    mins = {}
    for value in results:
        currmin = 2 # COV should be always smaller than 1
        currmin_metric = "NO_METRIC_FOUND"
        metrics = [i for i in results[value]]
        for m in metrics:
            if results[value][m][0] < currmin:
                currmin = results[value][m][0]
                currmin_metric = m
        mins[value] = (currmin_metric, currmin)
    return mins
