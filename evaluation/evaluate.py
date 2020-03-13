# Main script for evaluation purposes
import boltons.statsutils as su
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import Data
from approach import metricanalyzer
from approach import util
import approach.approach
import approach.dataprovider as dp
import sys
from sklearn.metrics import mean_squared_error

# Configurable base folder for the experiments
my_basefolder = "..\\mowgli-ml-data\\results\\scalability-ycsb-write\\openstack\\cassandra"

res_folder = "results\\robust-metrics"


def calculate_and_plot_robustness_metrics():
    # List of metrics to be analyzed
    metrics = [('Mean', np.mean),
               ('Median', np.median),
               #('Max', np.max),
               #('Min', np.min),
               ('95th percentile', lambda x: np.percentile(x, 95)),
               ('90th percentile', lambda x: np.percentile(x, 90)),
               ('80th percentile', lambda x: np.percentile(x, 80)),
               ('70th percentile', lambda x: np.percentile(x, 70)),
               #('20th percentile', lambda x: np.percentile(x, 20)),
               #('10th percentile', lambda x: np.percentile(x, 10)),
               #('5th percentile', lambda x: np.percentile(x, 5)),
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
    for measurement in performance:
        plot_robustness_barchart(measurement, res_folder, performance[measurement])
    optimal_metrics = metricanalyzer.find_optimal_metric(ds, metrics)
    for metric in optimal_metrics:
        print("For metric " + metric + ", the best metric was " + str(
            optimal_metrics[metric][0]) + " with an average COV of " + str(optimal_metrics[metric][1]) + ".")


def plot_robustness_barchart(name, folder, metrics):
    avgs = [i[0] for i in list(metrics.values())]
    stds = [i[1] for i in list(metrics.values())]
    fig = plt.figure(figsize=(8, 6))
    rects = plt.bar(metrics.keys(), height=avgs, yerr=stds)
    plt.xticks(rotation=90)
    plt.tight_layout()
    #rects = ax.patches
    # Make labels.
    for rect in rects:
        height = rect.get_height()
        fig.axes[0].annotate('{:.3}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')

    plt.show()
    plt.savefig(folder + "\\" + name + ".png")
    plt.close()


def evaluate_measurement_point_selection():
    # create all feature instances
    data = dp.DataProvider(my_basefolder, approach.approach.PerformancePredictior.applied_robust_metric)
    combinations = util.get_cartesian_feature_product(data.get_all_possible_values())

    # create approach instance
    predictor = approach.approach.PerformancePredictior(my_basefolder)
    gold = []
    estimated = []
    max_diff = 0
    for feats in combinations:
        full_vector = data.get_exp("target/throughput", feats)
        gold_median = approach.approach.PerformancePredictior.measurement_point_aggregator(full_vector)
        gold.append(gold_median)
        est = predictor.get_one_measurement_point(feats)
        estimated.append(est)
        print(str(feats) + ": " + str(gold_median) + ", estimated: " + str(est)+".")
        diff = abs(est - gold_median)
        if diff > max_diff:
            max_diff = diff
            max_difffeat = feats
    mse = mean_squared_error(gold, estimated)
    mape = mean_absolute_percentage_error(gold, estimated)
    no_ms = predictor.get_total_number_of_measurements()
    print("Achieved a MAPE of "+ str(mape)+ " using a total of "+str(no_ms)+" measurement points.")
    print("Achieved a MSE of "+ str(mse)+ " using a total of "+str(no_ms)+" measurement points.")
    print("The maximal deviation happened at "+str(feats)+" with a diference of "+str(max_diff)+". ")

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100




if __name__ == "__main__":
    #calculate_and_plot_robustness_metrics()
    evaluate_measurement_point_selection()
