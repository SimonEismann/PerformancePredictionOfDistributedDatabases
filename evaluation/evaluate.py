"""
Main script for evaluation.
"""
import math
import time
import random

import boltons.statsutils as su
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from pylatexenc.latexencode import unicode_to_latex
from sklearn.metrics import mean_squared_error

from dataload import Datasetloader
from approach import metricanalyzer
from approach import util
import approach.approach
import approach.units.dataproviderunit as dp

# Configurable base folder for the experiments
my_basefolder = "..\\mowgli-ml-data\\results\\scalability-ycsb-write\\openstack\\cassandra"

# Target folders for calculated results
res_folder = "results"
res_robust_folder = res_folder + "\\robust-metrics"
res_efficiency_folder = res_folder + "\\efficiencies"


def calculate_and_compare_robustness_metrics(compare_folder, file=None):
    """
    Compares the given metrics for two different data sets. The basefolder is always compared with the given one
    :param compare_folder: The folder containing the second data set to compare against.
    :return: None.
    """
    # List of metrics to be analyzed
    metrics = [('Mean', np.mean),
               ('Median', np.median),
               ('95th percentile', lambda x: np.percentile(x, 95)),
               ('90th percentile', lambda x: np.percentile(x, 90)),
               ('80th percentile', lambda x: np.percentile(x, 80)),
               ('70th percentile', lambda x: np.percentile(x, 70)),
               ('Trimmed(5%) mean', lambda x: stats.trim_mean(x, 0.05)),
               ('Trimmed(10%) mean', lambda x: stats.trim_mean(x, 0.1)),
               ('Trimmed(20%) mean', lambda x: stats.trim_mean(x, 0.2)),
               ('Trimmed(30%) mean', lambda x: stats.trim_mean(x, 0.3)),
               ('Winsorized(5%) mean', lambda x: np.mean(stats.mstats.winsorize(x, [0.05, 0.05]))),
               ('Winsorized(10%) mean', lambda x: np.mean(stats.mstats.winsorize(x, [0.1, 0.1]))),
               ('Winsorized(20%) mean', lambda x: np.mean(stats.mstats.winsorize(x, [0.2, 0.2]))),
               ('Winsorized(30%) mean', lambda x: np.mean(stats.mstats.winsorize(x, [0.3, 0.3]))),
               ('Trimean', lambda x: su.Stats(x).trimean),
               ('Hodges-Lehmann', util.calculate_hodges_lehmann)
               ]
    ds = Datasetloader.load_data_set(compare_folder)
    performance = metricanalyzer.analyze_metrics(ds, metrics)

    d2 = Datasetloader.load_data_set(my_basefolder)
    p2 = metricanalyzer.analyze_metrics(d2, metrics)

    print("------------------------")
    print("LATEX: COMPARE")
    print("------------------------")
    strbuffer = "Metric\t&"
    any = None
    for measurement in performance:
        strbuffer = strbuffer + ("{0}: DS1,\t&{0}: DS2,\t&").format(measurement)
        any = measurement
    strbuffer = strbuffer[:-1] + "\\\\\\hline"
    print(strbuffer)
    for key, value in performance[any].items():
        # for each metric
        strbuffer = unicode_to_latex(str(key))
        for measurement in performance:
            strbuffer = strbuffer + ("\t&{0:.3f}").format(performance[measurement][key][0])
            strbuffer = strbuffer + ("\t&{0:.3f}").format(p2[measurement][key][0])
        print(strbuffer + "\\\\")

    print("------------------------")

    if file is not None:
        with open(file, "w+") as f:
            strbuffer = "Metric,"
            anything = None
            for measurement in performance:
                strbuffer = strbuffer + ("{0}: Data set 1, {0}: Data set 2,").format(measurement)
                anything = measurement
            f.write(strbuffer + "\n")
            for key, value in performance[anything].items():
                # for each metric
                strbuffer = str(key)
                for measurement in performance:
                    strbuffer = strbuffer + (", {0:.3f}, {1:.3f}").format(performance[measurement][key][0],
                                                                          p2[measurement][key][0])
                f.write(strbuffer + "\n")




def calculate_and_plot_robustness_metrics(print_individual=False, file=None, delete=0):
    """
    Analyzes the different robustness metrics by calculating them, plotting them, and printing a table with the results.
    :return: None
    """
    # List of metrics to be analyzed
    metrics = [('Mean', np.mean),
               ('Median', np.median),
               ('95th percentile', lambda x: np.percentile(x, 95)),
               ('90th percentile', lambda x: np.percentile(x, 90)),
               ('80th percentile', lambda x: np.percentile(x, 80)),
               ('70th percentile', lambda x: np.percentile(x, 70)),
               ('Trimmed(5%) mean', lambda x: stats.trim_mean(x, 0.05)),
               ('Trimmed(10%) mean', lambda x: stats.trim_mean(x, 0.1)),
               ('Trimmed(20%) mean', lambda x: stats.trim_mean(x, 0.2)),
               ('Trimmed(30%) mean', lambda x: stats.trim_mean(x, 0.3)),
               ('Winsorized(5%) mean', lambda x: np.mean(stats.mstats.winsorize(x, [0.05, 0.05]))),
               ('Winsorized(10%) mean', lambda x: np.mean(stats.mstats.winsorize(x, [0.1, 0.1]))),
               ('Winsorized(20%) mean', lambda x: np.mean(stats.mstats.winsorize(x, [0.2, 0.2]))),
               ('Winsorized(30%) mean', lambda x: np.mean(stats.mstats.winsorize(x, [0.3, 0.3]))),
               ('Trimean', lambda x: su.Stats(x).trimean),
               ('Hodges-Lehmann', util.calculate_hodges_lehmann)
               ]
    ds = Datasetloader.load_data_set(my_basefolder)
    # delete random samples from ds
    for i in range(delete):
        random_item_from_list = random.choice(ds.exps)
        ds.exps.remove(random_item_from_list)
    performance = metricanalyzer.analyze_metrics(ds, metrics)
    if print_individual:
        for measurement in performance:
            plot_robustness_barchart(measurement, res_robust_folder, performance[measurement])
            # print out performance tables
            print("LATEX: " + measurement + ":")
            print("------------------------")
            print("Metric& \tAvg& \tStd\\\\\\hline")
            for key, value in performance[measurement].items():
                print(unicode_to_latex(str(key)) + " & \t{0:.3f} & \t{1:.3f} \\\\".format(value[0], value[1]))
    if file is not None:
        with open(file, "w+") as f:
            strbuffer = "Metric,"
            any = None
            for measurement in performance:
                strbuffer = strbuffer + ("{0}: Avg, {0}: Std,").format(measurement)
                any = measurement
            f.write(strbuffer+"\n")
            for key, value in performance[any].items():
                # for each metric
                strbuffer = str(key)
                for measurement in performance:
                    strbuffer = strbuffer + (", {0:.3f}, {1:.3f}").format(performance[measurement][key][0],
                                                                 performance[measurement][key][1])
                f.write(strbuffer + "\n")
    print("------------------------")
    print("LATEX: COMBINED")
    print("------------------------")
    strbuffer = "Metric\t&"
    any = None
    for measurement in performance:
        strbuffer = strbuffer + ("{0}: Avg,\t&{0}: Std,\t&").format(measurement)
        any = measurement
    strbuffer = strbuffer[:-1] + "\\\\\\hline"
    print(strbuffer)
    for key, value in performance[any].items():
        # for each metric
        strbuffer = unicode_to_latex(str(key))
        for measurement in performance:
            strbuffer = strbuffer + ("\t&{0:.3f}\t& {1:.3f}").format(performance[measurement][key][0],
                                                         performance[measurement][key][1])
        print(strbuffer + "\\\\")

    print("------------------------")

    optimal_metrics = metricanalyzer.find_optimal_metric(ds, metrics)
    for metric in optimal_metrics:
        print("For metric " + metric + ", the best metric was " + str(
            optimal_metrics[metric][0]) + " with an average COV of " + str(optimal_metrics[metric][1]) + ".")


def plot_robustness_barchart(name, folder, metrics):
    """
    Plots a bar-chart of the robustness metrics.
    :param name: Name of the metric for the export file.
    :param folder: Folder to export the plot into.
    :param metrics: The actual metrics to plot.
    :return: None
    """
    avgs = [i[0] for i in list(metrics.values())]
    stds = [i[1] for i in list(metrics.values())]
    fig = plt.figure(figsize=(8, 6))
    rects = plt.bar(metrics.keys(), height=avgs, yerr=stds)
    plt.xticks(rotation=90)
    plt.tight_layout()
    # rects = ax.patches
    # Make labels.
    for rect in rects:
        height = rect.get_height()
        fig.axes[0].annotate('{:.3}'.format(height),
                             xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3),
                             textcoords="offset points",
                             ha='center', va='bottom')

    plt.show()
    plt.savefig(folder + "\\" + name + ".png")
    plt.close()


def get_real_prediction_value(dataprovider, features, target="target/throughput"):
    """
    Gets the real value for the given target features based on all available measurements.
    :param dataprovider: An instance of the dataprovider to use for loading the data.
    :param features: The target features to analyze.
    :param target: The target metric to search for. Default: "target/throughput"
    :return: The aggregated gold value, as well as the full measurement vector, based on which this value was derived.
    """
    full_vector = dataprovider.get_exp(target, features)
    gold_median = approach.approach.PerformancePredictior.MEASUREMENT_POINT_AGGREGATOR(full_vector)
    return gold_median, full_vector


def evaluate_measurement_point_selection(repetitions=100, file=None):
    """
    Evaluates the different techniques for measurement point selection.
    :return: None
    """
    resultsmape = {"approach": [], "gold": [], "1-point": [], "2-point": [], "3-point": [], "5-point": [],
                   "10-point": []}
    resultsrmse = {"approach": [], "gold": [], "1-point": [], "2-point": [], "3-point": [], "5-point": [],
                   "10-point": []}
    points = {"approach": [], "gold": [], "1-point": [], "2-point": [], "3-point": [], "5-point": [], "10-point": []}
    results_deviations = {"approach": [], "gold": [], "1-point": [], "2-point": [], "3-point": [], "5-point": [],
                   "10-point": []}
    accuracy_threshold = 0.01

    for i in range(repetitions):
        # create all feature instances
        data = dp.DataProvider(my_basefolder, approach.approach.PerformancePredictior.ROBUST_METRIC)
        combinations = util.get_cartesian_feature_product(data.get_all_possible_values())

        # create approach instance
        predictor = approach.approach.PerformancePredictior(my_basefolder).measurements
        max_diff = 0
        baselines = {"gold": [], "approach": [], "1-point": [], "2-point": [], "3-point": [], "5-point": [],
                     "10-point": []}
        for feats in combinations:
            gold_median, full_vector = get_real_prediction_value(dataprovider=data, features=feats)
            baselines["gold"].append(gold_median)
            compare_baseline_methods(baselines, full_vector)
            est = predictor.get_one_measurement_point(feats)
            baselines["approach"].append(est)
            # print(str(feats) + ": " + str(gold_median) + ", estimated: " + str(est)+".")
            diff = abs(est - gold_median)
            if diff > max_diff:
                max_diff = diff
                max_difffeat = feats

        rmse = math.sqrt(mean_squared_error(baselines["gold"], baselines["approach"]))
        mape = util.mean_absolute_percentage_error(baselines["gold"], baselines["approach"])
        no_of_deviated_points = get_deviations(baselines["gold"], baselines["approach"], threshold=accuracy_threshold)
        no_ms = predictor.get_total_number_of_measurements()
        print("Achieved a MAPE of " + str(mape) + " using a total of " + str(no_ms) + " measurement points.")
        print("Achieved a RMSE of " + str(rmse) + " using a total of " + str(no_ms) + " measurement points.")
        print("{0} points of a total of {1} points deviated by more than {2} percent. ".format(no_of_deviated_points, len(combinations), accuracy_threshold*100))
        print("The maximal deviation happened at " + str(max_difffeat) + " with a diference of " + str(max_diff) + ". ")
        for key in resultsmape:
            rmse = math.sqrt(mean_squared_error(baselines["gold"], baselines[key]))
            mape = util.mean_absolute_percentage_error(baselines["gold"], baselines[key])
            no_deviations = get_deviations(baselines["gold"], baselines[key], threshold=accuracy_threshold)
            resultsrmse[key].append(rmse)
            resultsmape[key].append(mape)
            results_deviations[key].append(no_deviations)

        points["approach"].append(no_ms / len(combinations))
        points["gold"].append(len(full_vector))
        points["1-point"].append(1)
        points["2-point"].append(2)
        points["3-point"].append(3)
        points["5-point"].append(5)
        points["10-point"].append(10)
    if file is not None:
        with open(file, "w+") as f:
            f.write("Approach , MAPE , RMSE , Deviations , # points\n")
            for key in resultsmape:
                f.write(("{0} , {1:.2f} , {2:.1f} , {3:.1f} {4:.2f}\n").format(str(key),
                                                                             np.mean(resultsmape[key]),
                                                                             np.mean(resultsrmse[key]),
                                                                             np.mean(results_deviations[key]),
                                                                             np.mean(points[key])))
    print("------------------------------------")
    print("Final Results.")
    for key in resultsmape:
        print(str(key) + ": Avg. MAPE: " + str(np.mean(resultsmape[key])) + "% using an average of " + str(
            np.mean(points[key])) + " measurement points.")
        print(str(key) + ": Avg. RMSE: " + str(np.mean(resultsrmse[key])) + " using an average of " + str(
            np.mean(points[key])) + " measurement points.")
    print("------------------------------------")
    print("LATEX TABLE.")
    print("Approach \t& MAPE \t& Deviations \t& # points\\\\")
    for key in resultsmape:
        print(("{0} \t& {1:.2f} \t& {2:.1f} \t& {3:.2f}\\\\").format(unicode_to_latex(str(key)),
                                                                     np.mean(resultsmape[key]),
                                                                     np.mean(results_deviations[key]),
                                                                     np.mean(points[key])))


def get_deviations(gold, predictions, threshold):
    """
    Calculates the number of feature combinations that deviate more than a given ratio from the given gold vectors.
    :param gold: List of gold values to compare against.
    :param predictions: List of predictions to compare against the gold values.
    :param threshold: Threshold to apply as a relative ratio.
    :return: Number of points that achieved a higher deviation.
    """
    no_of_deviated_points = 0
    for index, gold in enumerate(gold):
        pred = predictions[index]
        diff = abs(gold - pred)
        if diff / gold > threshold:
            no_of_deviated_points = no_of_deviated_points + 1
    return no_of_deviated_points


def compare_baseline_methods(results, values):
    """
    Adds the performance of the baseline methods to the given result lists.
    :param results: The dictionary containing the result lists.
    :param values: The values to aggregate from.
    :return: None
    """
    results["1-point"].append(values[0])
    results["2-point"].append(np.median(values[0:2]))
    results["3-point"].append(np.median(values[0:3]))
    results["5-point"].append(np.median(values[0:6]))
    results["10-point"].append(np.median(values[0:11]))


def evaluate_total_workflow():
    """
    Execute and evaluate on model training workflow.
    :return: None
    """
    # create approach instance
    predictor = approach.approach.PerformancePredictior(my_basefolder)
    predictor.start_training_workflow()
    print("Accuracy: {0}.".format(get_model_accuracy(predictor)))


def get_model_accuracy(predictor):
    """
    Calculate and return the accuracy of one performance model instance.
    :param predictor: The performance model to evaluate
    :return: The calculated MAPE of this performance model.
    """
    preds = []
    reals = []
    feature_set = predictor.configuration_provider.get_feature_space()
    measured_set = predictor.measurements.get_available_feature_set()
    for validation in feature_set:
        if validation in measured_set:
            # do nothing
            pass
        else:
            preds.append(predictor.get_prediction(validation))
            gold, full_vector = get_real_prediction_value(predictor.dataprovider, validation)
            reals.append(gold)
            # print("Features:", validation)
            # print("Prediction, Label, Error:", predictor.get_prediction(validation), gold, predictor.get_prediction(validation)/gold)
    rmse = math.sqrt(mean_squared_error(reals, preds))
    mape = util.mean_absolute_percentage_error(reals, preds)
    mae = mean_absolute_error(reals, preds)
    return mape


def get_approach_efficiency(model_type, repetitions=50):
    """
    Analyses the performance of a whole workflow execution based on a single model to use. Not multi-threading safe!
    :param model_type: The model to apply and to evaluate.
    :param repetitions: The number of repetitions to repeat the evaluation. Default: 50
    :return: Lists of achieved errors, required measurements, required configurations, and required computation times.
    """
    # Careful! This breaks multi-thread compatability!
    approach.approach.PerformancePredictior.MODEL_TYPE = model_type
    accs = []
    meas = []
    confs = []
    times = []
    for i in range(0, repetitions):
        predictor = approach.approach.PerformancePredictior(my_basefolder)
        start = time.time()
        predictor.start_training_workflow()
        end = time.time()
        times.append((end - start))
        accs.append(get_model_accuracy(predictor))
        meas.append(predictor.measurements.get_total_number_of_measurements())
        confs.append(len(predictor.measurements.get_available_feature_set()))
    return accs, meas, confs, times


def evaluate_accuracy_curves(approaches, repetitions=50):
    """
    Evaluate and plot the accuracy of different approaches in comparison to the target threshold.
    :param approaches: A list of approaches to include.
    :param repetitions: The number of repetitions. Default: 50.
    :return: None
    """
    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for name, app, color, marker in approaches:
        avgs = []
        mins = []
        maxs = []
        confis = []
        ranges = np.arange(0, 1, 0.05)
        for i in ranges:
            approach.approach.PerformancePredictior.ACC_THRESHOLD = - i
            accs, meas, confs, times = get_approach_efficiency(app, repetitions=repetitions)
            avgs.append(np.mean(accs))
            mins.append(np.min(accs))
            maxs.append(np.max(accs))
            confis.append(np.mean(confs))
        ax.fill_between(ranges, mins, maxs, alpha=0.3, facecolor=color)
        ax.plot(ranges, avgs, color=color)
        for j in range(len(confis)):
            plt.text(ranges[j], avgs[j], "{0:.1f}".format(confis[j]), color=color)

    plt.xlabel('Target accuracy threshold')
    plt.ylabel('Prediction error (MAPE)')
    plt.ylim((0, 100))
    plt.show()
    fig.savefig(res_efficiency_folder + r"\performance.pdf")


def evaluate_efficiency(repetitions=50, file=None, figure=None):
    """
    Creates a table and a scatter plot (connected with dashed lines) of the efficiencies of all available approaches.
    :param repetitions: The number of repetitions. Default: 50
    :return: None
    """
    approaches = [('LinReg', linear_model.LinearRegression(), "red", "o"),
                  ('Ridge', linear_model.Ridge(), "green", "v"),
                  ('ElasticNet', linear_model.ElasticNet(), "orange", "2"),
                  ('BayesianRidge', linear_model.BayesianRidge(), "blue", "3"),
                  ('HuberRegressor', linear_model.HuberRegressor(), "black", ">"),
                  ('GBDT', GradientBoostingRegressor(), "gray", "*"),
                  ('RandomForest', RandomForestRegressor(), "purple", "+"),
                  ('SVR', linear_model.SGDRegressor(), "pink", "x"),
                  ('ZeroR', DummyRegressor(), "brown", "d")
                  ]
    # for each model in approaches
    thresholds = [-0.1, -0.15, -0.2, -0.25, -0.3]
    names = []
    accs = []
    meass = []
    confs = []
    times = []

    print("Approach, Avg. MAPE, Avg. Measurements, Avg. Configurations, Avg. Time (s)\n")
    for model in approaches:
        for t in thresholds:
            approach.approach.PerformancePredictior.ACC_THRESHOLD = t
            # actural execution
            acc, meas, conf, time = get_approach_efficiency(model[1], repetitions=repetitions)
            # file export
            print("{0}, {1}, {2}, {3}, {4}\n".format(model[0], acc, meas, conf, time))

            print(
                "Approach {0}: {1}, {2}, {3}, {4}".format(unicode_to_latex(str(model[0] + str(t))), acc, meas, conf,
                                                          time))

            # storing for figure
            names.append(model[0] + str(t))
            accs.append(np.mean(acc))
            meass.append(np.mean(meas))
            confs.append(np.mean(conf))
            times.append(np.mean(time))

    print("-------------------")
    print("LATEX")
    print(
        "Approach \t& Avg. MAPE \t& Avg. Measurements (Max. 900) \t& Avg. Configrations (Max. 90) \t& Avg. Time (s)\\\\")
    for i in range(0, len(names)):
        # latex export
        print("{0}\t& {1:.2f}\t& {2:.2f}\t& {3:.2f}\t& {4:.2f}\\\\".format(unicode_to_latex(str(names[i])), accs[i],
                                                                           meass[i], confs[i], times[i]))

    if file is not None:
        with open(file, "w+") as f:
            f.write(
                "Approach , Avg. MAPE , Avg. Measurements (Max. 900) , Avg. Configrations (Max. 90) ,Avg. Time (s)\n")
            for i in range(0, len(names)):
                # latex export
                f.write("{0}, {1:.2f}, {2:.2f}, {3:.2f}, {4:.2f}\n".format(unicode_to_latex(str(names[i])),
                                                                                   accs[i],
                                                                                   meass[i], confs[i], times[i]))

    # Create plot
    if figure is not None:
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(1, 1, 1)

        for i in range(len(approaches)):
            conflist = []
            acclist = []
            for k in range(len(thresholds)):
                conflist.append(confs[i * len(thresholds) + k])
                acclist.append(accs[i * len(thresholds) + k])
            # ax.scatter(conflist, acclist, c=approaches[i][2], edgecolors='none', s=30,
            #           label=approaches[i][0], marker=approaches[i][3])
            ax.plot(conflist, acclist, c=approaches[i][2], linestyle='dashed', markersize=4,
                    label=approaches[i][0], marker=approaches[i][3], linewidth=0.5)

        # print scatter plot
        plt.xlabel('Required configuration points')
        plt.ylabel('Prediction error (MAPE)')
        plt.ylim((0, 50))
        plt.xlim((0, 85))
        plt.legend(loc=1)
        fig.savefig(figure)


if __name__ == "__main__":
    # Create output folders
    compare_folder = r"C:\Users\Johannes\Desktop\combined"
    # Experiment 1:
    #calculate_and_plot_robustness_metrics(False, file=res_folder+"\\TableIII.csv", delete=0)
    calculate_and_compare_robustness_metrics(compare_folder)
    # Experiment 2:
    #evaluate_measurement_point_selection(100, file=res_folder+"\\TableIV.csv")
    # Experiment 3:
    #evaluate_efficiency(3, file=res_folder+"\\TableV.csv", figure=res_folder+"\\Figure3.pdf")
