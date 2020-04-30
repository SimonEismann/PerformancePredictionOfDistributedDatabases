# Main script for evaluation purposes
import math

import boltons.statsutils as su
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

import Data
from approach import metricanalyzer
from approach import util
import approach.approach
import approach.dataprovider as dp
import time
from pylatexenc.latexencode import unicode_to_latex
from sklearn.metrics import mean_squared_error

# Configurable base folder for the experiments
my_basefolder = "..\\mowgli-ml-data\\results\\scalability-ycsb-write\\openstack\\cassandra"

res_folder = "results"
res_robust_folder = res_folder + "\\robust-metrics"
res_efficiency_folder = res_folder + "\\efficiencies"


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
        plot_robustness_barchart(measurement, res_robust_folder, performance[measurement])
        # print out performance tables
        print("LATEX: "+measurement+":")
        print("------------------------")
        print("Metric& \tAvg& \tStd\\\\\\hline")
        for key, value in performance[measurement].items():
            print(unicode_to_latex(str(key))+ " & \t{0:.3f} & \t{1:.3f} \\\\".format(value[0], value[1]))
        print("------------------------")
    print("LATEX: COMBINED")
    print("------------------------")
    tmp = "Metric\t&"
    any = None
    for measurement in performance:
        tmp = tmp + ("{0}: Avg,\t&{0}: Std,\t&").format(measurement)
        any = measurement
    tmp = tmp[:-1] + "\\\\\\hline"
    print(tmp)
    for key, value in performance[any].items():
        # for each metric
        tmp = unicode_to_latex(str(key))
        for measurement in performance:
             tmp = tmp + ("\t&{0:.3f}\t& {1:.3f}").format(performance[measurement][key][0], performance[measurement][key][1])
        print(tmp+ "\\\\")



    print("------------------------")

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


def get_real_prediction_value(dataprovider, features, target="target/throughput"):
    full_vector = dataprovider.get_exp(target, features)
    gold_median = approach.approach.PerformancePredictior.MEASUREMENT_POINT_AGGREGATOR(full_vector)
    return gold_median, full_vector


def evaluate_measurement_point_selection():
    resultsmape = {"approach": [], "gold":[], "1-point":[], "2-point":[], "3-point":[], "5-point":[], "10-point":[]}
    resultsrmse = {"approach": [], "gold": [], "1-point": [], "2-point": [], "3-point": [], "5-point": [], "10-point": []}
    points = {"approach": [], "gold": [], "1-point": [], "2-point": [], "3-point":[], "5-point": [], "10-point": []}

    for i in range(100):
        # create all feature instances
        data = dp.DataProvider(my_basefolder, approach.approach.PerformancePredictior.ROBUST_METRIC)
        combinations = util.get_cartesian_feature_product(data.get_all_possible_values())

        # create approach instance
        predictor = approach.approach.PerformancePredictior(my_basefolder).measurements
        max_diff = 0
        baselines = {"gold":[], "approach":[], "1-point": [], "2-point": [], "3-point":[], "5-point": [], "10-point": []}
        for feats in combinations:
            gold_median, full_vector = get_real_prediction_value(dataprovider=data, features=feats)
            baselines["gold"].append(gold_median)
            compare_baseline_methods(baselines, full_vector)
            est = predictor.get_one_measurement_point(feats)
            baselines["approach"].append(est)
            #print(str(feats) + ": " + str(gold_median) + ", estimated: " + str(est)+".")
            diff = abs(est - gold_median)
            if diff > max_diff:
                max_diff = diff
                max_difffeat = feats
        rmse = math.sqrt(mean_squared_error(baselines["gold"], baselines["approach"]))
        mape = mean_absolute_percentage_error(baselines["gold"], baselines["approach"])
        no_ms = predictor.get_total_number_of_measurements()
        print("Achieved a MAPE of "+ str(mape)+ " using a total of "+str(no_ms)+" measurement points.")
        print("Achieved a RMSE of "+ str(rmse)+ " using a total of "+str(no_ms)+" measurement points.")
        print("The maximal deviation happened at "+str(max_difffeat)+" with a diference of "+str(max_diff)+". ")
        for key in resultsmape:
            rmse = math.sqrt(mean_squared_error(baselines["gold"], baselines[key]))
            mape = mean_absolute_percentage_error(baselines["gold"], baselines[key])
            resultsrmse[key].append(rmse)
            resultsmape[key].append(mape)
        points["approach"].append(no_ms/len(combinations))
        points["gold"].append(len(full_vector))
        points["1-point"].append(1)
        points["2-point"].append(2)
        points["3-point"].append(3)
        points["5-point"].append(5)
        points["10-point"].append(10)
    print("------------------------------------")
    print("Final Results.")
    for key in resultsmape:
        print(str(key)+": Avg. MAPE: "+str(np.mean(resultsmape[key]))+"% using an average of "+str(np.mean(points[key])) + " measurement points.")
        print(str(key) + ": Avg. RMSE: " + str(np.mean(resultsrmse[key])) + " using an average of " + str(
            np.mean(points[key])) + " measurement points.")
    print("------------------------------------")
    print("LATEX TABLE.")
    print("Approach \t& MAPE \t& RMSE \t& # points\\\\")
    for key in resultsmape:
        print(("{0} \t& {1:.2f} \t& {2:.1f} \t& {3:.2f}\\\\").format(unicode_to_latex(str(key)), np.mean(resultsmape[key]), np.mean(resultsrmse[key]), np.mean(points[key])))

def compare_baseline_methods(results, values):
    results["1-point"].append(values[0])
    results["2-point"].append(np.median(values[0:2]))
    results["3-point"].append(np.median(values[0:3]))
    results["5-point"].append(np.median(values[0:6]))
    results["10-point"].append(np.median(values[0:11]))


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate_total_workflow():
    # create approach instance
    predictor = approach.approach.PerformancePredictior(my_basefolder)
    predictor.start_training_workflow()
    print("Accuracy: {0}.".format(get_model_accuracy(predictor)))


def get_model_accuracy(predictor):
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
    rmse = math.sqrt(mean_squared_error(reals, preds))
    mape = mean_absolute_percentage_error(reals, preds)
    return mape


def get_approach_efficiency(model_type, repetitions=50):
    # Careful! This breaks multi-thread compatability!
    approach.approach.PerformancePredictior.MODEL_TYPE = model_type
    accs = []
    meas = []
    confs = []
    times = []
    for i in range(0,repetitions):
        predictor = approach.approach.PerformancePredictior(my_basefolder)
        start = time.time()
        predictor.start_training_workflow()
        end = time.time()
        times.append((end - start))
        accs.append(get_model_accuracy(predictor))
        meas.append(predictor.measurements.get_total_number_of_measurements())
        confs.append(len(predictor.measurements.get_available_feature_set()))
    return np.mean(accs), np.mean(meas), np.mean(confs), np.mean(times)


def evaluate_efficiency_scatter_plot():
    repetitions = 5
    approaches = [('LinReg', linear_model.LinearRegression()),
                  ('Ridge', linear_model.Ridge()),
                  ('ElasticNet', linear_model.ElasticNet()),
                  ('BayesianRidge', linear_model.BayesianRidge()),
                  ('HuberRegressor', linear_model.HuberRegressor()),
                  #('MLP', MLPRegressor(max_iter=10000)),
                  ('GBDT', GradientBoostingRegressor()),
                  ('RandomForest', RandomForestRegressor()),
                  ('SVR', linear_model.SGDRegressor()),
                  ]
    # for each model in approaches
    names = []
    accs = []
    meass = []
    confs = []
    times = []
    with open(res_efficiency_folder+"\\efficiencies.csv", "w+") as file:
        file.write("Approach, Avg. MAPE, Avg. Measurements, Avg. Configurations, Avg. Time (s)\n")
        for model in approaches:
            # actural execution
            acc, meas, conf, time = get_approach_efficiency(model[1], repetitions=repetitions)
            # file export
            file.write("{0}, {1}, {2}, {3}, {4}\n".format(model[0], acc, meas, conf, time))
            file.flush()

            print("Approach {0}: {1}, {2}, {3}, {4}".format(unicode_to_latex(str(model[0])), acc, meas, conf, time))

            # storing for figure
            names.append(model[0])
            accs.append(acc)
            meass.append(meas)
            confs.append(conf)
            times.append(time)

    print("-------------------")
    print("LATEX")
    print("Approach \t& Avg. MAPE \t& Avg. Measurements (Max. 900) \t& Avg. Configrations (Max. 90) \t& Avg. Time (s)\\\\")
    for i in range(0, len(names)):
        # latex export
        print("{0}\t& {1:.2f}\t& {2:.2f}\t& {3:.2f}\t& {4:.2f}\\\\".format(unicode_to_latex(str(names[i])), accs[i], meass[i], confs[i], times[i]))

    pass


if __name__ == "__main__":
    #calculate_and_plot_robustness_metrics()
    #evaluate_measurement_point_selection()
    #evaluate_total_workflow()
    evaluate_efficiency_scatter_plot()