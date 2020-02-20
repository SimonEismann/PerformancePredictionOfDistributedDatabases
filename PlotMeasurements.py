import numpy as np
import Data as Data
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
import boltons.statsutils as su
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
#from pyearth import Earth
from sklearn.neural_network import MLPRegressor

from sklearn import linear_model

from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette(sns.color_palette("hls", 20))


distplots = True
prediction = True


def calculateHL(l_input):
    l_avgs = []
    k = 0
    j = 0
    while k < len(l_input):
        while j < len(l_input):
            l_avgs.append(np.mean([l_input[k], l_input[j]]))
            j = j + 1
        k = k + 1
        j = k
    return np.median(l_avgs)


if __name__ == "__main__":
    datasets = [('smallVM', Data.load_small_vm_data_set()),
                ('largeVM', Data.load_large_vm_data_set()),
                ('smallAndLargeVM', Data.load_data_set())
                ]
    approaches = [('LinReg', linear_model.LinearRegression()),
                  ('HuberRegressor', linear_model.HuberRegressor(epsilon=100)),
                  ('Ridge', linear_model.Ridge()),
                  ('ElasticNet', linear_model.ElasticNet()),
                  ('BayesianRidge', linear_model.BayesianRidge()),
                  #('MARS', Earth()),
                  ('RandomForest', RandomForestRegressor(n_estimators=10, random_state=42)),
                  ('SVR', linear_model.SGDRegressor(max_iter=1000, tol=1e-3)),
               ]
    metrics = [('Mean', np.mean),
               #('Median', np.median),
               #('Max', np.max),
               #('Min', np.min),
               ('95th percentile', lambda x: np.percentile(x, 95))
               #('90th percentile', lambda x: np.percentile(x, 90)),
               #('80th percentile', lambda x: np.percentile(x, 80)),
               #('70th percentile', lambda x: np.percentile(x, 70)),
               #('Trimmed(5%) mean', lambda x: stats.trim_mean(x, 0.05)),
               #('Trimmed(10%) mean', lambda x: stats.trim_mean(x, 0.1)),
               #('Trimmed(20%) mean', lambda x: stats.trim_mean(x, 0.2)),
               #('Trimmed(30%) mean', lambda x: stats.trim_mean(x, 0.3)),
               #('Winzorized(5%) mean', lambda x: np.mean(stats.mstats.winsorize(x, [0.05, 0.05]))),
               #('Winzorized(10%) mean', lambda x: np.mean(stats.mstats.winsorize(x, [0.1, 0.1]))),
               #('Winzorized(20%) mean', lambda x: np.mean(stats.mstats.winsorize(x, [0.2, 0.2]))),
               #('Winzorized(30%) mean', lambda x: np.mean(stats.mstats.winsorize(x, [0.3, 0.3]))),
               #('Trimean', lambda x: su.Stats(x).trimean),
               #('Hodges-Lehmann', calculateHL)
               ]

    # Plot
    if distplots:
        # Load Data
        ds = Data.load_data_set()

        for metric in metrics:
            df = ds.calculate_robust_metric(metric[1])
            all_tps = []
            all_configs = []
            for index, row in df.iterrows():
                for value in row['target/throughput']:
                    all_tps.append(value)
                    all_configs.append(row['configuration'])

            plt.figure(figsize=(30, 15))
            plt.scatter(all_configs, all_tps, label="Measurements")
            plt.scatter(df['configuration'], df['target/throughput'].apply(np.mean), label="Mean")
            plt.scatter(df['configuration'], df['target/throughput'].apply(np.median), label="Median")
            for i in range(0, 58, 2):
                plt.axvspan(i + 0.5, i+1.5, facecolor='0.2', alpha=0.3)
            plt.xticks(rotation=90, fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel("Configuration", fontsize=20)
            plt.ylabel(metric[0] + "Throughput", fontsize=20)
            plt.title(metric[0] + " Throughput", fontsize=26)
            plt.xlim(-0.5, 57.5)
            plt.legend(fontsize=20)
            plt.tight_layout()
            plt.savefig("results/distributionplots/" + metric[0] + ".png")
            print("Finished file " + metric[0] + ".png")

    # Regression
    if prediction:
        for approach in approaches:
            for dataset in datasets:
                plt.figure(figsize=(30, 15))
                for metric in metrics:
                    df = dataset[1].calculate_robust_metric(metric[1])
                    y = np.asarray(df['target/throughput'].apply(np.median))
                    x = np.asarray(df.drop(columns=['configuration', 'target/throughput']))
                    orderedMapes = []


                    for i in range(1, len(y)):
                        mapes = []
                        for j in range(1, 50):
                            xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=(len(y)-i)/len(y), random_state=j+20)
                            model = approach[1]
                            model.fit(xTrain, yTrain)
                            yPred = model.predict(xTest)
                            errors = abs(yPred - np.asarray(yTest))
                            mape = 100 * (errors / np.asarray(yTest))
                            mapes.append(np.mean(mape))
                        orderedMapes.append(np.mean(mapes))

                    if metric[0] != 'Min':
                        plt.plot(range(1, len(y)), orderedMapes, label=metric[0])

                plt.ylabel("MAPE [%]", fontsize=24)
                plt.xlabel("Trainingset size", fontsize=24)
                plt.title(approach[0] + "_" + dataset[0], fontsize=28)
                plt.legend(fontsize=20, ncol=3)
                plt.xticks(fontsize=24)
                plt.yticks(fontsize=24)
                plt.xlim(1, len(y)-1)
                plt.ylim(0, 30)
                plt.savefig('results/predictionaccuracy/' + approach[0] + "_" + dataset[0] + '.png')
                plt.close()
                print(approach[0], dataset[0])