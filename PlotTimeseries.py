import Data as Data
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ds = Data.load_data_set()
    df = ds.calculate_robust_metric(lambda x: x)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for index, row in df.iterrows():
        configname = row['configuration']
        timeseries = row['target/throughput']
        plt.figure(figsize=(12,6))
        plt.title(configname)
        plt.xlabel('Time [10s]')
        plt.ylabel('Throughput [Req/s]')
        plt.ylim([0, 20000])
        means = []
        i = 0
        for series in timeseries:
            plt.plot(series,color=colors[i])
            means.append(np.mean(series))
            plt.hlines(np.mean(series), xmin=0, xmax=len(series)-1,color=colors[i])
            i = i +1
        plt.hlines(np.median(means), xmin=0, xmax=len(series)-1)
        plt.savefig('results/timeseries/throughput-' + configname + ".png")
        plt.close()

    for index, row in df.iterrows():
        configname = row['configuration']
        timeseries = row['target/latency']
        plt.figure(figsize=(12,6))
        plt.title(configname)
        plt.xlabel('Time [10s]')
        plt.ylabel('Latency [?s]')
        plt.ylim([0, 7500])
        means = []
        for series in timeseries:
            plt.plot(series)
            means.append(np.mean(series))
            plt.hlines(np.mean(series), xmin=0, xmax=len(series)-1)
        plt.hlines(np.median(means), xmin=0, xmax=len(series)-1)
        plt.savefig('results/timeseries/latency-' + configname + ".png")
        plt.close()