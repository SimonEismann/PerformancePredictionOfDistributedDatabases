# This script evaluates the available data and calculates the best metrics in terms of robustness.
import Data
import numpy as np

myBasefolder = "..\\mowgli-ml-data\\results\\scalability-ycsb-write\\openstack\\cassandra"







if __name__ == "__main__":
    Data.basefolder = myBasefolder
    ds = Data.load_data_set()
    print(ds.calculate_robust_metric(np.mean).dtypes)