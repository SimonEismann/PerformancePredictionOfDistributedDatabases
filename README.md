# Performance Prediction Of Distributed Databases
Setting up this project requires downloading the measurement data and the configuration of the dependency management.

## Measurement Data ##
Checkout the measurement data from git into the base folder of this project, the resulting folder should be called mowgli-ml-data. The .gitignore is configured to exclude this folder.

    git clone https://omi-gitlab.e-technik.uni-ulm.de/ds/mowgli-ml-data.git

## Dependency Management ##
To setup a development environment for this project using pycharm and pipenv, follow these steps:
    
1. Create a new pycharm project in the folder `PerformancePredictionOfDistributedDatabases`.
2. In the following dialog select a python 3.6 base interpreter and the pipenv executable:
![alt text](https://github.com/SimonEismann/PerformancePredictionOfDistributedDatabases/blob/master/images/setup.png "")
3. Select 'Yes' in the following dialog asking 'Would you like to create a project from existing sources instead'?
