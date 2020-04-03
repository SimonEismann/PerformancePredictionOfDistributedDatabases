# This class takes care of training and evaluating performance models of a specified type.
class PerformanceModelProvider:

    # Initialize this object and specify the model to use for modeling.
    def __init__(self, model_type):
        self.model = model_type

    def create_model(self, measurements):
        print(measurements)
        model = PerformanceModel(model_to_use=self.model, training_data=measurements, label_data=measurements)


# Instances of this class represent our performance model.
class PerformanceModel:

    def __init__(self, model_to_use, measurements):
        self.training = measurements
        self.model = model_to_use
