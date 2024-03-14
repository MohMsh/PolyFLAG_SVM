import threading
import numpy as np
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report


class GlobalModel:

    def __init__(self, w_agg, b_agg, clients_parameters, aggregation_method, x_test, y_test, current_round):
        self.aggregation_method = aggregation_method
        self.w_agg = w_agg
        self.b_agg = b_agg
        self.clients_parameters = clients_parameters
        self.lock = threading.Lock()
        self.x_test = x_test
        self.y_test = y_test
        self.current_round = current_round

    def aggregate(self):
        features_number = len(self.x_test)

        # In the following code, w stands for weight and b for bias
        # Initialize w_best and b_best as zero vectors with the same number of columns as X_test
        w_best = np.zeros(self.x_test.shape[1])
        b_best = 0

        print("\n\t==========================================\n"
              "\tGlobal model training Initiated (Round ", self.current_round, ")",
              "\n\t==========================================\n")

        # Use the aggregator function to combine the parameters from all clients
        # there are four aggregators implemented in our code, we can choose one of them

        if self.aggregation_method == "average_aggregation":
            w_agg, b_agg = self.average_aggregator()
        elif self.aggregation_method == "clipped_avg_aggregator":
            w_agg, b_agg = self.clipped_average_aggregator()
        elif self.aggregation_method == "dp_avg_aggregator":
            w_agg, b_agg = self.dp_average_aggregator()
        elif self.aggregation_method == "momentum_aggregation":
            w_agg, b_agg = self.momentum_aggregator()
        else:
            w_agg, b_agg = self.average_aggregator()

        # Check if the accuracy of the aggregated parameters is better than the current best parameters
        if self.measure_accuracy(self.x_test, self.y_test, w_agg, b_agg) \
                > self.measure_accuracy(self.x_test, self.y_test, w_best, b_best):
            # If it is, update the best weights and biases
            w_best = copy.deepcopy(w_agg)
            b_best = copy.deepcopy(b_agg)

        # Evaluate the model and print the results
        self.assessModel(self.x_test, self.y_test, w_best, b_best)
        return w_agg, b_agg

    # This method calculates the accuracy of the predictions using the parameters w and b.
    def measure_accuracy(self, x_test, y_test, w, b):
        return accuracy_score(y_test, self.predict(x_test, w, b)) * 100

    # This method predicts the sentiment of the input text using the parameters weight (w) and bias (b).
    def predict(self, x_test, w, b):
        approx = np.dot(x_test, w) - b
        approx = np.sign(approx)
        return np.where(approx < 0, 0, 1)

    def assessModel(self, x_test, y_test, w, b):

        print("\tGlobal Model Performance Metrics:\n\t==========================================")

        # Model Accuracy: how often is the classifier correct?
        print("\t\t==> Accuracy: ", round((accuracy_score(y_test, self.predict(x_test, w, b))) * 100, 2), "%")

        # Model Precision: what percentage of positive tuples are labeled as such?
        print("\t\t==> Precision: ", round((precision_score(y_test, self.predict(x_test, w, b))) * 100, 2), "%")

        # Model Recall: what percentage of positive tuples are labelled as such?
        print("\t\t==> Recall: ", round((recall_score(y_test, self.predict(x_test, w, b))) * 100, 2), "%")

        # Model F1 Score: F1 Score might be a better measure to use
        # if we need to seek a balance between Precision and Recall
        print("\t\t==> F1 Score: ", round((f1_score(y_test, self.predict(x_test, w, b))) * 100, 2), "%")

        # Model Specificity: a model's ability to predict true negatives of each available category
        print("\t\t==> Specificity: ",
              round((recall_score(y_test, self.predict(x_test, w, b), pos_label=0)) * 100, 2), "%")

        # Model Negative Predictive Value (NPV):
        print("\t\t==> Negative Predictive Value (NPV): ",
              round((precision_score(y_test, self.predict(x_test, w, b), pos_label=0)) * 100, 2), "%")

        # calculate confusion matrix
        cm = confusion_matrix(y_test, self.predict(x_test, w, b))

        # calculate TP, TN, FP and FN
        TP = cm[0, 0]
        TN = cm[1, 1]
        FP = cm[0, 1]
        FN = cm[1, 0]

        # print classification error
        classification_error = (FP + FN) / float(TP + TN + FP + FN)
        print("\n\t==> Classification error : {0:0.4f}".format(classification_error))

        print("\n\t Classification Report:\n\t==========================================")
        print(classification_report(y_test, self.predict(x_test, w, b)))
        print("===============================================================================\n")
        # ==========================================================================================

    def average_aggregator(self):

        self.lock.acquire()
        parameters = self.clients_parameters
        self.lock.release()

        clients_number = len(parameters)

        w_average = np.zeros(len(parameters[0]["client_weights"]))
        b_average = 0

        for parameter in parameters:
            w_average += [x + y for x, y in zip(parameter["client_weights"], w_average)]
            b_average += parameter["client_bias"]
        w_average = [x / clients_number for x in w_average]
        b_average /= clients_number

        return w_average, b_average

    def clipped_average_aggregator(self, clip_value=0.01):
        """
        Federated Averaging with Weight Clipping:
        In this approach, the weights from the participating clients are first clipped to a predefined range to limit
        the effect of outlier clients. Then, the average of the clipped weights is taken to obtain the updated global
         model weights. This can help to reduce the influence of clients with poorly performing models
        :param clip_value:
        :return:
        """
        self.lock.acquire()
        parameters = self.clients_parameters
        self.lock.release()

        clients_number = len(parameters)

        w_average = np.zeros(len(parameters[0]["client_weights"]))
        b_average = 0

        for parameter in parameters:
            # clip the weights
            clipped_weights = np.clip(parameter["client_weights"], -clip_value, clip_value)

            # compute the average
            w_average += clipped_weights
            b_average += parameter["client_bias"]

        w_average /= clients_number
        b_average /= clients_number

        return w_average, b_average

    def dp_average_aggregator(self, sigma=1.0):
        """
        Federated Averaging with Differential Privacy:
        Differential privacy is a technique that adds noise to the data to protect the privacy of individual clients.
        In the context of federated learning, this can be applied to the weights of the models to make them more robust
        against attacks that try to extract information about individual clients. This can also help to improve the
        generalization performance of the global model.
        :param sigma:
        :return:
        """
        self.lock.acquire()
        parameters = self.clients_parameters
        self.lock.release()

        clients_number = len(parameters)

        w_average = np.zeros(len(parameters[0]["client_weights"]))
        b_average = 0

        for parameter in parameters:
            # add noise to the weights
            noisy_weights = parameter["client_weights"] + \
                            np.random.normal(0, sigma, size=parameter["client_weights"].shape)

            # compute the average
            w_average += noisy_weights
            b_average += parameter["client_bias"]

        w_average /= clients_number
        b_average /= clients_number

        return w_average, b_average

    def momentum_aggregator(self, momentum=0.9):
        """
        Federated Averaging with Momentum:
        In this approach, the updated weights from each client are weighted according to a momentum parameter,
        which favors more recent updates over older ones. This can help to smooth out the updates and reduce the
        effect of noisy or poorly performing clients.
        :param momentum:
        :return:
        """
        self.lock.acquire()
        parameters = self.clients_parameters
        self.lock.release()

        clients_number = len(parameters)

        w_average = np.zeros(len(parameters[0]["client_weights"]))
        b_average = 0

        for parameter in parameters:
            # compute the weighted average
            w_average = momentum * w_average + (1 - momentum) * parameter["client_weights"]
            b_average = momentum * b_average + (1 - momentum) * parameter["client_bias"]

        w_average /= clients_number
        b_average /= clients_number

        return w_average, b_average
