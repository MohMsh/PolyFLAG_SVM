import copy
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, \
    f1_score


class SVM_Linear:
    """
    this class holds the SVM model at the client side
    """

    def __init__(self, x_train, y_train, x_test, y_test, validation=False,
                 learning_rate=None, lambda_param=None, iterations=None):
        """
        class constructor
        :param x_train: data features in training set
        :param y_train: class in training set
        :param x_test: data features in test set
        :param y_test: class in test set
        :param learning_rate: learning rate (hyperparameter)
        :param lambda_param: lambda parameter (hyperparameter)
        :param iterations: number of iterations while training
        :return:
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.validation = validation
        self.learning_rate = learning_rate
        self.lambda_parameter = lambda_param
        self.iterations = iterations
        self.w = np.array([])  # weight
        self.b = None  # bias

    def GradientUpdate(self, x_train, y_train, x_val=None, y_val=None):
        """
        this function works on gradients updating while training
        :param x_train: data features in training set
        :param y_train: class in training set
        :param x_val: validation input data
        :param y_val: validation target data
        :return:
        """
        n_samples, n_features = x_train.shape
        y_ = np.where(y_train <= 0, -1, 1)  # changing the target data to -1 or 1

        if self.w.size == 0 and self.b is None:  # if w and b are not initialized
            self.w = np.zeros(n_features)  # initialize w as zero array
            self.b = 0  # initialize b as 0

        w_best = np.zeros(n_features)  # stores the best weight vector
        b_best = 0  # stores the best bias value

        acc_list = []  # list to store accuracy
        for i in range(0, self.iterations):
            for idx, x_i in enumerate(x_train):  # iterate over the training data
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1  # calculate the condition
                if condition:
                    # if the condition is True, update w according to the formula
                    self.w -= self.learning_rate * (2 * self.lambda_parameter * self.w)
                else:
                    # update w according to the formula
                    self.w -= self.learning_rate * (2 * self.lambda_parameter * self.w - np.dot(x_i, y_[idx]))
                    # update b according to the formula
                    self.b -= self.learning_rate * y_[idx]
            if i % 10 == 0 and self.validation:  # after every 10 iterations and if the validation data is given
                approx_w = np.dot(x_val, self.w) - self.b  # calculate the dot product of validation data and w
                approx_w = np.sign(approx_w)  # get the sign of the dot product
                res_w = np.where(approx_w < 0, 0, approx_w)  # take the maximum of 0 and the dot product

                approx_w_best = np.dot(x_val, w_best) - b_best  # calculate the dot product of validation data and
                # w_best
                approx_w_best = np.sign(approx_w_best)  # get the sign of the dot product
                res_w_best = np.where(approx_w_best < 0, 0, approx_w_best)  # take the maximum of 0 and the dot product

                if accuracy_score(y_val, res_w_best) < accuracy_score(y_val, res_w):  # compare accuracy between
                    # res_w_best and res_w If the accuracy of res_w is greater than res_w_best, update w_best and
                    # b_best with the current values of self.w and self.b
                    w_best = copy.deepcopy(self.w)
                    b_best = copy.deepcopy(self.b)
                else:
                    # If the accuracy of res_w_best is greater than res_w, update self.w and self.b with w_best and
                    # b_best, respectively
                    self.w = copy.deepcopy(w_best)
                    self.b = copy.deepcopy(b_best)
                    break

    def fit(self):
        self.GradientUpdate(self.x_train, self.y_train)
        return self.w, self.b

    def predict(self):
        # Calculate the dot product between the test data and the weights, and subtract the bias
        approx = np.dot(self.x_test, self.w) - self.b
        # Apply the sign function to get the predictions
        approx = np.sign(approx)
        # Replace the negative values with 0
        return np.where(approx < 0, 0, approx)

    # Calculate the accuracy of the model.
    def accuracy(self):
        # Calculate the accuracy using the accuracy_score function
        return accuracy_score(self.y_test, self.predict()) * 100

        # This method assess the model by calculating the confusion matrix and printing different performance metrics
        # it can be called at any time, but we just use if after the final iteration in training to assess final model
    def assessModel(self, sns=None):

        print("\tModel Performance Metrics:\n\t===========================================================")

        # Model Accuracy: how often is the classifier correct?
        print("\t\t|-| Accuracy: ", round((accuracy_score(self.y_test, self.predict())) * 100, 2), "%")

        # Model Precision: what percentage of positive tuples are labeled as such?
        print("\t\t|-| Precision: ", round((precision_score(self.y_test, self.predict())) * 100, 2), "%")

        # Model Recall: what percentage of positive tuples are labelled as such?
        print("\t\t|-| Recall: ", round((recall_score(self.y_test, self.predict())) * 100, 2), "%")

        # Model F1 Score: F1 Score might be a better measure to use 
        # if we need to seek a balance between Precision and Recall
        print("\t\t|-| F1 Score: ", round((f1_score(self.y_test, self.predict())) * 100, 2), "%")

        # Model Specificity: a model's ability to predict true negatives of each available category
        print("\t\t|-| Specificity: ",
              round((recall_score(self.y_test, self.predict(), pos_label=0)) * 100, 2), "%")

        # Model Negative Predictive Value (NPV): 
        print("\t\t|-| Negative Predictive Value (NPV): ",
              round((precision_score(self.y_test, self.predict(), pos_label=0)) * 100, 2), "%")

        cm = confusion_matrix(self.y_test, self.predict())

        # calculate TP, TN, FP and FN
        TP = cm[0, 0]
        TN = cm[1, 1]
        FP = cm[0, 1]
        FN = cm[1, 0]

        # print classification error
        classification_error = (FP + FN) / float(TP + TN + FP + FN)
        print("\n\t\t|-| Classification error : {0:0.4f}".format(classification_error))

        print("\n\t\t|-| Classification Report:\n\t\t===========================================================")
        print("\t\t", classification_report(self.y_test, self.predict()))
        print("\t\t===========================================================\n")
