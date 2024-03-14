import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score


class SVM_RBF:
    def __init__(self, x_train, y_train, x_test, y_test, learning_rate=0.01, lambda_param=0.1,
                 iterations=10000, gamma=1):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.iterations = iterations
        self.gamma = gamma
        self.w = np.array([])  # weight
        self.b = None

    def fit(self):
        n_samples, n_features = self.x_train.shape
        y_ = np.where(self.y_train <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for i in range(self.iterations):
            for idx, x_i in enumerate(self.x_train):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx] * self.gamma))

                    self.b -= self.lr * y_[idx]

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
        print("\t\t|-|> Accuracy: ", round((accuracy_score(self.y_test, self.predict())) * 100, 2), "%")

        # Model Precision: what percentage of positive tuples are labeled as such?
        print("\t\t|-|> Precision: ", round((precision_score(self.y_test, self.predict())) * 100, 2), "%")

        # Model Recall: what percentage of positive tuples are labelled as such?
        print("\t\t|-|> Recall: ", round((recall_score(self.y_test, self.predict())) * 100, 2), "%")

        # Model F1 Score: F1 Score might be a better measure to use
        # if we need to seek a balance between Precision and Recall
        print("\t\t|-|> F1 Score: ", round((f1_score(self.y_test, self.predict())) * 100, 2), "%")

        # Model Specificity: a model's ability to predict true negatives of each available category
        print("\t\t|-|> Specificity: ",
              round((recall_score(self.y_test, self.predict(), pos_label=0)) * 100, 2), "%")

        # Model Negative Predictive Value (NPV):
        print("\t\t|-|> Negative Predictive Value (NPV): ",
              round((precision_score(self.y_test, self.predict(), pos_label=0)) * 100, 2), "%")

        cm = confusion_matrix(self.y_test, self.predict())

        # calculate TP, TN, FP and FN
        TP = cm[0, 0]
        TN = cm[1, 1]
        FP = cm[0, 1]
        FN = cm[1, 0]

        # print classification error
        classification_error = (FP + FN) / float(TP + TN + FP + FN)
        print("\n\t\t|-|> Classification error : {0:0.4f}".format(classification_error))

        print("\n\t\t|-|> Classification Report:\n\t\t===========================================================")
        print("\t\t", classification_report(self.y_test, self.predict()))
        print("\t\t===========================================================")
