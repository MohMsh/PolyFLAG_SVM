import random
import pandas as pd
from sklearn import datasets as data
import server

import aes
import warnings

warnings.filterwarnings("ignore")

host = 'localhost'
port = 61297
backlog_queue = 5
tableSize = 10  # Keys table size
minimum_clients = 3
timeout = 10
buffer_size = 4096
aggregation_method = "average_aggregation"  # "average_aggregation" # "clipped_avg_aggregator"
# "dp_avg_aggregator" "momentum_aggregation"

"""
# SHAREEDB Dataset
# read the validation dataset to use in global model assessment
main_data_frame = pd.read_csv("ExtractedFeaturesValidation.csv", header=None)
# Splitting Data Between Features & Results
x_test = main_data_frame.iloc[:, 0:26].values
y_test = main_data_frame.iloc[:, 26:27].values
"""

"""
# read from the generated simulation
main_data_frame = pd.read_csv("simulated_dataset.csv")

# Splitting Data Between Features & Results
x = main_data_frame.iloc[:, :-1].values
y = main_data_frame.iloc[:, -1].values

# select only 1/9 of the data instances to test
n = int(len(x) // 150)

# Get the indices of n random items
indices = random.sample(range(len(x)), n)

# Select the corresponding items from x and y
x_test = x[indices]
y_test = y[indices]
"""


# Surgical Deepnet
main_data_frame = pd.read_csv("Surgical-deepnet.csv")
# Splitting Data Between Features & Results
x = main_data_frame.iloc[:, :-1].values
y = main_data_frame.iloc[:, -1].values
# select only 1/9 of the data instances to test
n = int(len(x) // 80)
# Get the indices of n random items
indices = random.sample(range(len(x)), n)
# Select the corresponding items from x and y
x_test = x[indices]
y_test = y[indices]

connection = server.server(host=host, port=port, table_size=tableSize, backlog_queue=backlog_queue,
                           minimum_clients=minimum_clients, timeout=timeout, buffer_size=buffer_size,
                           print_incoming_messages=True, print_sent_messages=True,
                           aggregation_method=aggregation_method, x_test=x_test, y_test=y_test, global_rounds=3)
connection.start()
