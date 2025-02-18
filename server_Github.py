import os
import sys

from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

import flwr as fl
from flwr.common import Metrics

import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#CUDA_VISIBLE_DEVICES=""
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

import getDataset
from model import makeModel
from layers import createCNNLayers
import plotHistory

VERBOSE = 0
NUM_CLIENTS = 3
numRounds = 2

datasetDir = sys.argv[1] + '/'
imageIndex = int(sys.argv[2])

datasetLoader = getDataset.GetDataset(datasetDir, imageIndex)
testLoaders = datasetLoader.getTestLoaders()

testSet = testLoaders[1]

def get_evaluate_fn(testset: testSet):
    """Return an evaluation function for server-side (i.e. centralised) evaluation."""

    # The `evaluate` function will be called after every round by the strategy
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ):

        convLayers, denseLayers = createCNNLayers(testLoaders[4])
        learningRateList = [1e-8, 0.0001, 0.001, 0.01, 0.1]

        initialModel = Sequential([layers.Rescaling(scale = 1./255, input_shape=testLoaders[2])])
        model = makeModel(initialModel, convLayers, denseLayers, learningRateList[2])
        model.compile(optimizer='adam', #Adam(learning_rate=learningRate),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
                  
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(testset, verbose=VERBOSE)
        return loss, {"accuracy": accuracy}

    return evaluate

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    clientId = [m["Client Id"] for _, m in metrics]
    accuracyEachClient = [m["accuracy"] for _, m in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples), 
            "clientAccuracy": accuracyEachClient, 
            "Client Id": clientId,
            "Num Examples":examples}

client_resources = {"num_cpus": 1, "num_gpus": 0.0}

# Define strategy
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,
    evaluate_fn=get_evaluate_fn(testSet),  # global evaluation function
    )

# Start Flower server
history = fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=numRounds),
    strategy=strategy,
)

print(history)

print(f"{history.metrics_centralized = }")

plotHistory.plot(history, datasetLoader)
