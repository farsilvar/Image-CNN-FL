import os

import flwr as fl
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#CUDA_VISIBLE_DEVICES=""
from model import makeModel
from layers import createCNNLayers
from ypstruct import structure
import PIL
from PIL import Image
import numpy as np

from tensorflow import keras
from tensorflow.keras.utils import img_to_array, load_img
from keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import sys
import getDataset

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

datasetDir = sys.argv[2] + '/'
imageIndex = int(sys.argv[3])

print(datasetDir)
print(imageIndex)

cid = int(sys.argv[1])
datasetLoader = getDataset.GetDataset(datasetDir, imageIndex)
trainloaders = datasetLoader.getTrainLoaders()
testloaders = datasetLoader.getTestLoaders()

numEpochs = 50
VERBOSE = 0

convLayers, denseLayers = createCNNLayers(int(trainloaders[2][cid]))
learningRateList = [1e-8, 0.0001, 0.001, 0.01, 0.1]

initialModel = Sequential([layers.Rescaling(scale = 1./255, input_shape=trainloaders[1])])
cnnModel = makeModel(initialModel, convLayers, denseLayers, learningRateList[2])

cnnModel.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
model = cnnModel

# Define Flower client
class Client(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(trainloaders[0][cid], epochs=numEpochs, batch_size=32, verbose=VERBOSE)
        return model.get_weights(), trainloaders[3], {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(testloaders[0][cid])
        return loss, testloaders[5], {"accuracy": accuracy, "Client Id": cid}

# Start Flower client
fl.client.start_client(server_address="127.0.0.1:8080", client=Client().to_client())
