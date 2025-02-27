from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import tensorflow as tf
from keras.optimizers import Adam

def makeModel(model, convIn, denseIn, learningRate):

    for i in range(len(convIn)):
        model.add(layers.Conv2D(convIn[i].filters, kernel_size = convIn[i].kernels, strides = convIn[i].strides, padding = convIn[i].padding, activation = convIn[i].activation))
        if (convIn[i].pooling):
          if convIn[i].poolingType == 'max':
            model.add(layers.MaxPooling2D(strides = convIn[i].poolingStride))
          elif convIn[i].poolingType == 'avg':
            model.add(layers.AveragePooling2D(strides = convIn[i].poolingStride))
        else:
          pass

    model.add(layers.Flatten())
    for i in range(len(denseIn)):
        model.add(layers.Dense(denseIn[i].units, denseIn[i].activation))
        if denseIn[i].dropout == True:
            model.add(layers.Dropout(0.2))

    return model

def makeDictModel(model,convIn, denseIn, learningRate):

    for i in range(len(convIn)):
        model.add(layers.Conv2D(convIn[i].filters, kernel_size = convIn[i].kernels, strides = convIn[i].strides, padding = convIn[i].padding, activation = convIn[i].activation))
        if (convIn[i].pooling):
          if convIn[i].poolingType == 'max':
            model.add(layers.MaxPooling2D(strides = convIn[i].poolingStride))
          elif convIn[i].poolingType == 'avg':
            model.add(layers.AveragePooling2D(strides = convIn[i].poolingStride))
        else:
          pass

    model.add(layers.Flatten())
    for i in range(len(denseIn)):
        model.add(layers.Dense(denseIn[i].units, denseIn[i].activation))
        if denseIn[i].dropout == True:
            model.add(layers.Dropout(0.2))

    model.compile(optimizer=Adam(learning_rate=learningRate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model
