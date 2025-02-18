def createCNNLayers(num_classes, modelDict=None):

    class ConvLayer():
        def __init__(self,filters, kernels, padding, activation, strides,
                     pooling, poolingType, poolingStride, dropout):
            self.filters = filters
            self.kernels = kernels
            self.padding = padding
            self.activation = activation
            self.strides = strides
            self.pooling = pooling
            self.poolingType = poolingType
            self.poolingStride = poolingStride
            self.dropout = dropout
    
    class DenseLayer():
        def __init__(self, units, activation, dropout):
            self.units = units
            self.activation = activation
            self.dropout = dropout

    convLayersList = []
    convLayers = []
    denseLayers = []

    convLayersList.append([16, 3, 'same', 'relu', 1, True, 'max', None, False])
    convLayersList.append([32, 3, 'same', 'relu', 1, True, 'max', None, False])
    convLayersList.append([64, 3, 'same', 'relu', 1, True, 'max', None, False])

    for i in range(len(convLayersList)):
        convLayers.append(ConvLayer(convLayersList[i][0], # filters
                                    convLayersList[i][1], # kernel size
                                    convLayersList[i][2],
                                    convLayersList[i][3],
                                    convLayersList[i][4],
                                    convLayersList[i][5],
                                    convLayersList[i][6],
                                    convLayersList[i][7],
                                    convLayersList[i][8]))
    
    denseLayers.append(DenseLayer(128, 'relu', True))

    return convLayers, denseLayers
