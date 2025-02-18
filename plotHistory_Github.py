import matplotlib.pyplot as plt
import pandas as pd

def saveHistory(historyMetric, metricType, datasetInfo):
    trainDirList =  ['trainDataCppRAW_FL', 'trainDataCppGridPYFTS_FL', 'trainDataPolar_FL', 'trainDataPolarNew_FL']
    round = [data[0] for data in historyMetric["accuracy"]]
    acc = [100.0 * data[1] for data in historyMetric['accuracy']]
    histDict = {}
    histDict['round'] = round
    histDict['acc'] = acc
    if metricType == 'Distributed':
        clientId = [data[1] for data in historyMetric['Client Id']]
        clientAccuracy = [data[1] for data in historyMetric['clientAccuracy']]
        examples = [data[1] for data in historyMetric['Num Examples']]
        histDict['Client Id'] = clientId
        histDict['Client Accuracy'] = clientAccuracy
        histDict['Num Examples'] = examples
    plt.scatter(round, acc)
    plt.grid()
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Round")
    plt.title(datasetInfo.datasetDir[:-1] + " 3 clients with 3 clients per round")
    histDF = pd.DataFrame(histDict)
    histDF.to_csv(datasetInfo.path_to_dataSets + datasetInfo.datasetDir + 'AccHistory' + metricType + '_' + trainDirList[datasetInfo.listIndex] + '2R.csv', index = False)

def plot(history, datasetInfo):
    global_accuracy_centralised = history.metrics_centralized
    saveHistory(global_accuracy_centralised, 'Centralized', datasetInfo)
    global_accuracy_distributed = history.metrics_distributed
    saveHistory(global_accuracy_distributed, 'Distributed', datasetInfo)
