import os
from ypstruct import structure
import tensorflow as tf
from tensorflow.keras.utils import img_to_array, load_img
import sys

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32

ColorMode = structure()

#ColorMode.name = 'rgb'
#ColorMode.shape = 3

ColorMode.name = 'grayscale'
ColorMode.shape = 1

class GetDataset():

    def __init__(self, datasetDir, listIndex):
         
        self.path_to_dataSets = '/home/felipe/felipe/MEGA/doutorado/modelo/UCR/'
        self.datasetDir = datasetDir
        self.listIndex = listIndex

    def getTrainLoaders(self):

        trainDirList =  ['trainDataCppRAW_FL', 'trainDataCppGridPYFTS_FL', 'trainDataPolar_FL','trainDataPolarNew_FL']

        trainDir = trainDirList[self.listIndex]

        TRAIN_DATA_DIR = self.path_to_dataSets + self.datasetDir + trainDir

        clients = []

        clientsRootFolder = TRAIN_DATA_DIR

        totalTrainFiles = 0
        totalTrainClasses = 0

        for base, dirs, files in os.walk(clientsRootFolder):
            for Dirs in dirs:
                if Dirs.find('client') != -1:
                    clients.append(Dirs)
                    totalTrainClasses += 1

        print(clients)

        trainloaders = []
        num_classes = []
    
        for client in clients:
            TRAIN_FOLDER = clientsRootFolder + '/' + client + '/'
            for base, dirs, files in os.walk(TRAIN_FOLDER):
                for Files in files:
                    totalTrainFiles += 1
                for Dirs in dirs:
                    totalTrainClasses += 1
    
            testFile = base + '/' + files[0]
            img = load_img(testFile,
                    color_mode = ColorMode.name)

            img_height = min(256,img_to_array(img).shape[1])
            img_width = min(256,img_to_array(img).shape[0]) 

            train_generator = tf.keras.utils.image_dataset_from_directory(
                TRAIN_FOLDER,
                seed=None,
                color_mode=ColorMode.name,
                image_size=(img_height, img_width),
                batch_size=batch_size)
    
            for image_batch, labels_batch in train_generator:
                img_size = image_batch.shape[1:]
                break

            class_names = train_generator.class_names
            num_classes.append(len(class_names))
    
            train_ds = train_generator.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

            trainloaders.append(train_ds)

        return trainloaders, img_size, num_classes, totalTrainFiles

    def getTestLoaders(self):

        testDirList =  ['testDataCppRAW_FL', 'testDataCppGridPYFTS_FL',
                        'testDataPolar_FL','testDataPolarNew_FL']

        testDir = testDirList[self.listIndex]

        TEST_DATA_DIR = self.path_to_dataSets + self.datasetDir + testDir

        clients = []

        clientsRootFolder = TEST_DATA_DIR

        totalTestFiles = 0
        totalTestClasses = 0

        for base, dirs, files in os.walk(clientsRootFolder):
            for Dirs in dirs:
                if Dirs.find('client') != -1:
                    clients.append(Dirs)
                    totalTestClasses += 1

            testloaders = []
            num_classes_dist = []
    
        for client in clients:
            TEST_FOLDER = clientsRootFolder + '/' + client + '/'
            for base, dirs, files in os.walk(TEST_FOLDER):
                for Files in files:
                    totalTestFiles += 1
                for Dirs in dirs:
                    totalTestClasses += 1

            testFile = base + '/' + files[0]
            img = load_img(testFile,
                    color_mode = ColorMode.name)

            img_height = min(256,img_to_array(img).shape[1])
            img_width = min(256,img_to_array(img).shape[0])

            test_generator = tf.keras.utils.image_dataset_from_directory(
                TEST_FOLDER,
                seed=None,
                color_mode=ColorMode.name,
                image_size=(img_height, img_width),
                batch_size=batch_size)

            for image_batch, labels_batch in test_generator:
                img_size = image_batch.shape[1:]
                break

            class_names = test_generator.class_names
            num_classes_dist.append(len(class_names))

            test_ds = test_generator.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

            testloaders.append(test_ds)

        totalTestFiles = 0
        totalTestClasses = 0

        testDirList =  ['testDataCppRAW', 'testDataCppGridPYFTS',
                        'testDataPolar','testDataPolarNew']


        testDir = testDirList[self.listIndex]

        TEST_DATA_DIR = self.path_to_dataSets + self.datasetDir + testDir
    
        TEST_FOLDER = TEST_DATA_DIR

        for base, dirs, files in os.walk(TEST_FOLDER):
            for Files in files:
                totalTestFiles += 1
            for Dirs in dirs:
                totalTestClasses += 1

        testFile = base + '/' + files[0]
        img = load_img(testFile,
                    color_mode = ColorMode.name)

        img_height = min(256,img_to_array(img).shape[1])
        img_width = min(256,img_to_array(img).shape[0])

        test_generator = tf.keras.utils.image_dataset_from_directory(
            TEST_DATA_DIR,
            seed=None,
            color_mode=ColorMode.name,
            image_size=(img_height, img_width),
            batch_size=batch_size)

        class_names = test_generator.class_names
        num_classes_cent = len(class_names)

        for image_batch, labels_batch in test_generator:
            img_size = image_batch.shape[1:]
            break

        test_ds = test_generator.cache().prefetch(buffer_size=AUTOTUNE)

        return testloaders, test_ds, img_size, num_classes_dist, num_classes_cent, totalTestFiles
