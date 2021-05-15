import os
import cv2
import numpy as np
import random

#DATASET_PATH = 'C:/projects/dataset/thermal_image/training_dataset/'

class contrastive_data():

    def __init__(self, DATASET_PATH, shuffle_sel):
        self.dataset_path = DATASET_PATH
        self.data_class_set=[]
        self.class_id_matching = []
        self.id_val = 0
        self.X = []
        self.Y = []
        self.Z = []
        self.shuffle_sel = shuffle_sel

    def data_class_set_gen(self, shuffle_sel=True):
        for class_name in os.listdir(self.dataset_path):
            self.class_id_matching = [class_name, self.id_val]
            class_path = self.dataset_path + class_name
            for data in os.listdir(class_path):
                data_path = class_path +  '/' + data
                image = cv2.imread(data_path)
                data_class = [image, data, self.id_val]
                self.data_class_set.append(data_class)

            self.id_val = self.id_val + 1
        if self.shuffle_sel == shuffle_sel:    
            random.shuffle(self.data_class_set)
        return self.data_class_set

    def train_val_split(self, test_prob=0.2):

        for i in range(0, len(self.data_class_set)):
            self.X.append(self.data_class_set[i][0])
            self.Y.append(self.data_class_set[i][2])
            self.Z.append(self.data_class_set[i][1])

        if len(self.X) == len(self.Y):
            x_train = self.X[0:int(len(self.X)*(1 - test_prob))]
            x_test = self.X[int(len(self.X)*(1 - test_prob)):]
            
            y_train = self.Y[0:int(len(self.Y)*(1 - test_prob))]
            y_test = self.Y[int(len(self.X)*(1 - test_prob)):]

            z_train = self.Z[0:int(len(self.Y)*(1 - test_prob))]
            z_test = self.Z[int(len(self.X)*(1 - test_prob)):]

        else:
            raise('Check the length match of dataset matrix with class index matrix.')

        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        z_train = np.array(z_train)
        z_test = np.array(z_test)
        
        return x_train, x_test, y_train, y_test, z_train, z_test


# datagen = contrastive_data(DATASET_PATH = 'C:/projects/dataset/thermal_image/training_dataset/', shuffle_sel=True)
# data_class_set = datagen.data_class_set_gen()
# x_train, x_test, y_train, y_test, z_train, z_test = datagen.train_val_split(0.2)









