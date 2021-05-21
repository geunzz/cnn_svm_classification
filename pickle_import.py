import pickle
import gzip
import numpy as np
import random

def func_import(test_data_num = 300):
    X_train_bef = []
    X_test_bef = []
    fire_image_total = []
    none_image_total = []
    fire_total = []
    none_total = []
    Y_train = []
    X_train = []
    X_test = []
    Y_test = []
    Y_train_image = []
    X_train_image = []
    Y_test_image = []
    X_test_image = []

    with gzip.open('dataset_fire.pickle','rb') as f:
        dataset_fire = pickle.load(f)

    with gzip.open('dataset_none.pickle','rb') as f:
        dataset_none = pickle.load(f)

    for key in dataset_fire:
        fire_image_total.append(dataset_fire[key])
        fire_total.append([dataset_fire[key][0].reshape(-1), dataset_fire[key][1]])

    for key in dataset_none:
        none_image_total.append(dataset_none[key])
        none_total.append([dataset_none[key][0].reshape(-1), dataset_none[key][1]])

    random.shuffle(fire_total)
    random.shuffle(none_total)
    random.shuffle(fire_image_total)
    random.shuffle(none_image_total)

    X_train_fire = fire_total[int(test_data_num/2):]
    X_test_fire = fire_total[:int(test_data_num/2)]
    X_train_none = none_total[int(test_data_num/2):]
    X_test_none = none_total[:int(test_data_num/2)]

    X_train_image_fire = fire_image_total[int(test_data_num/2):]
    X_test_image_fire = fire_image_total[:int(test_data_num/2)]
    X_train_image_none = none_image_total[int(test_data_num/2):]
    X_test_image_none = none_image_total[:int(test_data_num/2)]

    X_train_bef = X_train_fire + X_train_none
    X_test_bef = X_test_fire + X_test_none
    X_train_image_bef = X_train_image_fire + X_train_image_none
    X_test_image_bef = X_test_image_fire + X_test_image_none

    random.shuffle(X_train_bef)
    random.shuffle(X_test_bef)
    random.shuffle(X_train_image_bef)
    random.shuffle(X_test_image_bef)

    for i in range(len(X_train_bef)):
        Y_train.append(X_train_bef[i][1]) #label
        X_train.append(X_train_bef[i][0]) #one dimension vector : (7500,)
        Y_train_image.append(X_train_image_bef[i][1]) #label
        X_train_image.append(X_train_image_bef[i][0]) #3 channel : (50, 50, 3)

    for i in range(len(X_test_bef)):
        Y_test.append(X_test_bef[i][1]) #label
        X_test.append(X_test_bef[i][0]) #one dimension vector : (7500,)
        Y_test_image.append(X_test_image_bef[i][1]) #label
        X_test_image.append(X_test_image_bef[i][0]) #3 channel : (50, 50, 3)

    #normalization
    X_train = np.array(X_train)
    X_train = X_train.astype('float32')
    X_train /= 255

    X_train_image = np.array(X_train_image)
    X_train_image = X_train_image.astype('float32')
    X_train_image /= 255

    Y_train_image = np.array(Y_train_image)
    Y_train_image = Y_train_image.astype('float32')

    X_test = np.array(X_test)
    X_test = X_test.astype('float32')
    X_test /= 255

    Y_test_image = np.array(Y_test_image)
    Y_test_image = Y_test_image.astype('float32')

    X_test_image = np.array(X_test_image)
    X_test_image = X_test_image.astype('float32')
    X_test_image /= 255

    print('Dataset imported.')
    print('X_train_shape :', np.shape(X_train))
    print('X_test_shape :', np.shape(X_test))
    print('Y_train_shape :', np.shape(Y_train))
    print('Y_test_shape :', np.shape(Y_test))
    print('X_train_image_shape :', np.shape(X_train_image))
    print('X_test_image_shape :', np.shape(X_test_image))
    print('Y_train_image_shape :', np.shape(Y_train_image))
    print('Y_test_image_shape :', np.shape(Y_test_image))

    return X_train, X_test, Y_train, Y_test, X_train_image, X_test_image, Y_train_image, Y_test_image

