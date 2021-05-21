import cv2
import glob
import pickle
import gzip

X_train_fire = []
X_train_non = []

fire_train = glob.glob('PATH/TO/THE/FIRE DATASET/*.jpg')
non_train = glob.glob('PATH/TO/THE/NONE DATASET/*.jpg')

for i in range(len(fire_train)):
    image_temp = cv2.imread(fire_train[i])
    X_train_fire.append([cv2.imread(fire_train[i]), 1])

for i in range(len(non_train)):
    image_temp = cv2.imread(non_train[i])
    X_train_non.append([cv2.imread(non_train[i]), 0])

i = 0
dataset_fire = {}
for i in range(len(X_train_fire)):
    name = 'fire_' + str(i)
    dataset_fire[name] = X_train_fire[i]

with gzip.open('dataset_fire.pickle', 'wb') as f:
    pickle.dump(dataset_fire, f, pickle.HIGHEST_PROTOCOL)

i = 0
dataset_none = {}
for i in range(len(X_train_non)):
    name = 'none_' + str(i)
    dataset_none[name] = X_train_non[i]

with gzip.open('dataset_none.pickle', 'wb') as f:
    pickle.dump(dataset_none, f, pickle.HIGHEST_PROTOCOL)

    
