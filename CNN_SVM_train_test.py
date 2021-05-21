from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow.keras as keras
from tensorflow.keras.models import Model, load_model
import pickle_import
from sklearn import svm
import joblib

#randomly divide train & test data.
#test data number = 300
X_train, X_test, Y_train, Y_test, X_train_image, X_test_image, Y_train_image, Y_test_image = pickle_import.func_import(300)

#imbalanced data handling
import collections
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

# Apply SMOTE method on training data
X_train_SMOTE, Y_train_SMOTE = SMOTE(random_state=0).fit_resample(X_train, Y_train)
X_train_image_SMOTE, Y_train_image_SMOTE = SMOTE(random_state=0).fit_resample(X_train_image.reshape(X_train_image.shape[0], -1), Y_train_image)
X_train_image_SMOTE = X_train_image_SMOTE.reshape(X_train_image_SMOTE.shape[0],50,50,3)

# Apply ADASYN method on training data
X_train_ADASYN, Y_train_ADASYN = ADASYN(random_state=0).fit_resample(X_train, Y_train)
X_train_image_ADASYN, Y_train_image_ADASYN = ADASYN(random_state=0).fit_resample(X_train_image.reshape(X_train_image.shape[0], -1), Y_train_image)
X_train_image_ADASYN = X_train_image_ADASYN.reshape(X_train_image_ADASYN.shape[0],50,50,3)

#0 - None fire image, 1 - fire image
print("Origin data :",collections.Counter(Y_train))
print("After SMOTE :", collections.Counter(Y_train_SMOTE))
print("After ADASYN :", collections.Counter(Y_train_ADASYN))

#model define
model = models.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu', input_dim=None))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

my_callbacks = [
    keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}.h5', save_freq='epoch', save_weights_only=False),
]

#model train
#you can train the data appied SMOTE or ADASYN : model.fit(X_train_image_SMOTE, Y_train_image_SMOTE ~
model.fit(X_train_image_SMOTE, Y_train_image_SMOTE, batch_size=10, epochs=10, callbacks=my_callbacks)

model = load_model('model.10.h5')
model2 = Model(inputs=model.input, outputs=model.layers[-2].output)
model2.summary()

#when you train model the data applied SMOTE, you should train SVM also SMOTE applied data.
features_X_train = model2.predict(X_train_image_SMOTE)
features_X_test = model2.predict(X_test_image)

#check the best C, Gamma value through the SVM_train_test.py
C = 5
Gamma = 0.002
clf=svm.SVC(kernel="rbf", C=C, gamma = Gamma, max_iter=20000)

clf.fit(features_X_train, Y_train_image_SMOTE)
Y_pred = clf.predict(features_X_test)

pred_count = 0
for i in range(len(Y_pred)):
    if (Y_pred[i] == 0) and (Y_test_image[i] == 0):
        pred_count = pred_count + 1
    elif (Y_pred[i] == 1) and (Y_test_image[i] == 1):
        pred_count = pred_count + 1

accuracy = pred_count / len(Y_pred)
accuracy = round(accuracy, 5)
print("accuracy : ", accuracy*100, "[%]")

#save the SVM trained model
filename = 'G_' + str(Gamma) + '_C_' + str(C) + '_' + str(accuracy) + '_rbf.pkl'
joblib.dump(clf, filename)


