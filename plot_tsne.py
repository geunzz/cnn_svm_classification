#plot TSNE - imbalanced data
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model, load_model
import pickle
import gzip
import numpy as np

X_test_image = []
Y_test_image = []

with gzip.open('test_img.pickle','rb') as f:
    test_img = pickle.load(f)

for key in test_img:
    X_test_image.append(test_img[key][0])
    Y_test_image.append(test_img[key][1])

Y_test_image = np.array(Y_test_image)
Y_test_image = Y_test_image.astype('float32')

X_test_image = np.array(X_test_image)
X_test_image = X_test_image.astype('float32')
X_test_image /= 255

model = load_model('model.10.h5')
model2 = Model(inputs=model.input, outputs=model.layers[-2].output)
model2.summary()

features_X_test = model2.predict(X_test_image)

tsne = TSNE(n_components=2, random_state=0)
# Project the data in 2D
X_2d = tsne.fit_transform(features_X_test)
# Visualize the data
target_ids = range(len(Y_test_image))

plt.figure(figsize=(5, 5))
colors =  'r', 'b'
labels =  'FIRE', 'None'
for i, c, label in zip(target_ids, colors, labels):
    plt.scatter(X_2d[Y_test_image == i, 0], X_2d[Y_test_image == i, 1], c=c, label=label, s=2 , alpha=0.3)

plt.legend()
plt.show()

