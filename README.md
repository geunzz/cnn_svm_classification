# cnn_svm_Classification
[code description]
data_mover.py : Randomly transfers data to a specified path as many times as a given number.
data_generation.py : Load image data directly from a folder. 
(Image files must be separated into folders for each class, and the upper folder must be designated.)
data_to_pickle.py : Converts the image file in the path to a pickle file.
pickle_import.py : Import pickle file and load 3-channel image data and labels in numpy array format and also
                In addition, training data and test data are randomly distributed and returned to a pickle file.
CNN_train_test.py : Train CNN model and measure accuracy with test data.
CNN_SVM_train_test.py : cnn is the same, but classifier is used as SVM.
