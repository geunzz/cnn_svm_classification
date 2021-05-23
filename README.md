# cnn_svm_classification  

This code solves the binary classification problem that classifies images with and without fire.
It includes a code that combines cnn and binary classifier, 
which is used to solve a general image classification problem, and a code that uses cnn+svm as a classifier.
Codes for handling image data so that images can be put into cnn and code for loading images saved in the form of pickle files are also included.  
  
In general, even when only svm is used as a classifier, the classification performance is quite excellent, 
but the performance is not superior to the classification problem using a neural network. 
Therefore, in recent years, a deep learning method using a neural network has been in the spotlight among machine learning methods.
In the image classification problem, a neural network called cnn is mainly used, 
and a binary classifer is placed in the last layer of cnn and the classification is performed by activating it as a sigmoid.  
In addition to the general form, the last layer was partially modified to extract the feature vector from cnn, 
and this vector was implemented as a code so that it can be classified as svm. 
That is, only the classifier part is replaced with svm, and it shows slightly better performance.  


code description
------------------------------
The data_mover.py file randomly selects and moves the number of images specified by the user to another path to which the user wants to move.
In the path where the file is located, there must be a folder classified by class and an image of each class in it.
For example, assume that there are folders named'class1,'class2' in the path named'path', and there are 100 images in each.
If 50 is given in the data_mover.py file as the number of images to be moved, each 25 images in the class1 folder and class2 folder are moved to the desired path.

    data_mover.py  
    
The data_generation.py file must also have a folder structure divided by class in the specified path like data_mover.py.
Executing the data_generation.py file loads the image data, and saves the class value of the image in the form of a python list together.  

    data_generation.py  
    
An example of execution of the data_generation.py file is as the following code, 
and several options can be set as follows to execute the data_generator class in the file.
If 'shuffle_sel=True' is set, it is an option to shuffle the order while loading images, 
and test_prob refers to the ratio of test data. If'test_prob=0' is set, an empty list of x_test, y_test, and z_test appears.
Here, x_train and x_test store the image array values, y_train and y_test store the corresponding label values, and z_train and z_test store the image names.

    datagen = data_generator(DATASET_PATH = 'PATH/TO/THE/IMAGE DATA', shuffle_sel=True)
    data_class_set, data_array, label, data_name = datagen.data_label_set_gen()
    x_train, x_test, y_train, y_test, z_train, z_test = datagen.train_val_split(data_array, label, data_name, test_prob=0.2)


data_to_pickle.py is a file that saves image data in the form of pickle. 
The image is loaded using the openCV library, and the image is saved in a state that maintains the three-channel format as it is.  

    data_to_pickle.py
    
Contrary to the code above, this is the code that loads the image saved as a pickle file. 
While loading the pickle file, it randomly distributes training data and test data. 
At this time, the number to be divided is adjusted to the value given by the user.  

    pickle_import.py  
    
The CNN_train_test.py file is a code that loads an image using pickle_import.py, trains it, 
and tests the result model to measure the accuracy. 
Both training and testing are performed with a single execution of the code. 
Because the pickle_import.py file is used, training and test data are different for each execution.

    CNN_train_test.py  

The CNN_SVM_train_test.py file is all the same as CNN_train_test.py, except that svm is used as a classifier.  

    CNN_SVM_train_test.py  
    
The plot_tsne.py file is a code that visualizes the feature vector generated from the trained model by reducing it to 2D. 
TSNE is used as a visualization method, and the overall data structure can be checked by reducing the high-dimensional data to the low-dimensional.
The code is configured to draw the feature vecrtor obtained by passing test data on the trained model.  

    plot_tsne.py

    
    
    
    
    
