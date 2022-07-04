import tensorflow
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.utils  import np_utils
import sys
import numpy as np
from PIL import Image
from tensorflow import keras

# class label
labels = ["outlet"]
# directory
dataset_dir = "../models/dataset.npy" # pre-processed data
model_dir   = "../models/cnn_h5"      # learned models
# resizing setting
resize_settings = (50,50)


  
   
# the funcfion of learning models
def predict(X_train):
    
    #instantiate the class
    model = Sequential()
    
    # the first layer (convoluted)
    model.add(Conv2D(32,(3,3),padding="same", input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    # the second layer (Max Pooling)
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    # the third layer (Max Pooling)
    model.add(MaxPooling2D(pool_size=(2,2)))                     
    model.add(Dropout(0.3))                     
    # the fourth layer(convoluting) 
    model.add(Conv2D(64,(3,3),padding="same"))                   
    model.add(Activation('relu'))
    # the fifth layer (convoluting)
    model.add(Conv2D(64,(3,3))) 
    model.add(Activation('relu'))
    # the sixth layer (Max pooling)
    model.add(MaxPooling2D(pool_size=(2,2)))
    # make the data one dimenstion
    model.add(Flatten())
    # the seventh layer (make it altogether)
    model.add(Dense(512))                                       
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # the output layer ( return 0 or 1 by softmax)
    model.add(Dense(1)) 
    model.add(Activation('softmax'))
    # optimized algorithm
    opt = tensorflow.keras.optimizers.RMSprop(lr=0.005, decay=1e-6)
    # loss function
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"]
                 )
                  
    model = keras.models.load_model("../models/cnn_h5")
    
    return model
    
# main funciton
def main(path):
    X_train,X_test,y_train,y_test = np.load(dataset_dir, allow_pickle=True)
    X_train = X_train.astype("float") / X_train.max()
    X     = []                               # store the predcited model
    image = Image.open(path)                 # read an image
    image = image.convert("RGB")             # RGB transformation
    image = image.resize(resize_settings)    # resizing
    data  = np.asarray(image)                # numeric array conversion
    X.append(data)
    X     = np.array(X)
    
    # call the prediction function
    model = predict(X_train)
    
    # get the predicted value by giving a numpy formatted data x
    model_output = model.predict([X])[0]
    # return the heighst index of predicted value in the array of model_output by setting the argmax()
    predicted = model_output.argmax()
    # the rate of correction
    accuracy = int(model_output[predicted] *100)
    print("{0} ({1} %)".format(labels[predicted],accuracy))

#prediction
path = "../models/outlet/outlet_000.jpg"
image = Image.open(path)

main(path)
    
