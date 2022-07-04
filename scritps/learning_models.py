import tensorflow
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.utils  import np_utils
import numpy as np

# class label
labels = ["outlet"]
# directory
dataset_dir = "../models/dataset.npy" # pre-processed data
model_dir   = "../models/cnn_h5"      # learned models
# resizing setting
resize_settings = (50,50)

# main funciton
def main():
    """
    1. pre-processing data (encoding)
    """
    #read the numpy data preserved before
    X_train,X_test,y_train,y_test = np.load(dataset_dir, allow_pickle=True)
    
    #normalize it so that the number of range can be put in btw 0 and 1 becasue this is an integer range
    X_train = X_train.astype("float") / X_train.max()
    X_test  = X_test.astype("float") /  X_train.max()
    
    # adopt the one-hot method (if it's coorect, then shows 1; otherwise 0)
    y_train = np_utils.to_categorical(y_train,len(labels))
    y_test  = np_utils.to_categorical(y_test,len(labels))
    """
    2. learning models and evaluate them
    """
    #learning models
    model = model_train(X_train,y_train)
    
    #evaluateing them
    evaluate(model,X_test, y_test)
    
  
   
# the funcfion of learning models
def model_train(X_train,y_train):
    
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
                  
    #learning models
    model.fit(X_train,y_train,batch_size=10,epochs=150)
    # save the result of models
    model.save(model_dir)
    return model
    
# evaluation function
def evaluate(model,X_test,y_test):
    # evaluate the models
    scores = model.evaluate(X_test,y_test,verbose=1)
    print("Test Loss: ", scores[0])
    print("test Accuracy: ", scores[1])

model = main()