import tensorflow
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.utils  import np_utils
import numpy as np

# class label
labels = ["outlet"]
# directory
dataset_dir = "models/dataset.npy" # pre-processed data
model_dir   = "models/cnn_h5"      # learned models
# resizing setting
resize_settings = (50,50)

# main funciton
def main(path):
    X     = []                               # 推論データ格納
    image = Image.open(path)                 # 画像読み込み
    image = image.convert("RGB")             # RGB変換
    image = image.resize(resize_settings)    # リサイズ
    data  = np.asarray(image)                # 数値の配列変換
    X.append(data)
    X     = np.array(X)
    
    # モデル呼び出し
    model = predict()
    
    # numpy形式のデータXを与えて予測値を得る
    model_output = model.predict([X])[0]
    # 推定値 argmax()を指定しmodel_outputの配列にある推定値が一番高いインデックスを渡す
    predicted = model_output.argmax()
    # アウトプット正答率
    accuracy = int(model_output[predicted] *100)
    print("{0} ({1} %)".format(labels[predicted],accuracy))
    
  
   
# the funcfion of learning models
def predict():
    
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
    model.add(Dense(3)) 
    model.add(Activation('softmax'))
    # optimized algorithm
    opt = tensorflow.keras.optimizers.RMSprop(lr=0.005, decay=1e-6)
    # loss function
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"]
                 )
                  
    model = keras.models.load_model("data/cnn_h5")
    
    return model
    
# evaluation function
def evaluate(model,X_test,y_test):
    # evaluate the models
    scores = model.evaluate(X_test,y_test,verbose=1)
    print("Test Loss: ", scores[0])
    print("test Accuracy: ", scores[1])
