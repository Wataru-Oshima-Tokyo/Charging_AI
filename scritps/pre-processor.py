from PIL import Image
import numpy as np
import os,glob

#class labels
labels = ["outlet"]


# directory
dataset_dir = "../models/dataset.npy" # pre-processed data
model_dir   = "../models/cnn_h5"      # learned model
# resizeing setting
resize_settings = (50,50)

# picture data
X_train = [] # learning
y_train = [] # larning label
X_test  = [] # test
y_test  = [] # testing label

for class_num, label in enumerate(labels):
    
    # the directory of picutres
    photos_dir = "../models/outlet/" + label
    
    # get the picture data
    files = glob.glob(photos_dir + "/*.jpg")
    
    #get a picture in order  
    for i,file in enumerate(files):
        
        # read one of them
        image = Image.open(file)
        
        # convert it to rgb
        image = image.convert("RGB")
        
        # resize the image
        image = image.resize(resize_settings)
        
        # transform it to numeric array
        data  = np.asarray(image) 

        # add a test data
        if i%4 ==0:
            X_test.append(data)
            y_test.append(class_num)
            
        # create differet angle of picture data
        else:           
            # add 4 degree rotated data from -20 to 20 degree 
            for angle in range(-25,20,5):
                # rotate it
                img_r = image.rotate(angle)
                #image to nemric array
                data  = np.asarray(img_r)
                # addition
                X_train.append(data)
                y_train.append(class_num)
                # flip the image
                img_trans = image.transpose(Image.FLIP_LEFT_RIGHT)
                data      = np.asarray(img_trans)
                X_train.append(data)
                y_train.append(class_num)        
        
        
# transfom it to numpy array so that tensorflow can easily handle it
X_train = np.array(X_train)
X_test  = np.array(X_test)
y_train = np.array(y_train)
y_test  = np.array(y_test)


# save the pre-processed data
dataset = (X_train,X_test,y_train,y_test)
np.save(dataset_dir,dataset)
