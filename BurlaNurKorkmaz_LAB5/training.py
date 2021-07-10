import pandas as pd
import cv2 as cv
import numpy 
import random


train_label = pd.read_csv("gridslabel.csv") 
train_label = pd.DataFrame(data=train_label)
train_label = train_label.drop(columns=['gridimage_name'])
train_label = train_label.to_numpy()

#%%
path = "GridsFolder\\" 
train_images = numpy.empty(len(train_label), dtype=object)
for i in range(len(train_label)):
    image = cv.imread(path + str(i) + ".png")
    train_images[i] =  cv.cvtColor(image, cv.COLOR_BGR2RGB)
    print("reading cropped images " + str(i))
    
train_images = train_images.tolist()

#%%
data = list(zip(train_label, train_images))
random.shuffle(data)
#%%
count_zero = 0;
count_one = 0;
for i in range(len(train_label)):
    if(train_label[i] == 0):
        count_zero += 1
    elif(train_label[i] == 1):
        count_one += 1 
#%%
counter = 0
for i in range(len(train_label)):          
    
   if(count_zero > count_one):            
        if(data[i-counter][0] == 0):
            del data[i-counter]
            counter+=1
        if(count_one*2 == len(data)):
            break
  
        
#%%
train_label, train_images = zip(*data)
train_label = numpy.array(train_label)
train_images = numpy.array(train_images)

#%%

from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam

train_label = to_categorical(train_label, num_classes = 2) # convert to one-hot-encoding

#%%

model = Sequential()
    
model.add(Conv2D(filters = 32, kernel_size = (7,7),padding = 'Same', 
                     activation ='relu', input_shape = (32,32,3)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
    
model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', 
                     activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2), strides=(3,3)))
model.add(Dropout(0.25))
    
model.add(Conv2D(filters = 128, kernel_size = (2,2),padding = 'Same', 
                     activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
    
model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation = "softmax"))
    
# Define the optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        
# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
    
model.fit(train_images, train_label, epochs=30, batch_size=64)


model.save("model.h5")
#%%





















