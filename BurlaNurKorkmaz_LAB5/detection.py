from PIL import Image
from random import randrange

image = Image.open(r"banana-detection\\bananas_test\\images\\0.png")  
x, y = image.size

crop_size = 32
sample = 10000
sample_list = []
x1_list = []
y1_list = []

for i in range(sample):
    x1 = randrange(0, x - crop_size)
    y1 = randrange(0, y - crop_size)
    x1_list.append(x1)
    y1_list.append(y1)
    sample_list.append(image.crop((x1, y1, x1 + crop_size, y1 + crop_size)))
    
    
#%%
from keras.models import load_model
model = load_model('model.h5')

from skimage import img_as_ubyte
import numpy 
from matplotlib import pyplot as plt

fig = plt.figure(figsize = (10, 7))
prediction_banana = []
x1_banana = []
y1_banana = []

for i in range(sample):
    img = img_as_ubyte(sample_list[i])
    img = numpy.expand_dims(img,axis=0)
    
    prediction = model.predict(img)
    
    if prediction[0][1] > prediction[0][0]:
        prediction_banana.append(model.predict(img))
        x1_banana.append(x1_list[i])
        y1_banana.append(y1_list[i])
        
               
#%%
banana_scores = []
for i in range(len(prediction_banana)):
    banana_scores.append(prediction_banana[i][0][1])    

best_pred = banana_scores.index(max(banana_scores))
best_x1 = x1_banana[best_pred]
best_y1 = y1_banana[best_pred]

#%%
crop_size_banana = 32
sample_banana = 100
sample_list_banana = []
x1_list_banana = []
y1_list_banana = []

number = 32
if(best_x1 - number < 0):
    best_x1 = 0

if(best_y1 - number < 0):
    best_y1 = 0
    
if(best_x1 + number > 255):
    best_x1 = 255
    
if(best_y1 + number > 255):
    best_y1 = 255
      
for i in range(sample_banana):
    
    x1_banana_v2 = randrange(best_x1 - number, best_x1 + number)
    y1_banana_v2 = randrange(best_y1 - number, best_y1 + number)
    x1_list_banana.append(x1_banana_v2)
    y1_list_banana.append(y1_banana_v2)
    sample_list_banana.append(image.crop((x1_banana_v2, y1_banana_v2, x1_banana_v2 + crop_size_banana, y1_banana_v2 + crop_size_banana)))

#%%

fig = plt.figure(figsize = (10, 7))

prediction_banana_v2 = []
x1_banana_v2 = []
y1_banana_v2 = []

for i in range(sample_banana):
    img = img_as_ubyte(sample_list_banana[i])
    img = numpy.expand_dims(img,axis=0)
    
    prediction = model.predict(img)
    
    if prediction[0][1] > prediction[0][0]:
        prediction_banana_v2.append(model.predict(img))
        x1_banana_v2.append(x1_list_banana[i])
        y1_banana_v2.append(y1_list_banana[i])
        

#%%
import cv2 as cv
image = img_as_ubyte(image)
max_x1 = max(x1_banana_v2)
max_y1 = max(y1_banana_v2)

min_x1 = min(x1_banana_v2)
min_y1 = min(y1_banana_v2)

#%%
cv.rectangle(image,(min_x1,min_y1),(max_x1,max_y1),(255,0,0),3)

window_name = 'image'
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
cv.imshow(window_name, image)

cv.waitKey(0) 
cv.destroyAllWindows()  
  















