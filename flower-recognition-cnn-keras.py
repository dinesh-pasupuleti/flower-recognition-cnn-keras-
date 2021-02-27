#!/usr/bin/env python
# coding: utf-8

# # Flower Recognition CNN Keras

# 

# In[45]:


import os
print(os.listdir(r'C:\Users\User\Downloads\flowers'))


# In[93]:


species=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


# ## CONTENTS ::

# <a id="content1"></a>
# ## 1 ) Importing Various Modules.

# In[46]:



import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
 

get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder


from keras.preprocessing.image import ImageDataGenerator


from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical


from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
 
import tensorflow as tf
import random as rn


import cv2                  
import numpy as np  
from tqdm import tqdm
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image


# <a id="content2"></a>
# ## 2 ) Preparing the Data

# ## 2.1) Making the functions to get the training and validation set from the Images

# In[47]:


X=[]
Z=[]
IMG_SIZE=150
FLOWER_DAISY_DIR=r'C:\Users\User\Downloads\flowers\daisy'
FLOWER_SUNFLOWER_DIR=r'C:\Users\User\Downloads\flowers\sunflower'
FLOWER_TULIP_DIR=r'C:\Users\User\Downloads\flowers\tulip'
FLOWER_DANDI_DIR=r'C:\Users\User\Downloads\flowers\dandelion'
FLOWER_ROSE_DIR=r'C:\Users\User\Downloads\flowers\rose'


# In[48]:


def assign_label(img,flower_type):
    return flower_type
    


# In[49]:


def make_train_data(flower_type,DIR):
    for img in tqdm(os.listdir(DIR)):
        label=assign_label(img,flower_type)
        path = os.path.join(DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        
        X.append(np.array(img))
        Z.append(str(label))
        
        
        


# In[50]:


make_train_data('Daisy',FLOWER_DAISY_DIR)
print(len(X))


# In[51]:


make_train_data('Sunflower',FLOWER_SUNFLOWER_DIR)
print(len(X))


# In[52]:


make_train_data('Tulip',FLOWER_TULIP_DIR)
print(len(X))


# In[54]:


make_train_data('Dandelion',FLOWER_DANDI_DIR)
print(len(X))


# In[53]:


make_train_data('Rose',FLOWER_ROSE_DIR)
print(len(X))


# ## 2.2 ) Visualizing some Random Images

# In[55]:


fig,ax=plt.subplots(5,2)
fig.set_size_inches(15,15)
for i in range(5):
    for j in range (2):
        l=rn.randint(0,len(Z))
        ax[i,j].imshow(X[l])
        ax[i,j].set_title('Flower: '+Z[l])
        
plt.tight_layout()
        


# ## 2.3 ) Label Encoding the Y array (i.e. Daisy->0, Rose->1 etc...) & then One Hot Encoding

# In[56]:


le=LabelEncoder()
Y=le.fit_transform(Z)
Y=to_categorical(Y,5)
X=np.array(X)
X=X/255


# ## 2.4 ) Splitting into Training and Validation Sets

# In[57]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)


# ## 2.5 ) Setting the Random Seeds

# In[59]:


np.random.seed(42)
rn.seed(42)


# In[ ]:





# <a id="content3"></a>
# ## 3 ) Modelling

# ## 3.1 ) Building the ConvNet Model

# In[60]:


# # modelling starts using a CNN.

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
 

model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(5, activation = "softmax"))


# ## 3.2 ) Using a LR Annealer

# In[61]:


batch_size=128
epochs=50

from keras.callbacks import ReduceLROnPlateau
red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)


# ## 3.3 ) Data Augmentation to prevent Overfitting

# In[62]:


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)


# ## 3.4 ) Compiling the Keras Model & Summary

# In[63]:


model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])


# In[64]:


model.summary()


# ## 3.5 ) Fitting on the Training set and making predcitons on the Validation set

# In[65]:


History = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)
# model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data = (x_test,y_test))


# In[ ]:





# <a id="content4"></a>
# ## 4 ) Evaluating the Model Performance

# In[66]:


plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()


# In[69]:


plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()


# <a id="content5"></a>
# ## 5 ) Visualizing Predictons 

# ### Predicting a random image

# In[83]:


img1 = Image.open(r'C:\Users\User\Desktop\predictit.jpg')
imResize = img1.resize((150,150), Image.ANTIALIAS)
imResize.save('predictit.jpg', 'JPEG', quality=90)


# In[85]:


predictimg=cv2.imread("predictit.jpg",1)
predictimg=np.array(predictimg)
plt.imshow(predictimg)
predictimg=predictimg/255.0

predictimg = np.expand_dims(predictimg, axis=0)
predictimg.shape


# In[88]:


predition=model.predict(predictimg)

predition=np.squeeze(predition)
print(predition)
predIndex=np.argmax(predition)


# In[92]:


print("The Species of given image is",species[predIndex])


# # THE END
