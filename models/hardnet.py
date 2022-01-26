from re import M
import numpy as np
import cv2
from os.path import dirname ,join
import glob
import PIL.Image
from keras.models import *
from keras.layers import *
from tensorflow.keras.optimizers import  *
import matplotlib.pyplot as plt


################################
#1er HARDBLK 4layers k=16, m=1.6
################################

#Calculer les k*m
def hardbloc4layers(x, k=16, m=1.6):
  out_channels = k*m
  out_channels = int(int(out_channels +1 )/2 )*2

  out_channels_2 = k*m*m
  out_channels_2 = int(int(out_channels_2 +1 )/2 )*2

  out_channels_3 = k*m*m*m
  out_channels_3 = int(int(out_channels_3 +1 )/2 )*2

  out_channels_4 = k*m*m*m*m
  out_channels_4 = int(int(out_channels_4 +1 )/2 )*2
###############################
#First Block with 4 layers 
###############################

#First layer with depth of k:
  conv3 = Conv2D(k, kernel_size=(3, 3),strides=1 ,padding='same')(x)
  bn1 = BatchNormalization()(conv3)
  relu1 = Activation('relu')(bn1)

#Second layer with depth of k*m
  relu1_reshaped = Resizing(56,56)(relu1)
  merge1 = concatenate([relu1_reshaped, x],axis=3)
  conv4 = Conv2D(out_channels, kernel_size=(3, 3),strides=1, padding='same')(merge1)
  bn2 = BatchNormalization()(conv4)
  relu2 = Activation('relu')(bn2)

#Third layer with depth of k
  conv5 = Conv2D(k, kernel_size=(3, 3),strides=1, padding='same')(relu2)
  bn3 =BatchNormalization()(conv5)
  relu3 = Activation('relu')(bn3)

#Fouth layer with depth of k*m^2
  relu2_reshaped = Resizing(56,56)(relu2)
  relu3_reshaped = Resizing(56,56)(relu3)
  merge2 = concatenate([relu3_reshaped, relu2_reshaped, x],axis=3)
  conv6 = Conv2D(out_channels_2, kernel_size=(3, 3),strides=1, padding='same')(merge2)
  bn4 = BatchNormalization()(conv6)
  relu4 = Activation('relu')(bn4)

  return [relu4, relu3, relu1 ]

###############################
#Second Block with 16 layers 
###############################
def hardbloc16layers(x, k=20, m=1.6):
  out_channels = k*m
  out_channels = int(int(out_channels +1 )/2 )*2

  out_channels_2 = k*m*m
  out_channels_2 = int(int(out_channels_2 +1 )/2 )*2

  out_channels_3 = k*m*m*m
  out_channels_3 = int(int(out_channels_3 +1 )/2 )*2

  out_channels_4 = k*m*m*m*m
  out_channels_4 = int(int(out_channels_4 +1 )/2 )*2

  out_channels_5 = k*m*m*m*m*m
  out_channels_5 = int(int(out_channels_5 +1 )/2 )*2

#First layer with depth of k
  block2_conv1 = Conv2D(k, kernel_size=(3, 3),strides=1 ,padding='same')(x)
  block2_bn1 = BatchNormalization()(block2_conv1)
  block2_relu1 = Activation('relu')(block2_bn1)

#Second layer with depth of k*m
  block2_merge1 = concatenate([block2_relu1, x],axis=3)
  block2_conv2 = Conv2D(out_channels, kernel_size=(3, 3),strides=1, padding='same')(block2_merge1)
  block2_bn2 = BatchNormalization()(block2_conv2)
  block2_relu2 = Activation('relu')(block2_bn2)

#Third layer with depth of k
  block2_conv3 = Conv2D(k, kernel_size=(3, 3),strides=1, padding='same')(block2_relu2)
  block2_bn3 =BatchNormalization()(block2_conv3)
  block2_relu3 = Activation('relu')(block2_bn3)

#Fouth layer with depth of k*m^2
  block2_merge2 = concatenate([block2_relu3,block2_relu2, x],axis=3)
  block2_conv4 = Conv2D(out_channels_2, kernel_size=(3, 3),strides=1, padding='same')(block2_merge2)
  block2_bn4 = BatchNormalization()(block2_conv4)
  block2_relu4 = Activation('relu')(block2_bn4)

#Fifth layer with depoth of k
  block2_conv5 = Conv2D(k, kernel_size=(3, 3),strides=1, padding='same')(block2_relu4)
  block2_bn5 =BatchNormalization()(block2_conv5)
  block2_relu5 = Activation('relu')(block2_bn5)

#Sixth layer with depth of k*m
  block2_merge3 = concatenate([block2_relu5,block2_relu4],axis=3)
  block2_conv6 = Conv2D(out_channels, kernel_size=(3, 3),strides=1, padding='same')(block2_merge3)
  block2_bn6 =BatchNormalization()(block2_conv6)
  block2_relu6 = Activation('relu')(block2_bn6)

#Seventh layer with depth of k 
  block2_conv7 = Conv2D(k, kernel_size=(3, 3),strides=1, padding='same')(block2_relu6)
  block2_bn7 =BatchNormalization()(block2_conv7)
  block2_relu7 = Activation('relu')(block2_bn7)

#Eith layer with depth of k*m^3
  block2_merge4 = concatenate([block2_relu7, block2_relu6, block2_relu4, x],axis=3)
  block2_conv8 = Conv2D(out_channels_3, kernel_size=(3, 3),strides=1, padding='same')(block2_merge4)
  block2_bn8 =BatchNormalization()(block2_conv8)
  block2_relu8 = Activation('relu')(block2_bn8)

#Nineth layer with depth of k 
  block2_conv9 = Conv2D(k, kernel_size=(3, 3),strides=1, padding='same')(block2_relu8)
  block2_bn9 =BatchNormalization()(block2_conv9)
  block2_relu9 = Activation('relu')(block2_bn9)

#Tenth layer with depth of k*m
  block2_merge5 = concatenate([block2_relu9, block2_relu8],axis=3)
  block2_conv10 = Conv2D(out_channels, kernel_size=(3, 3),strides=1, padding='same')(block2_merge5)
  block2_bn10 =BatchNormalization()(block2_conv10)
  block2_relu10 = Activation('relu')(block2_bn10)

#Eleventh layer with depth of k 
  block2_conv11 = Conv2D(k, kernel_size=(3, 3),strides=1, padding='same')(block2_relu10)
  block2_bn11 =BatchNormalization()(block2_conv11)
  block2_relu11 = Activation('relu')(block2_bn11)

#Twelfth layer with depth of k*m^4
  block2_merge6 = concatenate([block2_relu11 , block2_relu10, block2_relu8],axis=3)
  block2_conv12 = Conv2D(out_channels_4, kernel_size=(3, 3),strides=1, padding='same')(block2_merge6)
  block2_bn12 =BatchNormalization()(block2_conv12)
  block2_relu12 = Activation('relu')(block2_bn12)

#thirteenth layer with depth of k 
  block2_conv13 = Conv2D(k, kernel_size=(3, 3),strides=1, padding='same')(block2_relu12)
  block2_bn13 =BatchNormalization()(block2_conv13)
  block2_relu13 = Activation('relu')(block2_bn13)

#Fourteenth layer with depth of k*m
  block2_merge7 = concatenate([block2_relu13, block2_relu12],axis=3)
  block2_conv14 = Conv2D(k, kernel_size=(3, 3),strides=1, padding='same')(block2_merge7)
  block2_bn14 =BatchNormalization()(block2_conv14)
  block2_relu14 = Activation('relu')(block2_bn14)

#Fifteenth layer with depth of k 
  block2_conv15 = Conv2D(k, kernel_size=(3, 3),strides=1, padding='same')(block2_relu14)
  block2_bn15 =BatchNormalization()(block2_conv15)
  block2_relu15 = Activation('relu')(block2_bn15)

#Sixteenth layer with depth of k*m^5
  block2_merge8 = concatenate([block2_relu15, block2_relu14, block2_relu12, block2_relu8],axis=3)
  block2_conv16 = Conv2D(out_channels_5 , kernel_size=(3, 3),strides=1, padding='same')(block2_merge8)
  block2_bn16 =BatchNormalization()(block2_conv16)
  block2_relu16 = Activation('relu')(block2_bn16)

  return [block2_relu16 , block2_relu15, block2_relu13, block2_relu11 , block2_relu9 , block2_relu7 ,block2_relu5 ,block2_relu3, block2_relu1 ]

###############################
#End of the Second Block 
###############################


###############################
#Third Block with 8 layers 
###############################

#Calculer les k*m
def hardbloc8layers(x, k=64, m=1.6):
  out_channels = k*m
  out_channels = int(int(out_channels +1 )/2 )*2

  out_channels_2 = k*m*m
  out_channels_2 = int(int(out_channels_2 +1 )/2 )*2

  out_channels_3 = k*m*m*m
  out_channels_3 = int(int(out_channels_3 +1 )/2 )*2

  #First layer with depth of k
  block3_conv1 = Conv2D(k, kernel_size=(3, 3),strides=1 ,padding='same')(x)
  block3_bn1 = BatchNormalization()(block3_conv1)
  block3_relu1 = Activation('relu')(block3_bn1)

#Second layer with depth of k*m
  block3_merge1 = concatenate([block3_relu1, x],axis=3)
  block3_conv2 = Conv2D(out_channels, kernel_size=(3, 3),strides=1, padding='same')(block3_merge1)
  block3_bn2 = BatchNormalization()(block3_conv2)
  block3_relu2 = Activation('relu')(block3_bn2)

#Third layer with depth of k
  block3_conv3 = Conv2D(k, kernel_size=(3, 3),strides=1, padding='same')(block3_relu2)
  block3_bn3 =BatchNormalization()(block3_conv3)
  block3_relu3 = Activation('relu')(block3_bn3)

#Fouth layer with depth of k*m^2
  block3_merge2 = concatenate([block3_relu3,block3_relu2, x],axis=3)
  block3_conv4 = Conv2D(out_channels_2, kernel_size=(3, 3),strides=1, padding='same')(block3_merge2)
  block3_bn4 = BatchNormalization()(block3_conv4)
  block3_relu4 = Activation('relu')(block3_bn4)

#Fifth layer with depoth of k
  block3_conv5 = Conv2D(k, kernel_size=(3, 3),strides=1, padding='same')(block3_relu4)
  block3_bn5 =BatchNormalization()(block3_conv5)
  block3_relu5 = Activation('relu')(block3_bn5)

#Sixth layer with depth of k*m
  block3_merge3 = concatenate([block3_relu5,block3_relu4],axis=3)
  block3_conv6 = Conv2D(out_channels, kernel_size=(3, 3),strides=1, padding='same')(block3_merge3)
  block3_bn6 =BatchNormalization()(block3_conv6)
  block3_relu6 = Activation('relu')(block3_bn6)

#Seventh layer with depth of k 
  block3_conv7 = Conv2D(k, kernel_size=(3, 3),strides=1, padding='same')(block3_relu6)
  block3_bn7 =BatchNormalization()(block3_conv7)
  block3_relu7 = Activation('relu')(block3_bn7)

#Eith layer with depth of k*m^3
  block3_merge4 = concatenate([block3_relu7, block3_relu6, block3_relu4, x],axis=3)
  block3_conv8 = Conv2D(out_channels_3, kernel_size=(3, 3),strides=1, padding='same')(block3_merge4)
  block3_bn8 =BatchNormalization()(block3_conv8)
  block3_relu8 = Activation('relu')(block3_bn8)

  return [block3_relu8, block3_relu7, block3_relu5, block3_relu3, block3_relu1 ]


###############################
#Fourth Block with 4 layers 
###############################
#Calculer les k*m
def hardbloc4layers_2(x, k=160, m=1.6):
  out_channels = k*m
  out_channels = int(int(out_channels +1 )/2 )*2

  out_channels_2 = k*m*m
  out_channels_2 = int(int(out_channels_2 +1 )/2 )*2

  out_channels_3 = k*m*m*m
  out_channels_3 = int(int(out_channels_3 +1 )/2 )*2
  
  #First layer with depth of k
  block4_conv1 = Conv2D(k, kernel_size=(3, 3),strides=1 ,padding='same')(x)
  block4_bn1 = BatchNormalization()(block4_conv1)
  block4_relu1 = Activation('relu')(block4_bn1)

  #Second layer with depth of k*m
  block4_merge1 = concatenate([block4_relu1,x],axis=3)
  block4_conv2 = Conv2D(out_channels, kernel_size=(3, 3),strides=1, padding='same')(block4_merge1)
  block4_bn2 = BatchNormalization()(block4_conv2)
  block4_relu2 = Activation('relu')(block4_bn2)

  #Third layer with depth of k
  block4_conv3 = Conv2D(k, kernel_size=(3, 3),strides=1, padding='same')(block4_relu2)
  block4_bn3 =BatchNormalization()(block4_conv3)
  block4_relu3 = Activation('relu')(block4_bn3)

  #Fouth layer with depth of k*m^2
  block4_merge2 = concatenate([block4_relu3,block4_relu2, x],axis=3)
  block4_conv4 = Conv2D(out_channels_2, kernel_size=(3, 3),strides=1, padding='same')(block4_merge2)
  block4_bn4 = BatchNormalization()(block4_conv4)
  block4_relu4 = Activation('relu')(block4_bn4)

  return [block4_relu4, block4_relu3, block4_relu1]


###############################
#End of the Fourth Block 
###############################

#################
#### HARDNET39DS
#################
def hardnet(input_shape = (224, 224, 3)):
  inputs = Input(input_shape)
  
  #1er couche conv 3x3,32,stide = 2
  conv1 = Conv2D(24, kernel_size=(3, 3), strides=2 )(inputs)

  #2eme couche conv 1x1,48
  conv2 = Conv2D(48, kernel_size=(1, 1), strides=1 )(conv1)

  #MaxPooling avec padding = 1 
  padd1 = ZeroPadding2D(padding=(1, 1))(conv2)
  maxpool1 = MaxPooling2D(pool_size=(3, 3), strides=2 )(padd1)

  block4layers = hardbloc4layers(maxpool1)
  mergeblock1 = concatenate(block4layers, axis=3)
  conv7 = Conv2D(96, kernel_size=(1, 1),strides=1, padding='same')(mergeblock1)

  block16layers = hardbloc16layers(conv7)
  mergeblock2 = concatenate(block16layers, axis=3)
  block2_end_conv = Conv2D(320, kernel_size=(1, 1),strides=1, padding='same')(mergeblock2)

  block16layers = hardbloc8layers(block2_end_conv)
  mergeblock3 = concatenate(block16layers, axis=3)
  block3_end_conv = Conv2D(640, kernel_size=(1, 1),strides=1, padding='same')(mergeblock3)

  block4layers = hardbloc4layers_2(block3_end_conv)
  mergeblock4 = concatenate(block4layers, axis=3)
  block4_end_conv = Conv2D(1024, kernel_size=(1, 1),strides=1, padding='same')(mergeblock4)

  Up1 = UpSampling2D((2,2))(block4_end_conv)
  conv8 = Conv2D(3, kernel_size=(3, 3),strides=1, padding='same')(Up1)
  bn5 = BatchNormalization()(conv8)
  relu5 = Activation('relu')(bn5)
  Up2 = UpSampling2D((2,2))(relu5)
  bn6 = BatchNormalization()(Up2)
  Soft = Activation('Softmax')(bn6)

  model = Model(inputs, Soft)

  return model



if __name__ == '__main__':
  model = hardnet()
  print(model.summary())

"""
optimizer = SGD(learning_rate = 0.05)

#Charger les images 
xtrain, xtest = x_data()
ytrain, ytest = Ytrain_data(), Ytest_data()

model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()


h = model.fit(xtrain, ytrain, batch_size = 8, epochs = 150, validation_data = (xtest, ytest), verbose = 1)


#######################
#Save the model 
#######################

#model.save(join(current_dir, "Save"))

#######################
#Load the model 
#######################

#model = load_model(join(current_dir, "Save"))


#######################
#Show some segmented images
#######################
# create figure
fig = plt.figure(figsize=(10, 7))
  
# setting values to rows and column variables
rows = 2
columns = 2
  
# reading images
Image1 = model.predict( np.asarray( [xtrain[5]]))
Image2 = model.predict( np.asarray( [xtrain[6]]))
Image3 = model.predict( np.asarray( [xtest[1]]))
Image4 = model.predict( np.asarray( [xtest[6]]))
  
# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)
  
# showing image
plt.imshow(np.reshape(Image1,(224,224,3)))
plt.axis('off')
plt.title("First")
  
# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
  
# showing image
plt.imshow(np.reshape(Image2,(224,224,3)))
plt.axis('off')
plt.title("Second")
  
# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)
  
# showing image
plt.imshow(np.reshape(Image3,(224,224,3)))
plt.axis('off')
plt.title("Third")
  
# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 4)
  
# showing image
plt.imshow(np.reshape(Image4,(224,224,3)))
plt.axis('off')
plt.title("Fourth")
plt.show()
"""


##############
#Courbes d'accuracy et de loss
##############
"""
import matplotlib.pyplot as plt

plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig("accuracy.jpg")

plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig("loss.jpg")

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
score = list()

p = model.predict(xtest)

for i, j in zip(xtest, ytest):
  p = model.predict(np.asarray([i]))[0] > 0.5
  p = p.astype(np.int32)
  acc = accuracy_score(j.flatten(), p.flatten())
  recall = recall_score(j.flatten(), p.flatten(), labels = [0, 1, 2])
  prec = precision_score(j.flatten(), p.flatten(), labels = [0, 1, 2])
  f1score = f1_score(j.flatten(), p.flatten(), labels = [0, 1, 2])

  score.append( (acc, f1score, recall, prec) )

t = np.mean(score, axis = 0)
print(f"Accuracy :{t[0]:0.5f}")
print(f"Precision score :{t[3]:0.5f}")
print(f"F1_score :{t[1]:0.5f}")
print(f"Recall :{t[2]:0.5f}")


HardNEt39DS.py
Affichage de HardNEt39DS.py en cours
"""
