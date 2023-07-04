from Models.hardnet import *
from data_pre_processing import *


d = data([0, 1, 2], (224, 224))


from tensorflow.keras.optimizers import *
from keras.callbacks import ModelCheckpoint, TensorBoard

# hyper parameters
model_name = "hardnet39DS.hdf5"

model = hardnet()

print(model.summary())

optimizer = SGD(learning_rate = 0.05)

model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

model_checkpoint = ModelCheckpoint(model_name, monitor = 'loss', save_best_only = True, verbose = True)
tensorboard = TensorBoard()

h = model.fit(d.Xtrain, d.Ytrain, batch_size = 8, epochs = 150, validation_data = (d.Xtest, d.Ytest), callbacks = [tensorboard, model_checkpoint])


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
p = model.predict(d.Xtest)

for i, j in zip(d.Xtest, d.Ytest):
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

predictions = model.predict( np.asarray( [d.Xtest[1]]))

print(type(predictions))
print(predictions.shape)
plt.imshow( np.reshape(predictions,(256,256,3)) ) 
plt.show()

plt.imshow( d.Xtest[1] ) 
plt.show()

plt.imshow( d.Ytest[1] ) 
plt.show()
