#import libraries
import numpy as np
#import matplotlib.pyplot as plt


from tensorflow.keras import regularizers

from keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical

from keras.layers import Dropout


from keras.datasets import cifar10
import copy 

#import CIFAR10
NUM_CLASSES = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(y_train[5][0])

x_test_org=x_test


#Change number of classes to 3
#Cat,Dog, others
for i in range (0,len(x_train)):
    if (y_train[i][0]<3 or y_train[i][0]==4 or y_train[i][0]>5):
        y_train[i]=2;
    elif (y_train[i][0]==3):
        y_train[i]=0
    elif (y_train[i][0]==5):
        y_train[i]=2

for i in range (0,len(x_test)):
    if (y_test[i][0]<3 or y_test[i][0]==4 or y_test[i][0]>5):
        y_test[i]=2;
    elif (y_test[i][0]==3):
        y_test[i]=0
    elif (y_test[i][0]==5):
        y_test[i]=2
        
NUM_CLASSES = 3

#change it back to 10 classes
NUM_CLASSES = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(y_train[5][0])

actual_single1=y_test;

#=============================
#normalization
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

meanTrainImage=np.zeros(3, dtype=float)
for j in range (0, 3):
    meanTrainImage[j] = np.mean(x_train[:,:,:,j].flatten())


for i in range (0, len(x_train)):
    for j in range (0, 3):
        x_train[i,:,:,j] = copy.copy(x_train[i,:,:,j] - meanTrainImage[j])/255

        
meanTestImage=np.zeros(3, dtype=float)
for j in range (0, 3):
    meanTestImage[j] = np.mean(x_test[:,:,:,j].flatten())


for i in range (0, len(x_test)):
    for j in range (0, 3):
        x_test[i,:,:,j] = copy.copy(x_test[i,:,:,j] - meanTestImage[j])/255
        


print(x_train.shape)    

#categorized the target
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)




#network structure:
input_layer = Input((32,32,3))

x=input_layer

x = Conv2D(32, (3, 3), padding="same", activation="relu") (x)
x = MaxPooling2D(pool_size=(2, 2),strides=(2, 2), padding='valid')(x)


x = Conv2D(16, (3, 3), padding="same", activation="relu") (x)
x = MaxPooling2D(pool_size=(2, 2),strides=(2, 2), padding='valid')(x)


x = Flatten()(x)

output_layer = Dense(NUM_CLASSES, activation = 'softmax')(x)

model = Model(input_layer, output_layer)




model.summary()


#Optimization part for the back propagation
opt = Adam(lr=0.0001) #learning rate (lr)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#model.compile(loss=keras.losses.categorical_crossentropy,
              #optimizer=keras.optimizers.Adadelta(),
              #metrics=['accuracy'])

#Training part
model.fit(x_train[1:10000], y_train[1:10000], validation_split=0.33 
          , batch_size=32
          , epochs=50
          , shuffle=True)



#Evaluate the model
model.evaluate(x_test, y_test)


CLASSES = np.array(['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck'])

preds = model.predict(x_test)
preds.shape
preds_single = CLASSES[np.argmax(preds, axis = -1)]
actual_single = CLASSES[np.argmax(y_test, axis = -1)]


#Accuracy of each class
class_correct = np.zeros (10)# list(0. for i in range(10))
class_total = np.zeros (10) # list(0. for i in range(10))

for i in range (0,len(x_test)):
    if (actual_single[i]==preds_single[i]):
        class_correct[actual_single1[i]]=class_correct[actual_single1[i]]+1;
    class_total[actual_single1[i]]=class_total[actual_single1[i]]+1;
    
for j in range (0,10):
    print("Accuracy of ",CLASSES[j], ":",class_correct[j]/class_total[j])
    
    

