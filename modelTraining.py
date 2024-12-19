import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from model import train_generator,val_generator
def CNN_model(input_shape):
    model=Sequential()
    
    model.add(Conv1D(filters=64,kernel_size=3,activation='relu',input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(filters=128,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    
    model.add(Conv1D(filters=256,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64,activation='relu'))
    
    model.add(Dense(4,activation='softmax'))
    
    return model

input_shape=(9000,1)
model=CNN_model(input_shape)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history =model.fit(train_generator,validation_data=val_generator,epochs=10)