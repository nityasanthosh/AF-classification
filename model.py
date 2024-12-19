from tensorflow.keras.utils import Sequence
from CNN import file_paths,labels
from scipy.io import loadmat
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
class Datagenerator(Sequence): 
    def __init__(self,file_paths,labels,batch_size,shuffle=True,target_shape=(9000,1),num_class=4):
        self.file_paths=file_paths
        self.labels=labels
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.target_shape=target_shape
        self.num_class=num_class
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.file_paths) // self.batch_size))
    
    def __getitem__(self,idx):
        batch_paths=self.file_paths[idx*self.batch_size:(idx+1)*self.batch_size]                                                                                     
        batch_labels=self.labels[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_labels=to_categorical(batch_labels,num_classes=self.num_class)
        batch_data=[self.load_mat_file(path) for path in batch_paths]
        return tf.convert_to_tensor(batch_data),np.array(batch_labels)
    
    def on_epoch_end(self):
        if self.shuffle:
            data=list(zip(self.file_paths,self.labels))
            np.random.shuffle(data)
            self.file_paths,self.labels=zip(*data)
            
    def load_mat_file(self,path):
        ecg_dict=loadmat(path)
        ecg_data=ecg_dict['val'].flatten()
        ecg_data = np.pad(ecg_data, (0, max(0, self.target_shape[0] - len(ecg_data))), 'constant')
        ecg_data = ecg_data[:self.target_shape[0]]  
        ecg_data = ecg_data.reshape(self.target_shape)
        return ecg_data  
    
batch_size=32

train_paths, val_paths, train_labels, val_labels = train_test_split(
    file_paths,  
    labels,    
    test_size=0.2,  
    random_state=42, 
    stratify=labels  
)

train_generator=Datagenerator(train_paths,train_labels,batch_size=batch_size)
print('train_generated')
val_generator=Datagenerator(val_paths,val_labels,batch_size=batch_size)
print('train_generated')
