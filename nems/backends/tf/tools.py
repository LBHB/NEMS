import logging

import numpy as np
import tensorflow as tf

log = logging.getLogger(__name__)


class simple_generator(tf.keras.utils.Sequence):
    
    def __init__(self, X, Y, batch_size=32, shuffle=False, frac=1, dtype='float32'):
        'Initialization'
        log.info(f"Setting up simple generator, batch_size={batch_size}")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.frac = frac
        
        if type(X) is dict:
            self.input_names = list(X.keys())
            self.X = X
        else:
            self.X = {'input', X}
            self.input_names = ['input']
        self.Y = Y
        self.output_name = 'output'
        self.index_count = Y.shape[0]
        
        self.dtype = dtype 
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))
    
    def __getitem__(self, i):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[i*self.batch_size:(i+1)*self.batch_size]

        if self.frac<0:
            f = int(np.round((1+self.frac)*len(indexes)))
            indexes = indexes[slice(f,None)]
        elif (self.frac<1) & (self.frac>0):
            f = int(np.round(self.frac*len(indexes)))
            indexes = indexes[slice(0,f)]

        # Generate data
        return self.__data_generation(indexes)
            
    def copy(self):
        return copy.copy(self)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.index_count)
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def stim_shape(self):
        return self.X['input'].shape
    
    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)
        # Initialization
        # stim.shape, resp.shape
        # (300, 2000, 19)), ((300, 2000, 30)
        if len(self.input_names)==1:
            X = self.X[self.input_names[0]][indexes]
        else:
            X = {i: self.X[i][indexes] for i in self.input_names}
        y = self.Y[indexes]
            
        return X, y

