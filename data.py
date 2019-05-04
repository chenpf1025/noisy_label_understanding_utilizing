
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import pickle
import numpy as np

from six.moves import urllib

import tensorflow as tf

def prepare_cifar10_data(data_dir, val_size=0, rescale=True):
    # swap, orginally should be reshape to 3*32*32
    train_data = []
    train_labels = []
    for i in range(5):
        file = data_dir+'/data_batch_'+str(i+1)
        with open(file, 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
        train_data.append(d[b'data'])
        train_labels.append(d[b'labels'])
    train_data = np.concatenate(train_data,axis=0)
    train_labels = np.concatenate(train_labels,axis=0)

    file = data_dir+'/test_batch'
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    test_data = d[b'data']
    test_labels = np.array(d[b'labels'])
    
    validation_data = train_data[:val_size, :]
    validation_labels = train_labels[:val_size]
    train_data = train_data[val_size:, :]
    train_labels = train_labels[val_size:]

    # convert to one-hot labels
    n_class=10
    train_labels = np.eye(n_class)[train_labels]
    validation_labels = np.eye(n_class)[validation_labels]
    test_labels = np.eye(n_class)[test_labels]
    
    if rescale:
        train_data = train_data/255
        validation_data = validation_data/255
        test_data = test_data/255

    train_data = train_data.reshape(-1,3,32,32).transpose([0, 2, 3, 1])
    
    if val_size>0:
        validation_data = validation_data.reshape(-1,3,32,32).transpose([0, 2, 3, 1])
    
    test_data = test_data.reshape(-1,3,32,32).transpose([0, 2, 3, 1])
       
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels
    
def flip_label(y, pattern, ratio, one_hot=True):
    #y: true label, one hot
    #pattern: 'pair' or 'sym'
    #p: float, noisy ratio
    
    #convert one hot label to int
    if one_hot:
        y = np.argmax(y,axis=1)#[np.where(r==1)[0][0] for r in y]
    n_class = max(y)+1
    
    #filp label
    for i in range(len(y)):
        if pattern=='sym':
            p1 = ratio/(n_class-1)*np.ones(n_class)
            p1[y[i]] = 1-ratio
            y[i] = np.random.choice(n_class,p=p1)
        elif pattern=='asym':
            y[i] = np.random.choice([y[i],(y[i]+1)%n_class],p=[1-ratio,ratio])            
            
    #convert back to one hot
    if one_hot:
        y = np.eye(n_class)[y]
    return y
