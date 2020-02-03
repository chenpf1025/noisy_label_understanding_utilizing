from keras.models import load_model
from my_model import create_model
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import data
import numpy as np
import gc
import os
#os.environ['CUDA_VISIBLE_DEVICES']='0'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--noise_ratio',type=float)
parser.add_argument('--noise_pattern',type=str)
parser.add_argument('--dataset',type=str)
args = parser.parse_args()

""" parameters """
noise_ratio = args.noise_ratio
noise_pattern = args.noise_pattern #'sym' or 'asym'
dataset = args.dataset

batch_size = 128
INCV_epochs = 50
INCV_iter = 4
epochs = 200
INCV_name = 'INCV_ResNet32'
save_dir = 'saved_model'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath_INCV = os.path.join(save_dir,INCV_name+'-noisy.h5')

#################################################################################################################################
""" Data preparation """

if dataset=='cifar10':
    x_train, y_train, _, _, x_test, y_test = data.prepare_cifar10_data(data_dir='data/cifar-10-batches-py')
    Num_top = 1 #using top k prediction
    
y_train_noisy = data.flip_label(y_train, pattern=noise_pattern, ratio=noise_ratio, one_hot=True)
input_shape = list(x_train.shape[1:])
n_classes = y_train.shape[1]
n_train = x_train.shape[0]
np.save('y_train_total.npy',y_train)
np.save('y_train_noisy_total.npy',y_train_noisy)
clean_index = np.array([(y_train_noisy[i,:]==y_train[i,:]).all() for i in range(n_train)])# For tracking only, unused during training
noisy_index = np.array([not i for i in clean_index])

# Generator for data augmantation
datagen = ImageDataGenerator(width_shift_range=4./32,  # randomly shift images horizontally (fraction of total width)
                             height_shift_range=4./32,  # randomly shift images vertically (fraction of total height)
                             horizontal_flip=True)  # randomly flip images       

#################################################################################################################################
""" Build model """
# INCV callbacks
class Noisy_acc(Callback):
     
    def on_epoch_end(self, epoch, logs={}):
        if First_half:
            idx = val2_idx # train on the first half while test on the second half 
        else:
            idx = val1_idx
        
        predict = np.argmax(self.model.predict(x_train[idx,:]),axis=1)
        _acc_mix = accuracy_score(np.argmax(y_train_noisy[idx,:],axis=1), predict)
        _acc_clean = accuracy_score(np.argmax(y_train_noisy[idx,:][clean_index[idx],:],axis=1), predict[clean_index[idx]])
        _acc_noisy = accuracy_score(np.argmax(y_train_noisy[idx,:][noisy_index[idx],:],axis=1), predict[noisy_index[idx]])

        print("- acc_mix: %.4f - acc_clean: %.4f - acc_noisy: %.4f\n" % (_acc_mix, _acc_clean, _acc_noisy))
        return
noisy_acc = Noisy_acc()

def INCV_lr_schedule(epoch):
    # Learning Rate Schedule
    lr = 1e-3
    if epoch > 40:
        lr *= 0.1
    elif epoch > 30:
        lr *= 0.25
    elif epoch > 20:
        lr *= 0.5
    print('Learning rate: ', lr)
    return lr
INCV_lr_callback = LearningRateScheduler(INCV_lr_schedule)

# Define optimizer and compile model
optimizer = optimizers.Adam(lr=INCV_lr_schedule(0), beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model = create_model(input_shape=input_shape, classes=n_classes, name=INCV_name, architecture='ResNet32')
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = ['accuracy'])
weights_initial = model.get_weights()
# Print model architecture
print('Architecture of INCV-model:')
model.summary()

##################################################################################################################################
""" INCV iteration """
train_idx = np.array([False for i in range(n_train)])
val_idx = np.array([True for i in range(n_train)])
INCV_save_best = True
for iter in range(1,INCV_iter+1):
    print('INCV iteration %d including first half and second half. In total %d iterations.'%(iter,INCV_iter))
    val_idx_int = np.array([i for i in range(n_train) if val_idx[i]]) # integer index
    np.random.shuffle(val_idx_int)
    n_val_half = int(np.sum(val_idx)/2)
    val1_idx = val_idx_int[:n_val_half] # integer index
    val2_idx = val_idx_int[n_val_half:] # integer index
    #Train model on the first half of dataset
    First_half = True
    print('Iteration ' + str(iter) + ' - first half')
    # reset weights
    model.set_weights(weights_initial)
    results = model.fit_generator(datagen.flow(np.concatenate([x_train[train_idx,:],x_train[val1_idx,:]]),
                                                  np.concatenate([y_train_noisy[train_idx,:],y_train_noisy[val1_idx,:]]),
                                                  batch_size = batch_size),
                                  epochs = INCV_epochs,
                                  validation_data=(x_train[val2_idx,:], y_train_noisy[val2_idx,:]),
                                  callbacks=[ModelCheckpoint(filepath=filepath_INCV, monitor='val_accuracy', verbose=1, save_best_only=INCV_save_best),
                                             noisy_acc,
                                             INCV_lr_callback])
    # Select samples of 'True' prediction
    del model
    model = load_model(filepath_INCV)    
    y_pred = model.predict(x_train[val2_idx,:])
    cross_entropy = np.sum(-y_train_noisy[val2_idx,:]*np.log(y_pred+1e-8),axis=1)
    top_pred = np.argsort(y_pred, axis=1)[:,-Num_top:]
    y_true_noisy = np.argmax(y_train_noisy[val2_idx,:],axis=1)
    top_True = [y_true_noisy[i] in top_pred[i,:] for i in range(len(y_true_noisy))]

    val2train_idx =  val2_idx[top_True]# integer index
    
    # evaluate noisy ratio and compute discard ratio
    if iter == 1:
        eval_ratio = 0.001
        product = np.sum(top_True)/(n_train/2.)
        while (1-eval_ratio)*(1-eval_ratio)+eval_ratio*eval_ratio/(n_classes/Num_top-1) > product:
            eval_ratio += 0.001
            if eval_ratio>=1:
                break
        print('noisy ratio evaluation: %.4f\n' % eval_ratio)
        discard_ratio = min(2, eval_ratio/(1-eval_ratio))       
        discard_idx = val2_idx[np.argsort(cross_entropy)[-int(discard_ratio*np.sum(top_True)):]] # integer index
    
    else:
        discard_idx = np.concatenate([discard_idx, val2_idx[np.argsort(cross_entropy)[-int(discard_ratio*np.sum(top_True)):]]]) # integer index        
    
    print('%d samples selected\n' % (np.sum(train_idx)+val2train_idx.shape[0]))
    
    #Train model on the second half of dataset
    First_half = False
    print('Iteration ' + str(iter) + ' - second half')
    # reset weights
    model.set_weights(weights_initial)
    results = model.fit_generator(datagen.flow(np.concatenate([x_train[train_idx,:],x_train[val2_idx,:]]), 
                                                  np.concatenate([y_train_noisy[train_idx,:],y_train_noisy[val2_idx,:]]),
                                                  batch_size = batch_size),
                                  epochs = INCV_epochs,
                                  validation_data=(x_train[val1_idx,:], y_train_noisy[val1_idx,:]),
                                  callbacks=[ModelCheckpoint(filepath=filepath_INCV, monitor='val_accuracy', verbose=1, save_best_only=INCV_save_best),
                                             noisy_acc,
                                             INCV_lr_callback])
    # Select samples of 'True' prediction
    del model
    model = load_model(filepath_INCV)
    y_pred = model.predict(x_train[val1_idx,:])
    cross_entropy = np.sum(-y_train_noisy[val1_idx,:]*np.log(y_pred+1e-8),axis=1)
    top_pred = np.argsort(y_pred, axis=1)[:,-Num_top:]
    y_true_noisy = np.argmax(y_train_noisy[val1_idx,:],axis=1)
    top_True = [y_true_noisy[i] in top_pred[i,:] for i in range(len(y_true_noisy))]    
    
    val2train_idx =  np.concatenate([val1_idx[top_True],val2train_idx])# integer index
    discard_idx = np.concatenate([discard_idx, val1_idx[np.argsort(cross_entropy)[-int(discard_ratio*np.sum(top_True)):]]])
    train_idx[val2train_idx]=True
    val_idx[val2train_idx]=False
    if noise_pattern == 'sym':
        val_idx[discard_idx]=False
    print('%d samples selected with noisy ratio %.4f\n' % (np.sum(train_idx),
                                                           (1-np.sum(clean_index[train_idx])/np.sum(train_idx))))
    
    if noise_pattern == 'asym' or eval_ratio > 0.6:
        iter_save_best = 1
    elif eval_ratio > 0.3:
        iter_save_best = 2
    else:
        iter_save_best = 4         
        
    if iter==iter_save_best:
        INCV_save_best = False
    
#np.save('train_idx.npy',train_idx)
#np.save('y_train_noisy.npy',y_train_noisy[train_idx,:])
#np.save('y_train.npy',y_train[train_idx,:])
#np.save('val_idx.npy',val_idx)
#np.save('y_val_noisy.npy',y_train_noisy[val_idx,:])
#np.save('y_val.npy',y_train[val_idx,:])
#np.save('y_discard_noisy.npy',y_train_noisy[discard_idx,:])
#np.save('y_discard.npy',y_train[discard_idx,:])

##################################################################################################################################
""" Main training """

model1_name = 'model1_ResNet32'
model2_name = 'model2_ResNet32'
filepath1 = os.path.join(save_dir,model1_name+'-noisy.h5')
filepath2 = os.path.join(save_dir,model2_name+'-noisy.h5')

# bulid model
lr = 1e-3
model1 = create_model(input_shape=input_shape, classes=n_classes, name=model1_name, architecture='ResNet32')
model1.compile(optimizer=optimizers.Adam(lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics = ['accuracy'])
model1.summary()
model2 = create_model(input_shape=input_shape, classes=n_classes, name=model2_name, architecture='ResNet32')
model2.compile(optimizer=optimizers.Adam(lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics = ['accuracy'])


# print loss and accuracy after every epoch
y_test_int = np.argmax(y_test, axis=1) # integer label
train_idx_int = np.array([i for i in range(n_train) if train_idx[i]]) # integer index 
def ep_end(model):
    _acc_test = accuracy_score(y_test_int, np.argmax(model.predict(x_test), axis=1))
    print("- acc_test: %.4f"%_acc_test)
    
    # cross entropy loss, keep track of training losses only and they are not used to supervise training
    idx = train_idx_int[np.random.choice(len(train_idx_int),min(len(train_idx_int),1000),replace=False)] # idx for keep track of training loss
    #idx = np.random.choice(len(x_train),1000)
    predict = model.predict(x_train[idx,:])
    _loss_mix = np.mean(np.sum(-y_train_noisy[idx,:]*np.log(predict+1e-8),axis=1))
    _loss_clean = np.mean(np.sum(-y_train_noisy[idx,:][clean_index[idx],:]*np.log(predict[clean_index[idx],:]+1e-8),axis=1))
    _loss_noisy = np.mean(np.sum(-y_train_noisy[idx,:][noisy_index[idx],:]*np.log(predict[noisy_index[idx],:]+1e-8),axis=1))
    print("- loss_mix: %.4f - loss_clean: %.4f - loss_noisy: %.4f" % (_loss_mix, _loss_clean, _loss_noisy))
    
    predict = np.argmax(predict,axis=1)
    _acc_mix = accuracy_score(np.argmax(y_train_noisy[idx,:],axis=1), predict)
    _acc_clean = accuracy_score(np.argmax(y_train_noisy[idx,:][clean_index[idx],:],axis=1), predict[clean_index[idx]])
    _acc_noisy = accuracy_score(np.argmax(y_train_noisy[idx,:][noisy_index[idx],:],axis=1), predict[noisy_index[idx]])
    print("-  acc_mix: %.4f -  acc_clean: %.4f -  acc_noisy: %.4f\n" % (_acc_mix, _acc_clean, _acc_noisy))
    
    return

# datagen flow for selected set, val1, val2
def merged_datagen_flow(x_S, y_S, x_val, y_val, batch_size_S, batch_size_val):
    gen_S = datagen.flow(x_S, y_S, batch_size=batch_size_S)
    gen_val = datagen.flow(x_val, y_val, batch_size=batch_size_val)
    while True:
        xy_S = gen_S.next()
        xy_val = gen_val.next()
        yield(xy_S[0], xy_S[1], xy_val[0], xy_val[1])

# set keep ratio
if noise_pattern == 'sym':
    class_constant = n_classes - 1
else:
    class_constant = 1

noise_ratio_selected = (eval_ratio*eval_ratio) / (eval_ratio*eval_ratio + class_constant*(1-eval_ratio)*(1-eval_ratio))
print('Evaluated noisy of selected training set: %.4f\n'%noise_ratio_selected)

# parameters of val set      
batch_size_val = min(int(np.sum(val_idx)*batch_size/(np.sum(train_idx))), int(0.5*batch_size))
print('val_size: %d'%(np.sum(val_idx)))
print('batch_size: %d'%batch_size)
print('batch_size_val: %d'%batch_size_val)


print('Main training process...')
Tk = 10
if noise_pattern == 'asym' or eval_ratio>0.6:
    e_warm_up = 80
else:
    e_warm_up = 40
train_size = np.sum(train_idx)
    
for e in range(epochs):    
    if e == 180:
        lr *= 0.5
        model1.compile(optimizer=optimizers.Adam(lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics = ['accuracy'])
        model2.compile(optimizer=optimizers.Adam(lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics = ['accuracy'])
    elif e == 160:
        lr *= 0.1
        model1.compile(optimizer=optimizers.Adam(lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics = ['accuracy'])
        model2.compile(optimizer=optimizers.Adam(lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics = ['accuracy'])
    elif e == 120:
        lr *= 0.1
        model1.compile(optimizer=optimizers.Adam(lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics = ['accuracy'])
        model2.compile(optimizer=optimizers.Adam(lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics = ['accuracy'])
    elif e == 80:
        lr *= 0.1
        model1.compile(optimizer=optimizers.Adam(lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics = ['accuracy'])
        model2.compile(optimizer=optimizers.Adam(lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics = ['accuracy'])

    n_keep = round(batch_size*(1-noise_ratio_selected*min(1,e/Tk)))
    print("Epoch: %d/%d; Learning rate: %.7f; n_keep: %d\n" % (e+1, epochs, lr, n_keep))
    
    batches = 0
    for x_S, y_S, x_val, y_val in merged_datagen_flow(
                                                    x_train[train_idx,:], y_train_noisy[train_idx,:],
                                                    x_train[val_idx,:], y_train_noisy[val_idx,:],
                                                    batch_size_S=batch_size, batch_size_val=batch_size_val):
        
        n_keep = round(len(x_S)*(1-noise_ratio_selected*min(1,e/Tk))) # the last batch size may not be 128
        if e<=e_warm_up:
            # select samples based on model 1
            y_pred = model1.predict(x_S)
            cross_entropy = np.sum(-y_S*np.log(y_pred+1e-8),axis=1)
            batch_idx1 = np.argsort(cross_entropy)[:n_keep]
    
            # select samples based on  model 2
            y_pred = model2.predict(x_S)
            cross_entropy = np.sum(-y_S*np.log(y_pred+1e-8),axis=1)
            batch_idx2 = np.argsort(cross_entropy)[:n_keep]
            
            # training
            model1.train_on_batch(x_S[batch_idx2,:], y_S[batch_idx2,:])  
            model2.train_on_batch(x_S[batch_idx1,:], y_S[batch_idx1,:]) 
        
        else:
            x_batch = np.concatenate([x_S, x_val])
            y_batch = np.concatenate([y_S, y_val])
            
            # select samples based on model 1
            y_pred = model1.predict(x_batch)
            cross_entropy = np.sum(-y_batch*np.log(y_pred+1e-8),axis=1)
            batch_idx1= np.argsort(cross_entropy)[:n_keep]
    
            # select samples based on  model 2
            y_pred = model2.predict(x_batch)
            cross_entropy = np.sum(-y_batch*np.log(y_pred+1e-8),axis=1)
            batch_idx2 = np.argsort(cross_entropy)[:n_keep]
            
            # training
            model1.train_on_batch(x_batch[batch_idx2,:], y_batch[batch_idx2,:])  
            model2.train_on_batch(x_batch[batch_idx1,:], y_batch[batch_idx1,:])              
        
        batches += 1
        if batches >= train_size / batch_size:
            break
        
    ep_end(model1)
    ep_end(model2)
    
    gc.collect()
    
model1.save(filepath1)
model2.save(filepath2)

""" Test """
print('Test on model1')
scores = model1.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

print('Test on model2')
scores = model2.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
