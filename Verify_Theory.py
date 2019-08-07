
from my_model import create_model
from keras.callbacks import Callback, LearningRateScheduler
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from sklearn.metrics import accuracy_score
import data
import numpy as np
import os
#os.environ['CUDA_VISIBLE_DEVICES']='1'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--noise_ratio',type=float)
parser.add_argument('--noise_pattern',type=str)
args = parser.parse_args()

""" parameters """
noise_ratio = args.noise_ratio
noise_pattern = args.noise_pattern #'sym' or 'asym'
batch_size = 128
epochs = 200
save_dir = 'Theory'
network = 'ResNet110'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir,network+'.h5')
print('\n#######################################\n noise_ratio: %.2f noise_pattern: %s\n#######################################\n'
      %(noise_ratio,noise_pattern))

#################################################################################################################################
""" Data preparation """
x_train, y_train, _, _, x_test, y_test = data.prepare_cifar10_data(data_dir='data/cifar-10-batches-py')
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
                             horizontal_flip=True
                             )  # randomly flip images    

#################################################################################################################################
""" Build model """

val_idx = np.array([True for i in range(n_train)])
val_idx_int = np.array([i for i in range(n_train) if val_idx[i]]) # integer index
np.random.shuffle(val_idx_int)
n_val_half = int(np.sum(val_idx)/2)
val1_idx = val_idx_int[:n_val_half] # integer index
val2_idx = val_idx_int[n_val_half:] # integer index

#checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=False)

class Noisy_acc(Callback):
    
    def on_epoch_end(self, epoch, logs={}):
        
        idx = val2_idx[np.random.choice(len(val2_idx),1000)] # train on the first half while test on the second half 
        
        predict = self.model.predict(x_train[idx,:])
        predict = np.argmax(predict,axis=1)
        _acc_mix = accuracy_score(np.argmax(y_train_noisy[idx,:],axis=1), predict)
        _acc_clean = accuracy_score(np.argmax(y_train_noisy[idx,:][clean_index[idx],:],axis=1), predict[clean_index[idx]])
        _acc_noisy = accuracy_score(np.argmax(y_train_noisy[idx,:][noisy_index[idx],:],axis=1), predict[noisy_index[idx]])

        print("- acc_mix: %.4f - acc_clean: %.4f - acc_noisy: %.4f\n" % (_acc_mix, _acc_clean, _acc_noisy))
        return
noisy_acc = Noisy_acc()

def lr_schedule(epoch):
    # Learning Rate Schedule
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
lr_callback = LearningRateScheduler(lr_schedule)

# Define optimizer and compile model
optimizer = optimizers.Adam(lr_schedule(0), beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model = create_model(input_shape=input_shape, classes=n_classes, name=network, architecture=network)
model.summary()

parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = ['accuracy'])


##################################################################################################################################
""" Main training """
results = parallel_model.fit_generator(datagen.flow(x_train[val1_idx,:], y_train_noisy[val1_idx,:], batch_size = batch_size),
                              epochs = epochs,
                              validation_data=(x_train[val2_idx,:], y_train_noisy[val2_idx,:]),
                              callbacks=[noisy_acc, lr_callback])

""" Test """
scores = parallel_model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])



y_pred = np.argmax(parallel_model.predict(x_train[val2_idx,:]), axis=1)
y_true_noisy = np.argmax(y_train_noisy[val2_idx,:],axis=1)
select_idx =  val2_idx[y_pred==y_true_noisy]# integer index
print('Noisy Validation Accuracy: %.4f'%(len(select_idx)/len(y_pred)))
print('Label Precision: %.4f'%(np.sum(clean_index[select_idx])/len(select_idx)))
print('Label Recall: %.4f'%(np.sum(clean_index[select_idx])/np.sum(clean_index[val2_idx])))

y_test_pred = np.argmax(parallel_model.predict(x_test), axis=1)
np.save(save_dir+'/y_pred_'+noise_pattern+str(noise_ratio)+'.npy',y_pred)
np.save(save_dir+'/y_true_'+noise_pattern+str(noise_ratio)+'.npy',np.argmax(y_train[val2_idx,:], axis=1))
np.save(save_dir+'/y_test_pred_'+noise_pattern+str(noise_ratio)+'.npy',y_test_pred)
np.save(save_dir+'/y_test_true_'+noise_pattern+str(noise_ratio)+'.npy',np.argmax(y_test, axis=1))
print('Noise ratio: %.2f'%noise_ratio)

model.save(filepath)
