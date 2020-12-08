import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

expno = 1 + max([0] + [int(d.split("_")[-1]) for d in os.listdir(".") if "exp_custom_expno_" in d])

# Init wandb
import wandb
from wandb.keras import WandbCallback
wandb.init(config={"Dataset": "DCASE Task 1b",
          "exp_no": expno},
      project="cmsc727-acoustic-scene-detector", 
      notes="Smaller modified mobnet model on smaller dataset. Added step_decay learning rate scheduler and early stopping for lr.")

import numpy as np
import keras
import tensorflow
from keras.optimizers import SGD

import sys
sys.path.append("..")
from utils import *
from funcs import *

from mobnet import model_mobnet
from training_functions import *

# from tensorflow import ConfigProto
# from tensorflow import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

data_path = '../../task1b/data_2020/'
train_csv = data_path + 'evaluation_setup/fold1_train.csv'
val_csv = data_path + 'evaluation_setup/fold1_evaluate.csv'
feat_path = '../../data/features/logmel128_scaled/'
experiments = f'exp_mobnet_expno_{wandb.config.exp_no}'

if not os.path.exists(experiments):
    os.makedirs(experiments)


def step_decay(epoch):
   initial_lrate = 0.1
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate


# random sample data, to keep all three classes have similar number of training samples
total_csv = balance_class_data(train_csv, experiments)

wandb.config.num_audio_channels  = num_audio_channels = 2
wandb.config.num_freq_bin        = num_freq_bin = 128
wandb.config.num_time_bin        = num_time_bin = 461
wandb.config.num_classes         = num_classes = 3
wandb.config.max_lr              = max_lr = 0.1
wandb.config.batch_size          = batch_size = 16
wandb.config.num_epochs          = num_epochs = 500
wandb.config.mixup_alpha         = mixup_alpha = 0.4
wandb.config.sample_num          = sample_num = 1000
# wandb.config.sample_num          = sample_num = len(open(train_csv, 'r').readlines()) - 1


data_val, y_val = load_data_2020(feat_path, val_csv, num_freq_bin, 'logmel')
y_val = keras.utils.to_categorical(y_val, num_classes)

model = model_mobnet(num_classes, input_shape=[num_freq_bin, num_time_bin, 3*num_audio_channels], num_filters=24, wd=1e-3)

model.compile(loss='categorical_crossentropy',
              optimizer = SGD(lr=max_lr,decay=0, momentum=0.9, nesterov=False),
              metrics=['accuracy']) #ori

model.summary()

callback_earlystop = EarlyStopping(monitor='loss', patience=5)
lr_callback = LearningRateScheduler(step_decay, verbose=1)
# lr_scheduler = LR_WarmRestart(nbatch=np.ceil(sample_num/batch_size), Tmult=2,
#                               initial_lr=max_lr, min_lr=max_lr*1e-4,
#                               epochs_restart = [3.0, 7.0, 15.0, 31.0, 63.0,127.0,255.0]) 
save_path = experiments + "/model-{epoch:02d}-{val_accuracy:.4f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(save_path, monitor='val_accuracy', verbose=1, save_best_only=False, mode='max')
# callbacks = [lr_scheduler, checkpoint, WandbCallback()]
callbacks = [lr_callback, checkpoint, WandbCallback()]

train_data_generator = Generator_balanceclass_timefreqmask_nocropping_splitted(feat_path, train_csv, total_csv, experiments, num_freq_bin, 
                              batch_size=batch_size,
                              alpha=mixup_alpha, splitted_num=4, sample_num=sample_num)()

history = model.fit_generator(train_data_generator,
                              validation_data=(data_val, y_val),
                              epochs=num_epochs, 
                              verbose=1, 
                              workers=4,
                              max_queue_size = 100,
                              callbacks=callbacks,
                              steps_per_epoch=np.ceil(sample_num/batch_size)
                              ) 

