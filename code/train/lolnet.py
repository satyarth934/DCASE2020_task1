import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, DepthwiseConv2D, UpSampling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
# from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, Input, concatenate, Lambda
# from tensorflow.keras.layers import  Add, Reshape, DepthwiseConv2D, Dropout
# from tensorflow.keras.utils import plot_model
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def encoder_decoder(code_dim = 10):
    inputs = Input(shape = (X_train.shape[1],))
    code = Dense(50, activation= 'relu')(inputs)
    code = Dense(50, activation = 'relu')(code)
    code = Dense(code_dim, activation = 'relu')(code)
    
    outputs = Dense(50, activation = 'relu')(code)
    outputs = Dense(50, activation = 'relu')(outputs)
    outputs = Dense(X_train.shape[1], activation = 'sigmoid')(outputs)
    
    auto_encoder = Model(inputs = inputs, outputs = outputs)
    auto_encoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')
    
    return auto_encoder


def modelConvAutoencoder(input_shape, num_classes):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    tf.random.set_seed(54)
    inputs = Input(shape=input_shape)
    
    # Encoder
    encode = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same')(inputs)
    encode = BatchNormalization(axis=channel_axis)(encode)
    encode = Activation('relu')(encode)
    encode = MaxPooling2D(pool_size=(2,2), padding='same')(encode)

    encode = Conv2D(filters=48, kernel_size=(3,3), strides=(1,1), padding='same')(encode)
    encode = BatchNormalization(axis=channel_axis)(encode)
    encode = Activation('relu')(encode)
    encode = MaxPooling2D(pool_size=(2,2), padding='same')(encode)
    
    encode = Conv2D(filters=56, kernel_size=(3,3), strides=(1,1), padding='same')(encode)
    encode = BatchNormalization(axis=channel_axis)(encode)
    encode = Activation('relu')(encode)
    encode = MaxPooling2D(pool_size=(2,2), padding='same')(encode)
    
    # # Decoder
    # decode = DepthwiseConv2D(kernel_size=(3,3), strides=(1, 1), depth_multiplier=1, padding='same')(encode)
    # decode = BatchNormalization(axis=channel_axis)(decode)
    # decode = Activation('relu')(decode)
    # # decode = Conv2D(filters=5, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(encode)
    # decode = UpSampling2D(size=(2,2))(decode)
    
    # decode = DepthwiseConv2D(kernel_size=(3,3), strides=(1, 1), depth_multiplier=1, padding='same')(encode)
    # decode = BatchNormalization(axis=channel_axis)(decode)
    # decode = Activation('relu')(decode)
    # # decode = Conv2D(filters=10, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(decode)
    # decode = UpSampling2D(size=(2,2))(decode)    
    
    outputs = Conv2D(filters=num_classes, kernel_size=(3,3), activation='relu', padding='same')(encode)

    outputs = BatchNormalization(center=False, scale=False)(outputs)
    outputs = GlobalMaxPooling2D()(outputs)
    outputs = Activation('softmax')(outputs)

    auto_encoder = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return auto_encoder
