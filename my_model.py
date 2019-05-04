
import keras
from keras.layers import Input,Conv2D,Dense,BatchNormalization,Activation,add,GlobalAveragePooling2D,GlobalMaxPooling2D
from keras.layers import AveragePooling2D, Flatten
from keras.models import Model
from keras.regularizers import l2

# ResNet building block of two layers
def building_block(X, filter_size, filters, stride=1):

    # Save the input value for shortcut
    X_shortcut = X

    # Reshape shortcut for later adding if dimensions change
    #if stride > 1:
    if stride > 1 or filters>X.get_shape()[-1]:

        X_shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same', kernel_regularizer=l2(1e-4))(X_shortcut)
        X_shortcut = BatchNormalization(axis=3)(X_shortcut)

    # First layer of the block
    X = Conv2D(filters, kernel_size = filter_size, strides=stride, padding='same', kernel_regularizer=l2(1e-4))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Second layer of the block
    X = Conv2D(filters, kernel_size = filter_size, strides=(1, 1), padding='same', kernel_regularizer=l2(1e-4))(X)
    X = BatchNormalization(axis=3)(X)
    X = add([X, X_shortcut])  # Add shortcut value to main path
    X = Activation('relu')(X)

    return X


# Full model
def create_model(input_shape, classes, name, architecture='ResNet32'):

    # Define the input
    X_input = Input(input_shape)

    if architecture == 'ResNet32':
        # Stage 1
        X = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same')(X_input)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
    
        # Stage 2
        X = building_block(X, filter_size=3, filters=16, stride=1)
        X = building_block(X, filter_size=3, filters=16, stride=1)
        X = building_block(X, filter_size=3, filters=16, stride=1)
        X = building_block(X, filter_size=3, filters=16, stride=1)
        X = building_block(X, filter_size=3, filters=16, stride=1)
    
        # Stage 3
        X = building_block(X, filter_size=3, filters=32, stride=2)  # dimensions change (stride=2)
        X = building_block(X, filter_size=3, filters=32, stride=1)
        X = building_block(X, filter_size=3, filters=32, stride=1)
        X = building_block(X, filter_size=3, filters=32, stride=1)
        X = building_block(X, filter_size=3, filters=32, stride=1)
    
        # Stage 4
        X = building_block(X, filter_size=3, filters=64, stride=2)  # dimensions change (stride=2)
        X = building_block(X, filter_size=3, filters=64, stride=1)
        X = building_block(X, filter_size=3, filters=64, stride=1)
        X = building_block(X, filter_size=3, filters=64, stride=1)
        X = building_block(X, filter_size=3, filters=64, stride=1)
    
        # Average pooling and output layer
        X = GlobalAveragePooling2D()(X) 
        X = Dense(classes, activation='softmax')(X)
        
    elif architecture == 'WRN-28-10':
        
        # Stage 1
        X = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same')(X_input)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)

        # Stage 2
        for i in range(9):
            X = building_block(X, filter_size=3, filters=160, stride=1)

        # Stage 3
        X = building_block(X, filter_size=3, filters=320, stride=2)  # dimensions change (stride=2)
        for i in range(1,9):
            X = building_block(X, filter_size=3, filters=320, stride=1)

        # Stage 4
        X = building_block(X, filter_size=3, filters=640, stride=2)  # dimensions change (stride=2)
        for i in range(1,9):
            X = building_block(X, filter_size=3, filters=640, stride=1)        
        
        # Average pooling and output layer
        X = GlobalAveragePooling2D()(X)
        X = Dense(classes, activation='softmax')(X)
    
    elif architecture == 'ResNet110':
        return resnet_v2(input_shape, depth=110, num_classes=classes)
    
    elif architecture == 'ResNet164':
        return resnet_v2(input_shape, depth=164, num_classes=classes)
        
    elif architecture == 'CNN9':        
        X = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same', kernel_regularizer=l2(1e-4))(X_input)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        X = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same', kernel_regularizer=l2(1e-4))(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        X = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same', kernel_regularizer=l2(1e-4))(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        X = GlobalMaxPooling2D()(X)
        
        X = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same', kernel_regularizer=l2(1e-4))(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        X = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same', kernel_regularizer=l2(1e-4))(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        X = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same', kernel_regularizer=l2(1e-4))(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        X = GlobalMaxPooling2D()(X)
        
        X = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same', kernel_regularizer=l2(1e-4))(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        X = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same', kernel_regularizer=l2(1e-4))(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)
        X = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same', kernel_regularizer=l2(1e-4))(X)
        X = BatchNormalization(axis=3)(X)
        X = Activation('relu')(X)

        # Average pooling and output layer
        X = GlobalAveragePooling2D()(X)
        X = Dense(classes, activation='softmax')(X)
        

    # Create model
    model = Model(inputs=X_input, outputs=X, name=name)

    return model

# borrow from https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal'#,
                  #kernel_regularizer=l2(1e-4)
                  )


    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
