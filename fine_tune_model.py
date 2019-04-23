import numpy as np
import random
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.applications.inception_v3 import InceptionV3

from keras import optimizers, initializers

# ------------------------------------------------------------------------------


def get_data(flag):
    if flag == 'train':
        X_train = np.load('X_train.npy')
        X_train = X_train/255
        y_train = np.load('y_train.npy')
        print('Shape of X_train: '+str(X_train.shape))
        print('Shape of y_train: '+str(y_train.shape))
    elif flag == 'test':
        X_train = np.load('X_test.npy')
        X_train = X_train/255
        y_train = np.load('y_test.npy')
        print('Shape of X_test: '+str(X_train.shape))
        print('Shape of y_test: '+str(y_train.shape))
    elif flag == 'train_aug':
        X_train = np.load('X_train_aug.npy')
        X_train = X_train / 255
        y_train = np.load('y_train_aug.npy')
        print('Shape of X_train_aug: ' + str(X_train.shape))
        print('Shape of y_train_aug: ' + str(y_train.shape))
    elif flag == 'train_aug2':
        X_train = np.load('d:/X_train_aug2.npy')
        X_train = X_train / 255
        y_train = np.load('d:/y_train_aug2.npy')
        print('Shape of X_train_aug: ' + str(X_train.shape))
        print('Shape of y_train_aug: ' + str(y_train.shape))
    if flag == 'crop':
        X_train = np.load('X_crop.npy')
        X_train = X_train/255
        y_train = np.load('y_train.npy')
        print('Shape of X_crop: '+str(X_train.shape))
        print('Shape of y_train: '+str(y_train.shape))
    if flag == 'test_crop':
        X_train = np.load('X_test_crop.npy')
        X_train = X_train/255
        y_train = np.load('y_train.npy')
        print('Shape of X_crop: '+str(X_train.shape))
        print('Shape of y_train: '+str(y_train.shape))
    print('Data loaded!')
    return X_train, y_train


def freeze_layers(model):
    for layer in model.layers:
        layer.trainable = False


def model_define(modeltype, inputshape):
    if modeltype == 'VGG16':
        model = VGG16(include_top=False, weights=None, input_tensor=None, input_shape=inputshape, pooling=None)
        freeze_layers(model)
        model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
        # model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5')
        # model.layers[-1].kernel_initializer = initializers.glorot_normal()
        # model.layers.append(Dense(4096, activation = 'relu'))
        # model.layers.append(Dense(4096, activation='relu'))
        # model.layers.append(Dense(196, activation='softmax'))
        print('Model: VGG 16, weights loaded!')
    elif modeltype == 'InceptionV3':
        model = InceptionV3(include_top=False, weights=None,input_shape=inputShape)
        freeze_layers(model)
        model.load_weights('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
        print('Model:InceptionV3, weights loaded!')
    elif modeltype == 'VGG19':
        model = VGG19(include_top=False, weights=None, input_tensor=None, input_shape=inputshape, pooling=None)
        freeze_layers(model)
        model.load_weights('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
        print('Model: VGG 19, weights loaded!')
    else:
        pass

    # model.layers[-1].trainable = True
    # model.layers[-1] = Dense(196,activation = 'softmax')
    return model


def fine_tune(basemodel, method):
    # Some adjustments can be made in this function
    if method == 0:
        x = basemodel.output
        x = Flatten()(x)
        # x = BatchNormalization(axis=-1, epsilon=0.001, center=True, scale=True)(x)
        x = Dense(4096, activation='relu')(x)
        # x = Dropout(0.2)(x)
        x = Dense(4096, activation='relu')(x)
        # x = Dropout(0.2)(x)
        # x = BatchNormalization(axis=-1, epsilon=0.001, center=True, scale=True)(x)
        predictions = Dense(196, activation='softmax')(x)
        model = Model(inputs=basemodel.input, outputs=predictions)
        print('VGG fine tune, success!')
        return model
    elif method == 1:
        x = basemodel.output
        x = Flatten()(x)
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.01, center=True,scale=True)(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(196, activation='softmax')(x)
        model = Model(inputs=basemodel.input, outputs=predictions)
        print('Inception V3 fine tune, success!')
        return model
    else:
        return basemodel


def make_batch(X,y, batchsize):
    n = len(y)
    slice = random.sample(range(n),batchsize)
    return X[slice],y[slice]


def encode(y):
    temp = np.zeros((len(y), 196))
    # print(temp.shape)
    # print(range(len(y) - 1))
    for count in range(len(y) - 1):
        temp[count][y[count] - 1] = 1

    # print(temp.shape)
    return temp


# ---------------------------------------------------------------------------------------
inputShape = (224, 224, 3)
learningRate = 0.05
modelType = 'VGG19'
# A valid value for 3 Dense FC layers is 0.005
# A valid learning rate for 3fc with BN layers is 0.05

batchSize = 10
epochs = 10
# One valid number of epochs of 3 FC layers is 25
# print(model.layers[16].trainable)
# ---------------------------------------------------------------------------------------

sgd = optimizers.SGD(lr=learningRate, decay=1e-4, momentum=0.5, nesterov=False)
adam = optimizers.Adam()

# ----------------------------------------------------------------------------------------

if __name__ == '__main__':
    baseModel = model_define(modelType, inputShape)
    model = fine_tune(baseModel, 0)
    # THE DEFAULT VALUE 0 HERE IS CORRESPONDING TO THE VGG NETWORK

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    print('Model compiled!')

    # Load the data
    X_train, y_train = get_data('train')
    y_train = encode(y_train)
    X_train = np.rollaxis(X_train, 1, 4)

    print(model.summary())
    history = model.fit(x=X_train, y=y_train, batch_size=batchSize, epochs=epochs, validation_split=0.1, verbose=2)
    np.save('Model_History.npy', history.history)
    model.save('d:/VGG19_3FC_2BN_FINAL_CROP.h5')
