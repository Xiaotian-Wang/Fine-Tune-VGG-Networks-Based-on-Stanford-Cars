import numpy as np
import tensorflow as tf
import random
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers, initializers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import load_model
from fine_tune_model import get_data, encode
# -----------------------------


model = load_model('d:/VGG19_3FC_2BN_FINAL_AUG1DATA.h5')

X_test,y_test = get_data('test')
y_test = encode(y_test)
X_test = np.rollaxis(X_test,1,4)

a = model.evaluate(X_test, y_test,verbose=1)
print(a)
