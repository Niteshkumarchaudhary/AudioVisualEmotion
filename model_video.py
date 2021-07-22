"""
Nitesh Kumar Chaudhary
niteshku001@e.ntu.edu.sg
"""
from keras.models import Model
from keras import layers
from keras.models import Sequential
from keras.layers import Input, LSTM, BatchNormalization, Conv2D, UpSampling2D, GlobalAveragePooling2D, GRU, concatenate
from keras.layers.core import Flatten, Dense, Dropout, Activation, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import regularizers

import sys
sys.path
sys.path.append('/media/nitesh/Windows/NCSwork/RaspberryPiTrain/AudioSetVGGishTensorflowClassification/VGGish')

#from vggish import VGGish


def build_small_model(category=8):
  sound_model = VGGish(include_top=True, load_weights=True, input_shape=(45, 2048, 1))
  #sound_model.trainable = False
    
  for layer in sound_model.layers:
      layer.trainable = False
  print("Input : ", sound_model.input)
  
  x = sound_model.get_layer(name="conv4/conv4_2").output
  print("FC: ", x.shape)
  #x = sound_model.get_layer(name="conv4/conv4_1").output
  #x = GlobalAveragePooling2D()(x)
  #inputs = layers.Input(shape=(192, 64, 1))
  #x = layers.Flatten()(inputs)
  #x = layers.Dense(128, activation='relu', name='FC1')(x)
  #x = layers.Dense(512, activation='elu', name='FC2')(x)
  #x = layers.Dropout(0.4)(x)
  x = layers.Dense(1024, activation='relu', name='FC3')(x)
  x = layers.Dense(category, name='logits')(x)
  preds = layers.Activation('softmax', name='Softmax')(x)
  model = Model(inputs=sound_model.inputs, outputs=preds)
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
  #model.summary()
  return model

#small_model = build_small_model()
#small_model.summary()

def VGG_LSTM_model_v1(inputDim):

    sound_model = VGGish(include_top=True, load_weights=True, input_shape=(96, 64, 1))
  #sound_model.trainable = False

    for layer in sound_model.layers:
        layer.trainable = False
    print("Input : ", sound_model.input)

    x = sound_model.get_layer(name="conv4/conv4_2").output
    print("FC: ", x.shape)
    input_feat = Input(shape=inputDim)
    x = LSTM(256, return_sequences=True)(input_feat)
    x = LSTM(256, return_sequences=True)(input_feat)
    x = LSTM(256, return_sequences=False)(x)
    x = Dense(256, activation = 'elu')(x)
    x = Dense(8, activation = 'softmax')(x)
    Lmodel = Model(input_feat, x)
    Lmodel.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

    return Lmodel

#V_LSTM = VGG_LSTM_model_v1((96, 64))
#V_LSTM.summary()

def LSTM_model(inputDim):

    input_feat = Input(shape=inputDim)
    x = LSTM(2048, return_sequences=True)(input_feat)
    x = LSTM(1024, return_sequences=False)(x)
    x = Dense(1024, activation = 'relu')(x)
    x = Dense(8, activation = 'softmax')(x)
    Lmodel = Model(input_feat, x)
    Lmodel.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

    return Lmodel





def LSTM_model_v1(inputDim):

    input_feat = Input(shape=inputDim)
    x = LSTM(1024, return_sequences=True)(input_feat)
    #x = LSTM(1024, return_sequences=False)(x)
    x = LSTM(1024, return_sequences=False)(x)
    x = Dense(1024, activation = 'elu')(x)
    x = Dense(8, activation = 'softmax')(x)
    Lmodel = Model(input_feat, x)
    Lmodel.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

    return Lmodel

def lstm_new(inputDim):

    input_feat = Input(shape=inputDim)
    x = LSTM(2048, return_sequences=True)(input_feat)
    #x = LSTM(1024, return_sequences=False)(x)
    x = LSTM(2048, return_sequences=False)(x)
    x = Dense(1024, activation = 'elu')(x)
    x = Dense(512, activation='elu')(x)
    x = Dense(8, activation = 'softmax')(x)
    Lmodel = Model(input_feat, x)
    Lmodel.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

    return Lmodel



def CNN_Model(weights_path=None, category=11):
  model = Sequential()
  model.add(ZeroPadding2D((1, 1), input_shape=(512, 12, 1)))
  #model.add(ZeroPadding2D((1, 1), input_shape=(384, 64, 1)))
  model.add(Convolution2D(32, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(32, 3, 3, activation='elu'))
  #model.add(MaxPooling2D((1, 2), strides=(2, 2)))

  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(16, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(16, 3, 3, activation='relu'))
  model.add(MaxPooling2D((1, 2), strides=(2, 2)))

  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(16, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(16, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(32, 3, 3, activation='relu'))
  model.add(MaxPooling2D((1, 2), strides=(2, 2)))

  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(32, 3, 3, activation='elu'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(32, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(32, 3, 3, activation='relu'))
  model.add(MaxPooling2D((1, 1), strides=(1, 1)))

  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(64, 3, 3, activation='elu'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(64, 3, 3, activation='relu'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(64, 3, 3, activation='relu'))
  model.add(MaxPooling2D((1, 2), strides=(2, 2)))

  '''
  # Adding RNN Layer here
  model.add(Reshape((model.output_shape[1]*model.output_shape[2], model.output_shape[3])))
  model.add(GRU(128, activation='relu',return_sequences=True))
  model.add(Dropout(0.1))
  model.add(GRU(64, activation='relu',return_sequences=True))
  #model.add(Flatten())
  '''

  model.add(Flatten())
  #model.add(Dense(512, activation='relu'))
  #model.add(Dropout(0.3))
  model.add(Dense(1024, activation='relu'))
  model.add(Dropout(0.2))

  model.add(Dense(512, activation='relu'))
  model.add(Dense(category, name='logits'))
  model.add(Activation('softmax'))
  model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  if weights_path:
    model.load_weights(weights_path)

  #model.summary()
  return model


