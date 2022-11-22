# required imports
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Bidirectional, LSTM, Reshape
from sklearn.model_selection import train_test_split
from tensorflow.keras import activations
from sklearn.utils import shuffle
from keras import backend as K
from keras.utils import np_utils
import pickle
from data_encoding import *
import pickle



def create_model(checkpoint):
    lrelu = lambda x: tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    # Define  Embedding matrix for Protein and NA sequence
    protein_input = keras.Input(shape=(1000, 20, 1), name="pinp")
    dr_input = keras.Input(shape=(75, 5, 1), name="drinp")

    # Model architecture Starts
    p1 = layers.Conv2D(48, (4,20), activation="relu",kernel_regularizer=keras.regularizers.l1(2e-4))(protein_input)
    p1 = layers.Conv2D(48, (8,1), activation="relu",kernel_regularizer=keras.regularizers.l1(2e-4))(p1)
    p1 = layers.Conv2D(48, (12,1), activation="relu",kernel_regularizer=keras.regularizers.l1(2e-4))(p1)

    p2 = layers.Conv2D(48, (8,20), activation="relu",kernel_regularizer=keras.regularizers.l1(2e-4))(protein_input)
    p2 = layers.Conv2D(48, (12,1), activation="relu",kernel_regularizer=keras.regularizers.l1(2e-4))(p2)
    p2 = layers.Conv2D(48, (4,1), activation="relu",kernel_regularizer=keras.regularizers.l1(2e-4))(p2)

    p3 = layers.Conv2D(48, (12,20), activation="relu",kernel_regularizer=keras.regularizers.l1(2e-4))(protein_input)
    p3 = layers.Conv2D(48, (4,1), activation="relu",kernel_regularizer=keras.regularizers.l1(2e-4))(p3)
    p3 = layers.Conv2D(48, (8,1), activation="relu",kernel_regularizer=keras.regularizers.l1(2e-4))(p3)

    d1 = layers.Conv2D(24, (2,5), activation="relu",kernel_regularizer=keras.regularizers.l1(2e-4))(dr_input)
    d1 = layers.Conv2D(24, (4,1), activation="relu",kernel_regularizer=keras.regularizers.l1(2e-4))(d1)
    d1 = layers.Conv2D(24, (8,1), activation="relu",kernel_regularizer=keras.regularizers.l1(2e-4))(d1)

    d2 = layers.Conv2D(24, (4,5), activation="relu",kernel_regularizer=keras.regularizers.l1(2e-4))(dr_input)
    d2 = layers.Conv2D(24, (8,1), activation="relu",kernel_regularizer=keras.regularizers.l1(2e-4))(d2)
    d2 = layers.Conv2D(24, (2,1), activation="relu",kernel_regularizer=keras.regularizers.l1(2e-4))(d2)

    d3 = layers.Conv2D(24, (8,5), activation="relu",kernel_regularizer=keras.regularizers.l1(2e-4))(dr_input)
    d3 = layers.Conv2D(24, (2,1), activation="relu",kernel_regularizer=keras.regularizers.l1(2e-4))(d3)
    d3 = layers.Conv2D(24, (4,1), activation="relu",kernel_regularizer=keras.regularizers.l1(2e-4))(d3)

    p4=layers.concatenate([p1, p2, p3])
    p5=layers.MaxPooling2D(pool_size=(979,1))(p4)
    p5=layers.Flatten()(p5)
    #p6=layers.AveragePooling2D(pool_size=(979,1))(p4)
    #p6=layers.Flatten()(p6)

    d4=layers.concatenate([d1, d2, d3])
    d5=layers.MaxPooling2D(pool_size=(64,1))(d4)
    d5=layers.Flatten()(d5)
    #d6=layers.AveragePooling2D(pool_size=(64,1))(d4)
    #d6=layers.Flatten()(d6)

    #tpnn=layers.Dense(6, activation="relu")(tp_input)
    pdnn0=layers.concatenate([p5,d5])
    pdnn=layers.Dense(216, activation="relu")(pdnn0)
    pdnn=layers.Dropout(0.25, input_shape=(216,))(pdnn)
    pdnn=layers.Dense(216, activation="relu")(pdnn)
    pdnn=layers.Dropout(0.25, input_shape=(216,))(pdnn)
    pdnn=layers.Add()([pdnn,pdnn0])
    pdnn=layers.Dense(1,kernel_regularizer=keras.regularizers.l1(2e-4))(pdnn)


    # Define Model
    model = keras.Model(
        inputs=[protein_input, dr_input],
        outputs=[pdnn])

    #Define Roont means square error
    rmse=tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error", dtype=None)

    # Model Compilation
    model.compile(optimizer='adam',loss='mse',metrics=[rmse,'mae'])

    #model.summary()
    # history = model.fit({"pinp":X1_train1, "drinp":X1_train2},y1_train1, epochs=500, batch_size=512, validation_data=({"pinp":X1_test1, "drinp":X1_test2},y1_test1), verbose=1,shuffle=True, callbacks=[chk_callback])


    # Load Model check Point files
    # Predict the y  for the input X(s)
    model.load_weights(checkpoint)

    return model
