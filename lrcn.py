import numpy as np
import tensorflow as tf
from keras import layers, models, applications
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Lambda
import cv2
from keras import backend as K
from keras.utils import plot_model

# set learning phase to 0
K.set_learning_phase(0)


video = layers.Input(shape=(None, 299,299,3),name='video_input')
cnn = applications.inception_v3.InceptionV3(
    weights='imagenet',
    include_top=False,
    pooling='avg')
cnn.trainable = False
# wrap cnn into Lambda and pass it into TimeDistributed
encoded_frame = layers.TimeDistributed(Lambda(lambda x: cnn(x)))(video)
encoded_vid = layers.LSTM(256)(encoded_frame)
outputs = layers.Dense(128, activation='relu')(encoded_vid)
model = models.Model(inputs=[video],outputs=outputs)
model.compile(optimizer='adam',loss='mean_squared_logarithmic_error')

# plot_model(model, to_file='model.png')

# model.summary()

# Generate random targets
y = np.random.random(size=(128,)) 
y = np.reshape(y,(-1,128))
frame_sequence = np.random.random(size=(1, 48, 299, 299, 3))
model.fit(x=frame_sequence, y=y, validation_split=0.0,shuffle=False, batch_size=1, epochs=5)

