import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from keras import layers, models, metrics

from Hydromatic_Simulator.utils.config import config

# ----------- Encoding layer + LSTM (timestep: -1) -----------
class EncodingModule(keras.Model):

    def __init__(self):
        super().__init__()
        
        self.embed_design = layers.Dense(config['feature-dim'], activation='relu')
        self.lstm_cell = layers.LSTMCell(config['lstm-dim'])

    def call(self, inputs, hidden_states, training):
        if hidden_states == None:
            batch_size = tf.shape(inputs)[0]
            hidden_states = tf.zeros([2, batch_size, config['lstm-dim']], dtype=tf.float32)

        feature_vector = self.embed_design(inputs, training=training)
        _, new_hidden_states = self.lstm_cell(
            inputs=feature_vector, states=hidden_states, training=training)
        return new_hidden_states

# ----------- LSTM (timestep: 0 to T-1) -----------
class DecodingModule(keras.Model):

    def __init__(self):
        super().__init__()
        # Position embedder and LSTM
        self.embed_pos = layers.Dense(config['feature-dim'], activation='relu')
        self.lstm_cell = layers.LSTMCell(config['lstm-dim'])

        # Output layers
        self.dense1 = layers.Dense(config['dense-dim'], activation='relu')
        self.dense2 = layers.Dense(config['dense-dim'], activation='relu')
        self.dense3 = layers.Dense(config['nodal-dim'], activation='linear')

    def call(self, inputs, hidden_states, training):

        x = self.embed_pos(inputs, training=training)
        x, new_hidden_states = self.lstm_cell(
            x, states=hidden_states, training=training)

        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)
        out = self.dense3(x, training=training)
        return out, new_hidden_states
