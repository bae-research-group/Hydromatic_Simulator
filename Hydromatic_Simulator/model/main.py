import os
import numpy as np
import pickle

import tensorflow as tf

from Hydromatic_Simulator.model.model import GeneratorModel
from Hydromatic_Simulator.utils.config import config
from Hydromatic_Simulator.model.module import EncodingModule, DecodingModule

def load_trained_model(data_path=os.path.join(".", "Hydromatic_Simulator", "dataset"),
                       weights_path=os.path.join(".", "Hydromatic_Simulator", "model", "weights")):

  test_data = import_data(data_path)

  models = []
  for i in range(config['num-coord']):

    test_ = test_data[i]
    
    test_dataset = make_dataset(
          config['batch-size'], test_['idx'], test_['design'], test_['X'], test_['Y']
    )
    

    model_ = GeneratorModel()
    _ = model_.predict(test_dataset, verbose=0)

    model_.encoder.load_weights(
        os.path.join(weights_path,'encoder_{}coords_{}.weights.h5'.format(config['num-coord'], str(i))))
    model_.decoder.load_weights(
        os.path.join(weights_path,'decoder_{}coords_{}.weights.h5'.format(config['num-coord'], str(i))))

    models.append(model_)

  return models


def import_data(data_path):
    with open(os.path.join(data_path, 'test_data.pkl'), 'rb') as f:
        test_data = pickle.load(f)
        
    return test_data

def make_dataset(batch_size, design_idx, design_input, true_nodal_x, true_nodal_y):
    tf1 = tf.convert_to_tensor(design_idx)
    tf2 = tf.convert_to_tensor(design_input)
    tf3 = tf.stack(
        [tf.convert_to_tensor(true_nodal_x), tf.convert_to_tensor(true_nodal_y)], axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices((tf1, tf2, tf3))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
