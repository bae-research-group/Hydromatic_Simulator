import numpy as np
import tensorflow as tf

from Hydromatic_Simulator.utils.config import config


def preprocess_binary_input(binary_str):
    return np.array([int(bit) for bit in binary_str[:config['structure-dim']]])

def run_inference(models, binary_str):
    design_input = preprocess_binary_input(binary_str)

    shape_seq_pred = predict_(models, design_input)

    return shape_seq_pred.squeeze()

def predict_(models, design_input):

  init_nodal_pos = config['init-pos']
  nodal_pos = []

  for i, model_ in enumerate(models):
    out = model_.recursive_generate(
        tf.convert_to_tensor([design_input]), tf.convert_to_tensor([init_nodal_pos[i]]), training=False)
    nodal_pos.append(out)

  return np.stack(nodal_pos, axis=1) 
