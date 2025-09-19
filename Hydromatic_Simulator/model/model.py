import numpy as np
import tensorflow as tf

from Hydromatic_Simulator.utils.config import config
from Hydromatic_Simulator.model.module import EncodingModule, DecodingModule 

class GeneratorModel(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.encoder = EncodingModule()
        self.decoder = DecodingModule()

    def call_encoder(self, inputs, training=False):
        return self.encoder(inputs, None, training=training)

    def call_decoder(self, inputs, hidden_states, training=False):
        return self.decoder(inputs, hidden_states, training=training)

    def round_differentiable(self, x, decimal_pt):
      factor = 10.0 ** decimal_pt
      x_scaled = x * factor
      x_nondiff = tf.round(x_scaled) / factor
      return x + tf.stop_gradient(x_nondiff - x)
    
    def teacher_forcing_generate(self, design_input, nodal_input_seq, training=False):

        nodal_output_seq = []

        # ----------- Encoding layer + LSTM (timestep: -1) -----------
        new_hidden_states = self.call_encoder(
            design_input, training=training
        )
        # ----------- LSTM (timestep: 0 to T-1) -----------
        for i in range(config['num-timesteps']):
            nodal_input = nodal_input_seq[:,i,:]
            hidden_states = new_hidden_states

            next_pos, new_hidden_states = self.call_decoder(
                nodal_input, hidden_states, training=training
            )
            nodal_output_seq.append(next_pos)

        nodal_output_seq = tf.transpose(tf.stack(nodal_output_seq), perm=[1,0,2])  # shape: (TOTAL_TIMESTAMP_NO, -1, shape_dim) -> (-1, TOTAL_TIMESTAMP_NO, shape_dim)
        return self.round_differentiable(nodal_output_seq, config['decimal-pt'])


    def recursive_generate(self, design_input, nodal_input, training=False):
        nodal_output_seq = []

        # ----------- Encoding layer + LSTM (timestep: -1) -----------
        new_hidden_states = self.call_encoder(
            design_input, training=training
        )

        # ----------- LSTM (timestep: 0 to T-1) --------------------
        #-------- Input initial nodal point (w/o Teacher forcing)
        hidden_states = new_hidden_states

        next_pos, new_hidden_states = self.call_decoder(
            nodal_input, hidden_states, training=training
        )
        nodal_output_seq.append(next_pos)

        for i in range(1,config['num-timesteps']):
            nodal_input = next_pos
            hidden_states = new_hidden_states

            next_pos, new_hidden_states = self.call_decoder(
                nodal_input, hidden_states, training=training
            )
            nodal_output_seq.append(next_pos)

        nodal_output_seq = tf.transpose(tf.stack(nodal_output_seq), perm=[1,0,2])  ########## shape: (TOTAL_TIMESTAMP_NO, -1, shape_dim) -> (-1, TOTAL_TIMESTAMP_NO, shape_dim)
        return self.round_differentiable(nodal_output_seq, config['decimal-pt'])


    def predict_step(self, batch_group):
        batch_design_idx, batch_design_input, batch_nodal_true = batch_group
        batch_nodal_pred_infer = self.recursive_generate(batch_design_input, batch_nodal_true[:,0,:], training=False)
        return batch_nodal_pred_infer
    
