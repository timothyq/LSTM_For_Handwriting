from typing import Any, Callable, Dict, Union
from flax.core.scope import CollectionFilter, DenyList, FrozenVariableDict
from flax.linen.module import KeyArray, Module, RNGSequences
import jax
import jax.numpy as jnp
import flax.linen as nn


class RMSNormedLSTMCell(nn.Module):
    features: int
    name: str = None

    def setup(self):
        self.lstm_cell = nn.LSTMCell(
            features=self.features, name=f"{self.name}_lstm")
        self.rn = nn.RMSNorm(name=f"{self.name}_rms_norm")

    # @nn.compact
    def __call__(self, carry, x):
        new_carry_lstm, x = self.lstm_cell(carry, x)
        x = self.rn(x)
        return new_carry_lstm, x

    def init_carry(self, rng, input_shape):
        return self.lstm_cell.initial_carry(rng, (input_shape[1], self.features))


class MultiRMSNormedLSTMCell(nn.Module):
    input_features: int
    hidden_size: int
    num_layers: int

    @nn.compact
    def __call__(self, carry, x):
        new_carrys = []
        outputs = []

        cell = RMSNormedLSTMCell(
            features=self.hidden_size, name=f'lstm_layer_0')
        new_carry, output = cell(carry[0], x)
        new_carrys.append(new_carry)
        outputs.append(output)
        for i in range(1, self.num_layers):
            input_x = jnp.concatenate([output, x], axis=-1)
            cell = RMSNormedLSTMCell(
                features=self.hidden_size, name=f'lstm_layer_{i}')
            new_carry, output = cell(carry[i], input_x)
            new_carrys.append(new_carry)
            outputs.append(output)

        outputs = jnp.concatenate(outputs, axis=-1)
        return new_carrys, outputs


class RMSNormLSTM(nn.Module):
    num_layers: int
    hidden_size: int
    input_features: int
    component_k: int

    @nn.compact
    def __call__(self, x):
        init_carry = [(jax.numpy.zeros((x.shape[0], self.hidden_size)),
                       jax.numpy.zeros((x.shape[0], self.hidden_size)))
                      for _ in range(self.num_layers)]

        multi_lstm_cell = MultiRMSNormedLSTMCell(input_features=self.input_features,
                                                 hidden_size=self.hidden_size,
                                                 num_layers=self.num_layers)

        def body_fn(cell, carry, x):
            # x = xs
            new_carrys, outputs = cell(carry, x)
            return new_carrys, outputs

        scanLSTM = nn.scan(
            body_fn, variable_broadcast="params",
            split_rngs={"params": False}, in_axes=1, out_axes=1)

        # input_sequence x has shape [batch_size, sequence_length, input_size]
        # output_sequence has shape [batch_size, sequence_length, hidden_size * num_layers]
        carry, output_sequence = scanLSTM(multi_lstm_cell, init_carry, x)

        # print("output_sequence shape: ", output_sequence.shape)

        final_output = nn.Dense(self.component_k * 6 + 1)(output_sequence)

        return final_output
