import jax
import jax.numpy as jnp
import flax.linen as nn
from models.flex_lstm_model import RMSNormedLSTMCell, MultiRMSNormedLSTMCell, RMSNormLSTM
import pytest


def print_nested_dict(d, depth, level=0):
    """Recursively prints nested dictionary contents up to a specified depth."""
    if isinstance(d, dict):
        for key, value in d.items():
            print("  " * level + str(key))
            if isinstance(value, dict) and level < depth - 1:
                print_nested_dict(value, depth, level + 1)
            else:
                print("  " * (level + 1) + str(value.shape))
    else:
        print(d)


@pytest.mark.flex_uncon
def test_rms_normed_lstm_cell():
    input_shape = (10, 3)  # (batch_size,  features)
    features = 300

    model = RMSNormedLSTMCell(features=features, name="lstm_cell")
    x = jnp.ones(input_shape)

    dummy_lstm = nn.LSTMCell(features=features, name="dummy_lstm")
    carry_lstm = dummy_lstm.initialize_carry(
        jax.random.key(1), x.shape)  # (batch_size, seq_len, features)

    variables = model.init(jax.random.PRNGKey(0), carry_lstm, x)
    print_nested_dict(variables, 10)

    print("carry_lstm[0]", carry_lstm[0].shape)
    print("carry_lstm[1]", carry_lstm[1].shape)

    new_carry_lstm, y = model.apply(variables, carry_lstm, x)

    assert y.shape == (input_shape[0],
                       features), "Output shape mismatch"
    assert len(new_carry_lstm) == 2, "LSTM carry state should have 2 elements"
    assert new_carry_lstm[0].shape == (
        input_shape[0], features), "LSTM hidden state shape mismatch"
    assert new_carry_lstm[1].shape == (
        input_shape[0], features), "LSTM cell state shape mismatch"


@pytest.mark.flex_uncon
def test_multi_rms_normed_lstm_cell():
    input_shape = (10, 3)  # (batch_size, seq_len, features)
    input_features = 3
    hidden_size = 300
    num_layers = 3

    model = MultiRMSNormedLSTMCell(
        input_features=input_features, hidden_size=hidden_size, num_layers=num_layers)
    x = jnp.ones(input_shape)

    carry = []
    dummy_lstm = nn.LSTMCell(features=hidden_size, name="dummy_lstm")
    for i in range(num_layers):
        input_size = input_features if i == 0 else input_features + hidden_size
        carry.append(dummy_lstm.initialize_carry(
            jax.random.PRNGKey(i), (input_shape[0], input_size)))

    variables = model.init(jax.random.PRNGKey(0), carry, x)

    new_carrys, outputs = model.apply(variables, carry, x)

    assert len(new_carrys) == num_layers, \
        "New carry state length should match the number of layers"
    assert outputs.shape == (input_shape[0], hidden_size * num_layers), \
        "Output shape mismatch"
    assert len(new_carrys[0]) == 2, "LSTM carry state should have 2 elements"
    for i, (h, c) in enumerate(new_carrys):
        layer_features = hidden_size
        assert h.shape == (
            input_shape[0], layer_features), f"Hidden state shape mismatch for layer {i}"
        assert c.shape == (
            input_shape[0], layer_features), f"Cell state shape mismatch for layer {i}"


@pytest.mark.flex_uncon
def test_rmsnorm_lstm():
    batch_size = 10
    sequence_length = 200
    input_features = 3
    component_k = 10

    num_layers = 3
    hidden_size = 300

    model = RMSNormLSTM(num_layers=num_layers, hidden_size=hidden_size,
                        input_features=input_features, component_k=component_k)

    x = jnp.ones((batch_size, sequence_length, input_features))

    variables = model.init(jax.random.PRNGKey(0), x)

    output = model.apply(variables, x)

    expected_output_shape = (batch_size, sequence_length, component_k * 6 + 1)
    assert output.shape == expected_output_shape, "Output shape mismatch"

    assert not jnp.all(output == 0), "Model output should not be all zeros"

    mean_output = jnp.mean(output)
    assert mean_output != 0, "Mean of the output should not be zero"
    assert jnp.isfinite(mean_output), "Mean of the output should be finite"

    output_again = model.apply(variables, x)
    assert jnp.allclose(
        output, output_again), "Model output should be deterministic"
