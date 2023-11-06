from models.lstm_model import Condition_LSTM, Uncondition_LSTM
from losses.custom_loss import MDN_loss_function
import pytest
import math
import torch
import torch.nn as nn

# ------------------- Uncondition_LSTM test session starts -------------------


@pytest.mark.uncon_lstm
def test_uncondition_lstm_params():
    # This test function is to:
    # 1. check if the model is able to instantiate without any errors.
    # 2. check if the model is initialized with the correct parameters.
    model = Uncondition_LSTM(
        input_size=3, hidden_size=400, num_layers=3, component_K=20)
    assert isinstance(model.lstm, nn.LSTM)
    assert isinstance(model.fc, nn.Linear)
    assert model.lstm.input_size == 3
    assert model.lstm.hidden_size == 400
    assert model.lstm.num_layers == 3
    assert model.fc.in_features == 400
    assert model.fc.out_features == 121


@pytest.mark.uncon_lstm
def test_uncondition_lstm_forward_backword():
    # This test function is to:
    # 1. check if the model is able to instantiate and forward without any errors.
    # 2. test backpropagation

    model = Uncondition_LSTM(
        input_size=3, hidden_size=400, num_layers=3, component_K=20)
    model.train()  # Ensure the model is in training mode
    x = torch.rand(10, 5, 3)  # [batch_size, seq_len, input_size]
    output = model.forward(x)  # [batch_size, seq_len, output_size]

    # Define a dummy target tensor
    target = torch.rand(10, 5, 121)

    loss = MDN_loss_function(output, target)

    # Test backpropagation
    loss.backward()

    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            # Ensure gradient is not NaN
            assert not torch.isnan(
                param.grad).any(), "Gradient has NaN values!"

            # Check if at least one gradient value is non-zero
            if torch.sum(param.grad).item() != 0:
                has_grad = True

    assert has_grad, "No gradients found!"
    assert output.shape == (10, 5, 121)


@pytest.mark.uncon_lstm
def test_uncondition_lstm_hidden_initialization():
    # This test function is to:
    # 1. check if the model is able to instantiate its hidden states without any errors.
    model = Uncondition_LSTM(
        input_size=3, hidden_size=400, num_layers=3, component_K=20)
    # x = torch.rand(10, 5, 3)  # [seq_len, batch_size, input_size]
    hidden, cell = model.init_hidden(batch_size=5)
    assert hidden.shape == (3, 5, 400)
    assert cell.shape == (3, 5, 400)


@pytest.mark.uncon_lstm
def test_uncondition_lstm_variant_forward():
    model = Uncondition_LSTM(
        input_size=3, hidden_size=900, num_layers=1, component_K=20)
    x = torch.rand(10, 5, 3)  # [seq_len, batch_size, input_size]
    output = model.forward(x)
    assert output.shape == (10, 5, 121)


@pytest.mark.uncon_lstm
def test_uncondition_lstm_variant_forward_backward():
    model = Uncondition_LSTM(
        input_size=3, hidden_size=900, num_layers=1, component_K=20)
    model.train()  # Ensure the model is in training mode
    x = torch.rand(10, 5, 3)  # [seq_len, batch_size, input_size]
    output = model.forward(x)

    # dummy target
    target = torch.rand(10, 5, 121)

    loss = MDN_loss_function(output, target)

    # Test backpropagation
    loss.backward()

    # Test if gradients are not all zeros
    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            # Ensure gradient is not NaN
            assert not torch.isnan(
                param.grad).any(), "Gradient has NaN values!"

            # Check if at least one gradient value is non-zero
            if torch.sum(param.grad).item() != 0:
                has_grad = True

    assert has_grad, "No gradients found!"
    assert output.shape == (10, 5, 121)

# ------------------- Condition_LSTM test session starts -------------------


@pytest.mark.con_lstm
def test_coditional_LSTM():
    # This test function is to:
    # 1. check if the model is able to instantiate without any errors.
    # 2. check if the model is able to forward pass without any errors.
    # 3. check if the model is able to return the correct output shape.
    # 4. check if the returned states are not all zeros.

    alphabet_size = 10
    seq_len_t = 50
    seq_len_c = 40
    window_K = 9
    input_size = 3
    hidden_size = 48
    num_layers = 1
    component_K = 20
    batch_size = 32
    output_size = 6 * component_K + 1

    model = Condition_LSTM(alphabet_size, window_K,
                           input_size, hidden_size, num_layers, component_K)

    # Dummy input and condition sequence
    x = torch.rand(batch_size, seq_len_t, input_size)
    c_seq = torch.rand(batch_size, seq_len_c, alphabet_size)

    # Forward pass
    outputs, states = model.forward(x, c_seq)
    h1_prev, c1_prev, h2_prev, c2_prev, kappa, w_t_pre = states

    # 1. Check shapes of the outputs and states
    assert outputs.shape == (
        batch_size, seq_len_t, output_size), f"Expected output shape {(batch_size, seq_len_t, output_size)}, but got {outputs.shape}"
    assert h1_prev.shape == c1_prev.shape == (
        num_layers, batch_size, hidden_size), f"Expected h1_prev and c1_prev shape {(num_layers, batch_size, hidden_size)}, but got {h1_prev.shape} and {c1_prev.shape}."
    assert h2_prev.shape == c2_prev.shape == (
        num_layers, batch_size, hidden_size), f"Expected h2_prev and c2_prev shape {(num_layers, batch_size, hidden_size)}, but got {h2_prev.shape} and {c2_prev.shape}."

    # 2. Check if states are not all zeros.
    assert not torch.all(
        h1_prev == 0), "Did not expect all zeros for h1_prev after forward pass"
    assert not torch.all(
        h2_prev == 0), "Did not expect all zeros for h2_prev after forward pass"


@pytest.mark.con_lstm
def test_compute_phi_output_shape():
    # This test function is to:
    # 1. check if the phi function is able to compute without any errors.

    alphabet_size = 10
    seq_len_t = 50
    seq_len_c = 40
    window_K = 9
    input_size = 3
    hidden_size = 48
    num_layers = 1
    component_K = 20
    batch_size = 32

    model = Condition_LSTM(alphabet_size, window_K,
                           input_size, hidden_size, num_layers, component_K)

    # Dummy input and target
    x = torch.rand(batch_size, seq_len_t, input_size)
    c_seq = torch.rand(batch_size, seq_len_c, alphabet_size)

    # Dummy input tensors
    alpha = torch.rand(batch_size, window_K)
    beta = torch.rand(batch_size, window_K)
    kappa = torch.rand(batch_size, window_K)
    sequence_c = torch.rand(batch_size, seq_len_c, alphabet_size)

    # Compute phi
    phi = model.compute_phi(alpha, beta, kappa, sequence_c)

    # Verify the shape of the phi output
    assert phi.shape == (
        batch_size, alphabet_size), f"Expected shape ({batch_size}, {alphabet_size}), but got {phi.shape}."


@pytest.mark.con_lstm
def test_condition_lstm_forward_backward():
    # This test function checks:
    # 1. if the model is able to instantiate, forward pass, and return the correct output shape.
    # 2. if the model returns correct shapes for each state tensor.
    # 3. if the model can compute a loss and perform a backward pass.

    alphabet_size = 10
    seq_len_t = 50
    seq_len_c = 40
    window_K = 9
    input_size = 3
    hidden_size = 48
    num_layers = 1
    component_K = 20
    batch_size = 32

    model = Condition_LSTM(alphabet_size, window_K,
                           input_size, hidden_size, num_layers, component_K)

    # Dummy input and target
    x = torch.rand(batch_size, seq_len_t, input_size)
    c_seq = torch.rand(batch_size, seq_len_c, alphabet_size)
    target = torch.rand(batch_size, seq_len_t, 6 * component_K + 1)

    # Forward pass
    output, states = model.forward(x, c_seq)

    # Check output tensor shape
    assert output.shape == (
        batch_size, seq_len_t, 6 * component_K + 1), f"Expected shape ({batch_size}, {seq_len_t}, {6 * component_K + 1}), but got {output.shape}"

    # Check state tensor shapes
    h1_prev, c1_prev, h2_prev, c2_prev, kappa, w_t_pre = states
    assert h1_prev.shape == c1_prev.shape == (
        num_layers, batch_size, hidden_size), f"Invalid LSTM1 states shape. Expected ({num_layers}, {batch_size}, {hidden_size}), got {h1_prev.shape} and {c1_prev.shape}."
    assert h2_prev.shape == c2_prev.shape == (
        num_layers, batch_size, hidden_size), f"Invalid LSTM2 states shape. Expected ({num_layers}, {batch_size}, {hidden_size}), got {h2_prev.shape} and {c2_prev.shape}."
    assert kappa.shape == (
        batch_size, window_K), f"Invalid kappa shape. Expected ({batch_size}, {window_K}), but got {kappa.shape}."
    assert w_t_pre.shape == (
        batch_size, alphabet_size), f"Invalid w_t_pre shape. Expected ({batch_size}, {alphabet_size}), but got {w_t_pre.shape}."

    # Compute loss and perform backward pass
    loss = MDN_loss_function(output, target)
    loss.backward()
