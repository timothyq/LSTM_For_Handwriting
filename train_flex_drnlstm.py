import os
from flax.core import freeze
import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
import optax
from models.flex_lstm_model import RMSNormLSTM, MultiRMSNormedLSTMCell
from losses.flex_loss import mdn_loss_function
import argparse
from data.stroke_dataset import Stroke_dataset
from torch.utils.data import random_split
import flax.linen as nn
from torch.utils.data import DataLoader
import torch
import pickle
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import flax

# def create_train_state(rng, learning_rate, model):
#     params = model.init(rng, jnp.ones(input_shape))['params']
#     tx = optax.rmsprop(learning_rate)
#     return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def print_params_structure(params, indent=0):
    for key, value in params.items():
        print("  " * indent + key)
        if isinstance(value, dict):
            # Recursive call for nested dictionaries (submodules)
            print_params_structure(value, indent + 1)
        else:
            # Print the shape of the parameter array
            print("  " * (indent + 1) + str(value.shape))


def loss_fn(params, state, batch):
    inputs, _ = batch
    x = inputs[:, :-1]
    y = inputs[:, 1:]
    logits = state.apply_fn({'params': params}, x)
    loss = mdn_loss_function(logits, y)
    return loss


def log_params(writer, params, tag_prefix="", global_step=None):
    for name, param in params.items():
        full_tag = f"{tag_prefix}/{name}" if tag_prefix else name

        if isinstance(param, (dict, flax.core.frozen_dict.FrozenDict)):
            log_params(writer, param, tag_prefix=full_tag,
                       global_step=global_step)
        elif isinstance(param, jnp.ndarray):
            # Ensure param is numeric before converting to NumPy array
            # Integer, unsigned integer, or float
            print("param.dtype.kind", param.dtype.kind)
            if param.dtype.kind in {'i', 'u', 'f'}:
                param_np = np.array(param)
                writer.add_histogram(full_tag, param_np,
                                     global_step=global_step)
            else:
                print(
                    f"Skipping non-numeric parameter: {full_tag} of type {param.dtype}")
        else:
            print(
                f"Skipping unexpected parameter type: {full_tag} of type {type(param)}")


@jax.jit
def train_step(state, batch):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, state, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss


def collate_fn(batch):
    stroke_sequences, sentences = zip(*batch)
    stroke_sequences_padded = pad_sequence([s.clone().detach() if isinstance(
        s, torch.Tensor) else torch.tensor(s) for s in stroke_sequences], batch_first=True, padding_value=0)

    max_sentence_len = max(sentence.shape[0] for sentence in sentences)
    sentences_padded = torch.stack([F.pad(sentence,
                                          (0, 0, 0, max_sentence_len -
                                           sentence.shape[0]),
                                          "constant", 0) for sentence in sentences])
    return stroke_sequences_padded, sentences_padded


def main():
    # 1. Parse arguments and setup environment
    parser = argparse.ArgumentParser(
        description="Training script for Condition_LSTM")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int,
                        default=1, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint_id", type=str, default="",
                        help="Identifier for the checkpoint e.g., '1e'")
    args = parser.parse_args()

    EPOCHS = args.epochs
    LR = args.lr
    BATCH_SIZE = args.batch_size
    DEVICE = args.device
    CHECKPOINT_ID = args.checkpoint_id
    start_epoch = int(CHECKPOINT_ID.rstrip('e')) if 'e' in CHECKPOINT_ID else 0

    SAVE_PATH = f"model_checkpoints/flex_rmsnorm_model_epoch_{CHECKPOINT_ID}.pkl"

    # 2. Load dataset
    stroke_data = Stroke_dataset(train=True)
    dataset_size = len(stroke_data)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    stroke_data.sort_by_sequence_length()

    train_dataset, val_dataset, test_dataset = random_split(
        stroke_data, [train_size, val_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # 3. Create schedular
    base_learning_rate = 0.001
    steps_per_epoch = len(train_loader)
    total_epochs = EPOCHS
    warmup_epochs = 1
    total_steps = steps_per_epoch * total_epochs
    print("total_steps", total_steps)
    lr_schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(init_value=0.0001, end_value=base_learning_rate,
                                  transition_steps=warmup_epochs * steps_per_epoch),
            optax.cosine_decay_schedule(
                init_value=base_learning_rate, decay_steps=total_steps - warmup_epochs * steps_per_epoch)
        ],
        boundaries=[warmup_epochs * steps_per_epoch]
    )

    # 3. Create model
    model = RMSNormLSTM(num_layers=3, hidden_size=400,
                        input_features=3, component_k=20)
    # lstm = nn.RNN(MultiRMSNormedLSTMCell)
    rng = jax.random.PRNGKey(0)
    tx = optax.rmsprop(learning_rate=lr_schedule,
                       decay=0.95, eps=1e-8)

    input_shape = (BATCH_SIZE, 300, 3)
    params = model.init(rng, jnp.ones(input_shape))['params']
    # print_params_structure(params)

    writer = SummaryWriter(
        f"runs/experiment_flex_{BATCH_SIZE}")
    checkpoint_dir = 'model_checkpoints'

    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

    for epoch in range(start_epoch, EPOCHS + start_epoch):
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}", unit='batch') as pbar:
            for iteration, batch in enumerate(train_loader, start=1):
                batch_x, batch_y = batch
                batch_x_jnp = jnp.array(batch_x.detach().numpy())
                batch_y_jnp = jnp.array(batch_y.detach().numpy())
                state, loss = train_step(state, (batch_x_jnp, batch_y_jnp))

                running_loss += loss.item()

                writer.add_scalar('Training/Loss', loss.item(),
                                  epoch * len(train_loader) + iteration)

                # if iteration % 1 == 0:
                #     log_params(writer, freeze(state.params),
                #                tag_prefix="Training/Params",
                #                global_step=epoch * len(train_loader) + iteration)
                running_loss += loss.item()
                pbar.set_description(
                    f"Epoch {epoch+1}/{EPOCHS + start_epoch} - Iteration {iteration} - Loss: {loss:.4f}")
                pbar.update(1)

        save_path = checkpoints.save_checkpoint(
            checkpoint_dir, target=state, step=epoch, prefix='flex_rmsnorm_model_', keep=3
        )
    print(f"Checkpoint saved to {save_path}")

    writer.close()


if __name__ == "__main__":
    main()
