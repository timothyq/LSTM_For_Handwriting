import torch
import re
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
from models.lstm_model import Condition_LSTM, Uncondition_LSTM
from data.stroke_dataset import Stroke_dataset
from losses.custom_loss import MDN_loss_function
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

import argparse
from torch.utils.data import DataLoader, random_split
# Assuming Stroke_dataset and Condition_LSTM are defined elsewhere


def parse_checkpoint_id(checkpoint_id):
    if checkpoint_id:
        match = re.match(r"(\d+)e_(\d+)", checkpoint_id)
        if match:
            return int(match.group(1)), int(match.group(2))
        else:
            raise ValueError(
                "Invalid checkpoint_id format. Expected format {epoch}e_{sequences}.")
    return None, None


def main():
    # 1. Parse arguments and setup environment
    parser = argparse.ArgumentParser(
        description="Training script for Condition_LSTM")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int,
                        default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint_id", type=str, default="",
                        help="Identifier for the checkpoint e.g., '1e_4800'")
    args = parser.parse_args()

    EPOCHS = args.epochs
    LR = args.lr
    BATCH_SIZE = args.batch_size
    DEVICE = args.device
    CHECKPOINT_ID = args.checkpoint_id

    # Append checkpoint ID to the save path if provided
    SAVE_PATH = f"models/condition_lstm{f'_{CHECKPOINT_ID}' if CHECKPOINT_ID else ''}.pt"

    start_epoch, start_sequence = parse_checkpoint_id(args.checkpoint_id)

    # 2. Load dataset
    stroke_data = Stroke_dataset(train=True)
    dataset_size = len(stroke_data)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        stroke_data, [train_size, val_size, test_size])

    # Here, the worker_init_fn should be set to a callable, like a function or a lambda, not an integer.
    # If you want to set the number of workers, use `num_workers` argument instead.
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 3. Instantiate the model, criterion, and optimizer
    model = Condition_LSTM(
        alphabet_size=78,
        window_K=10,
        input_size=3,
        hidden_size=400,
        num_layers=1,
        component_K=20
    ).to(DEVICE)

    # Load the model from checkpoint if provided
    if CHECKPOINT_ID:
        model.load_state_dict(torch.load(SAVE_PATH))
        print(f"Model loaded from {SAVE_PATH}")

    criterion = MDN_loss_function
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    # 4. Define training and validation loops

    # Initialize the TensorBoard writer
    writer = SummaryWriter('runs/experiment_1')

    def train_step(x, condition_sequence, step_idx, epoch_idx):

        model.train()

        # Split the data into input and target sequences
        input_seq = x[:, :-1]  # Everything except the last timestep
        target_seq = x[:, 1:]  # Everything except the first timestep

        input_seq, condition_sequence = input_seq.to(
            DEVICE), condition_sequence.to(DEVICE)

        outputs = model(input_seq, condition_sequence)
        loss = criterion(outputs, target_seq)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log to TensorBoard
        writer.add_scalar('Loss/train', loss.item(), epoch_idx *
                          len(x) + step_idx)
        # log gradients or weights
        for name, weight in model.named_parameters():
            writer.add_histogram(name, weight, epoch)
            writer.add_histogram(f'{name}.grad', weight.grad, epoch)

        return loss.item()

    def validate():
        model.eval()
        total_loss = 0
        with torch.no_grad():
            # i = 0
            for x, condition_sequence in tqdm(val_loader, desc="Validation"):
                # Split the data into input and target sequences
                input_seq = x[:, :-1]
                target_seq = x[:, 1:]

                input_seq, condition_sequence = input_seq.to(
                    DEVICE), condition_sequence.to(DEVICE)
                outputs = model(input_seq, condition_sequence)
                loss = criterion(outputs, target_seq)
                total_loss += loss.item()
                # i += 1
                # if i > 1:
                #     break

        return total_loss / len(val_loader)

    # 5. Train the model and validate periodically
    EPOCHS += start_epoch
    total_sequences_processed = start_sequence
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0
        for iteration, (x, condition_sequence) in enumerate(tqdm(train_loader, desc="Training")):
            # x: [batch_size, seq_len, input_size]
            print("x", x.shape)
            # condition_sequence: [batch_size, seq_len, alphabet_size]
            print("condition_sequence", condition_sequence.shape)
            iter_loss = train_step(x, condition_sequence, iteration, epoch)
            running_loss += iter_loss
            print("iter_loss", iter_loss)

            # Update the number of sequences processed
            total_sequences_processed += len(x)

            # Print every 10 iterations
            if iteration % 10 == 9:
                print(
                    f"Epoch: {epoch+1}/{EPOCHS}, Iteration: {iteration+1} - Training Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

        # Validate at the end of every epoch
        valid_loss = validate()
        print(f"Epoch: {epoch+1}/{EPOCHS} - Validation Loss: {valid_loss:.4f}")

    # 7. Save the model
    dynamic_save_path = f"models/condition_lstm_{epoch+1}e_{total_sequences_processed}.pt"
    torch.save(model.state_dict(), dynamic_save_path)
    print(f"Model saved to {dynamic_save_path}")


if __name__ == "__main__":
    main()
