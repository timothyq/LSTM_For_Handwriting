import torch
import re
import os
from tqdm import tqdm
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from models.lstm_model import Uncondition_LSTM
from data.stroke_dataset import Stroke_dataset
from losses.custom_loss import MDN_loss_function
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F


import argparse
from torch.utils.data import DataLoader, random_split, SequentialSampler


def parse_checkpoint_id(checkpoint_id):
    if checkpoint_id:
        match = re.match(r"(\d+)e_(\d+)", checkpoint_id)
        if match:
            return int(match.group(1)), int(match.group(2))
        else:
            raise ValueError(
                "Invalid checkpoint_id format. Expected format {epoch}e_{sequences}.")
    return None, None


def train_step(x, step_idx, epoch_idx, model, criterion, optimizer, device, writer, batch_size):
    model.train()

    input_seq = x[:, :-1]
    target_seq = x[:, 1:]

    input_seq = input_seq.to(device)

    outputs = model(input_seq)
    loss = criterion(outputs, target_seq)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Log weights and gradients to TensorBoard
    for name, param in model.named_parameters():
        writer.add_histogram(f'{name}', param, epoch_idx)
        if param.grad is not None:
            writer.add_histogram(f'{name}.grad', param.grad, epoch_idx)

    return loss.item()


def validate(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():  # No need to track gradients for validation
        for x in tqdm(val_loader, desc="Validation"):
            # Inputs for the model, excluding the last element
            input_seq = x[:, :-1]
            # Targets for the loss, excluding the first element
            target_seq = x[:, 1:]

            input_seq = input_seq.to(device)
            outputs = model(input_seq)
            loss = criterion(outputs, target_seq)  # Calculate the loss
            total_loss += loss.item()  # Sum up the loss

    avg_loss = total_loss / len(val_loader)  # Calculate the average loss
    return avg_loss


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
                        help="Identifier for the checkpoint e.g., '1e_4800'")
    args = parser.parse_args()

    EPOCHS = args.epochs
    LR = args.lr
    BATCH_SIZE = args.batch_size
    DEVICE = args.device
    CHECKPOINT_ID = args.checkpoint_id

    SAVE_PATH = f"models/condition_lstm{f'_{CHECKPOINT_ID}' if CHECKPOINT_ID else ''}.pt"

    start_epoch, start_sequence = parse_checkpoint_id(
        args.checkpoint_id) if CHECKPOINT_ID else 0, 0

    # 2. Load dataset
    stroke_data = Stroke_dataset(train=True)
    dataset_size = len(stroke_data)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    stroke_data.sort_by_sequence_length()

    train_dataset, val_dataset, test_dataset = random_split(
        stroke_data, [train_size, val_size, test_size])

    # train_dataset.sort_by_sequence_length()
    # val_dataset.sort_by_sequence_length()
    # test_dataset.sort_by_sequence_length()
    # train_dataset = SequentialSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 3. Instantiate the model, criterion, and optimizer
    model = Uncondition_LSTM(
        alphabet_size=78,
        input_size=3,
        hidden_size=400,
        num_layers=3,
        component_K=20,
        dropout=0.1
    ).to(DEVICE)
    model._initialize_weights()

    if CHECKPOINT_ID:
        model.load_state_dict(torch.load(SAVE_PATH))
        print(f"Model loaded from {SAVE_PATH}")

    criterion = MDN_loss_function
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4, nesterov=True)

    # Hyperparameters from the paper
    alpha = 0.95
    momentum = 0.9
    eps = 0.0001
    optimizer = optim.RMSprop(
        model.parameters(), lr=LR, alpha=alpha, eps=eps, momentum=momentum)

    writer = SummaryWriter(
        f"runs/experiment_uncon_RMSprop_outputlayerNorm_batch{BATCH_SIZE}")

# Train the model and validate periodically
    EPOCHS += start_epoch
    for epoch in range(start_epoch, EPOCHS):
        running_loss = 0.0
        for iteration, (x, _) in enumerate(tqdm(train_loader, desc="Training"), start=1):
            # break
            iter_loss = train_step(
                x, iteration, epoch, model, criterion, optimizer, DEVICE, writer, BATCH_SIZE)
            running_loss += iter_loss

            if iteration % 10 == 0:
                avg_loss = running_loss / 10
                print(
                    f"Epoch {epoch+1}/{EPOCHS}, Step {iteration}, Loss: {avg_loss:.4f}")
                running_loss = 0.0

        # Validate at the end of every epoch
        # valid_loss = validate(model, val_loader, criterion, DEVICE)
        # print(f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {valid_loss:.4f}")

        # Save each epoch
        dynamic_save_path = f"models/uncon_lstm_batch{BATCH_SIZE}_outputLayerNorm_{epoch+1}e.pt"
        torch.save(model.state_dict(), dynamic_save_path)
        print(f"Model saved to {dynamic_save_path}")

    writer.close()
    print("Training complete")


if __name__ == "__main__":
    main()
