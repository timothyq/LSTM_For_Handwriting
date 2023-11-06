from data.stroke_dataset import Stroke_dataset
from models.lstm_model import Condition_LSTM
import torch
import torch.nn.functional as F
import numpy as np

import random
import matplotlib.pyplot as plt
from utils import plot_stroke_with_ending, plot_stroke_with_ending2

# import sys
# print(sys.path)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize  model
model = Condition_LSTM(
    alphabet_size=78,
    window_K=10,
    input_size=3,
    hidden_size=400,
    num_layers=1,
    component_K=20
).to(DEVICE)

model.load_state_dict(torch.load('models/condition_lstm_1e_4800.pt'))

model.eval()

dataset = Stroke_dataset(train=False)

sample = dataset[random.randint(0, len(dataset))]
target, c_seq = sample
indices = np.argmax(c_seq, axis=1)
sentence = ''.join([dataset.vocab[index] for index in indices])
print(len(sentence))
print(sentence)
print("shape of target: ", target.shape)
seq_len = len(target)

# Predict
initial_input = torch.zeros(1, 1, 3).to(DEVICE)
predictions = model.predict(initial_input, c_seq.unsqueeze(0), seq_len)

# print full tensor:
# Set print options (temporarily)
torch.set_printoptions(threshold=10_000)
print(predictions[0])
print(target)
torch.set_printoptions(profile="default")
plot_stroke_with_ending2(predictions[0])
plot_stroke_with_ending(target)
