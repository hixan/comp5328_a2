from DataLoader import load_dataset, split_training_validation
from ModelFC import Model
import numpy as np
import torch


# set up a random generator
rng = np.random.default_rng(seed=42)


# function to partition a list into chunks
# as determined by the batch size
def divide_into_chunks(indices, batch_size):
    for i in range(0, len(indices), batch_size):
        yield indices[i:i + batch_size]


# set up the model and optimizer
input_size = 784
hidden_size = 48
output_size = 3
model = Model(input_size, hidden_size, output_size)
learning_rate = 0.01
optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

# load data and split training/validation set
Xtr_val, Str_val, Xts, Yts = load_dataset('../datasets/FashionMNIST0.5.npz')
Xtr, Str, Xval, Sval = split_training_validation(Xtr_val, Str_val, training_pct=0.8, seed=42)

n_epochs = 1
batch_size = 100

n_training_examples = Xtr.shape[0]

for epoch in range(n_epochs):

    indices = np.arange(n_training_examples)
    rng.shuffle(indices)
    batches = divide_into_chunks(indices, batch_size)

    for batch in batches:

        examples = Xtr[batch].astype(np.float32)
        examples = examples.reshape(examples.shape[0], -1) / 255
        labels = Str[batch]
        inputs = torch.from_numpy(examples)
        targets = torch.from_numpy(labels)

        loss = model.forward(inputs, targets)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()

        # update weights
        optimizer.step()

