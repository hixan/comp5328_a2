from Utils import load_dataset, split_training_validation
from Data import DataLoader
from ModelFC import Model
import torch
import numpy as np
import matplotlib.pyplot as plt
import math


# load data and split training/validation set
Xtr_val, Str_val, Xts, Yts = load_dataset('../datasets/FashionMNIST0.5.npz')
Xtr, Str, Xval, Sval = split_training_validation(Xtr_val, Str_val, training_pct=0.8, seed=42)

# batching parameters
params = {'batch_size': 200, 'shuffle': True}

Xtr = Xtr[0:2000]
Str = Str[0:2000]

# set up the training set
training_features = Xtr.reshape(Xtr.shape[0], -1).astype(np.float32) / 255
training_labels = Str
training_generator = DataLoader(training_features, training_labels, **params)

Xval = Xval[0:200]
Sval = Sval[0:200]

# set up the training set
validation_features = Xval.reshape(Xval.shape[0], -1).astype(np.float32) / 255
validation_labels = Sval
validation_generator = DataLoader(validation_features, validation_labels, **params)

# set up the model
input_size = training_features.shape[1]
output_size = np.unique(training_labels).shape[0]
hidden_size = int(round(math.sqrt(input_size * output_size)))
model = Model(input_size, hidden_size, output_size)

# set up the optimizer
learning_rate = 1e-2
optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

# number of epochs
n_epochs = 500

losses = []
batch_losses = []
accuracies = []

for epoch in range(n_epochs):

    avg_loss = 0
    n = 0

    batch_loss = []
    for inputs, targets in training_generator:

        # forward propagation
        loss = model.forward(inputs, targets)
        avg_loss = (n * avg_loss + loss.item()) / (n + 1)
        batch_loss.append(loss.item())
        n += 1

        # backward propagation
        optimizer.zero_grad()
        loss.backward()

        # update parameters
        optimizer.step()

    losses.append(avg_loss)
    batch_losses.append(batch_loss)

    correct = []
    n = 0

    for inputs, targets in validation_generator:

        # predict
        predictions = model.predict(inputs)

        # compute accuracy
        correct.append(torch.eq(predictions, targets).numpy())
        #n_examples = inputs.size()[0]
        #accuracy = (n * accuracy + n_correct / n_examples) / (n + 1)

    accuracies.append(np.array(correct).mean())

    if epoch % 10 == 0:
        print(epoch, avg_loss, accuracies[-1])

plt.plot(losses)
plt.show()

bl = np.array(batch_losses, dtype=float)
plt.plot(bl.mean(axis=1))
#plt.plot(bl.mean(axis=1) + bl.std(axis=1))
#plt.plot(bl.mean(axis=1) - bl.std(axis=1))

plt.plot(accuracies)
plt.show()
