from Model import Model
import torch


# set up the model
n_features = 2
n_classes = 3
model = Model(n_features, n_classes)

# create inputs
encoded_images = torch.tensor([[1, 2], [3,4], [2, 3], [1, 4]], dtype=torch.float)
targets = torch.tensor([0, 1, 2, 1])

loss = model.forward(encoded_images, targets)

print(loss)


