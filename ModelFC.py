import torch


class Model(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super(Model, self).__init__()

        # fully connected layers model
        self.model = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_size),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(hidden_size, output_size),
                                         )

        # softmax
        self.softmax = torch.nn.Softmax(dim=1)

        # loss function
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, inputs, targets):

        # forward pass
        probabilities = self.model(inputs)

        # loss
        loss = self.loss(probabilities, targets)

        return loss

    def predict(self, inputs):

        with torch.no_grad():

            # forward pass
            probabilities = self.softmax(self.model(inputs))

            # convert to prediction
            predictions = torch.argmax(probabilities, dim=1)

        return predictions
