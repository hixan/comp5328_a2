import torch


class Model(torch.nn.Module):

    def __init__(self, input_size, output_size, transition_matrix=None):

        super(Model, self).__init__()

        # classification layers
        self.fc = torch.nn.Linear(input_size, 100)
        self.softmax = torch.nn.Softmax(dim=1)

        # negative log likelihood loss
        # note: it expects log-probabilities
        #self.loss = torch.nn.NLLLoss()
        self.loss = torch.nn.CrossEntropyLoss()

        # transition matrix
        if transition_matrix is None:
            self.transition_matrix = torch.eye(output_size)
        else:
            self.transition_matrix = transition_matrix

    def _loss_forward(self, probabilities, targets):

        noisy_log_probabilities = torch.log(torch.matmul(probabilities, self.transition_matrix))
        average_loss = self.loss(noisy_log_probabilities, targets)

        return average_loss

    def forward(self, encoded_images, targets):

        # forward pass
        outputs = self.fc(encoded_images)
        #probabilities = self.softmax(outputs)

        # loss
        #loss = self._loss_forward(probabilities, targets)
        loss = self.loss(outputs, targets)

        return loss

    def predict(self, encoded_images):

        # forward pass
        outputs = self.fc(encoded_images)
        probabilities = self.softmax(outputs)

        predictions = torch.argmax(probabilities, dim=1)

        return predictions
