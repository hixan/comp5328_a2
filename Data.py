import torch


class Dataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset
    """

    def __init__(self, features, labels):
        """
        Initialisation

        :param features: features of the examples
        :param labels: labels of the examples
        """

        super(Dataset, self).__init__()

        self.features = features
        self.labels = labels

    def __len__(self):
        """
        The number of examples

        :return: the number of examples
        """

        return len(self.labels)

    def __getitem__(self, index):
        """
        Get an example by index

        :param index: index of the example
        :return: features and label of the example
        """

        # Load data and get label
        example_features = self.features[index]
        example_label = self.labels[index]

        return example_features, example_label


class DataLoader(torch.utils.data.DataLoader):
    """
    PyTorch DataLoader
    """

    def __init__(self, features, labels, **params):
        """

        :param features: features of the examples
        :param labels: labels of the examples
        :param params: parameters for the PyTorch DataLoader
        """

        dataset = Dataset(features, labels)

        super(DataLoader, self).__init__(dataset, **params)

