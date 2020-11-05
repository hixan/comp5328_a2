import numpy as np
import math


def load_dataset(file_path):

    # load dataset
    dataset = np.load(file_path)

    # training/validation set
    # features
    Xtr_val = dataset['Xtr']
    # labels
    Str_val = dataset['Str']

    # test set
    # features
    Xts = dataset['Xts']
    # labels
    Yts = dataset['Yts']

    return Xtr_val, Str_val, Xts, Yts


def split_training_validation(Xtr_val, Str_val, training_pct, seed):

    n_examples = Xtr_val.shape[0]
    n_examples_labels = Str_val.shape[0]

    if n_examples_labels != n_examples:
        print('Size mismatch between the size of features and labels')
        assert 0

    # set up a random generator
    rng = np.random.default_rng(seed)

    # create mask to separate training/validation set
    n_training_examples = int(math.floor(training_pct * n_examples + 0.5))
    indices = range(n_examples)
    training_indices = rng.choice(indices, n_training_examples, replace=False)
    mask = np.zeros(n_examples, dtype=bool)
    mask[training_indices] = True

    # the training set
    Xtr = Xtr_val[mask]
    Str = Str_val[mask]

    # the validation set
    Xval = Xtr_val[~mask]
    Sval = Str_val[~mask]

    return Xtr, Str, Xval, Sval
