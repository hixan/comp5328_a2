from DataLoader import load_dataset
from Encoder import Encoder
from PIL import Image
import numpy as np
from timeit import default_timer as timer


# instantiate the encoder
encoder = Encoder(cnn_model_name='ResNet18', features_map_size=7)

# images are passed to the encoder in batches for efficiency
batch_size = 100

general_start = timer()

# loop through the datasets
for dataset_path in ['../datasets/fashionMNIST0.5.npz', '../datasets/fashionMNIST0.6.npz', '../datasets/CIFAR.npz']:

    # load dataset
    Xtr_val, Str_val, Xts, Yts = load_dataset(dataset_path)

    # compute number of examples and batches
    n_examples = Xtr_val.shape[0]
    n_batches = n_examples // batch_size

    # prepare an array to receive encoded examples
    encoded_images_Xtr_val = np.zeros((n_examples, 512))

    # loop through the batches
    for j in range(n_batches):

        start = timer()

        start_index = batch_size * j
        images = []

        # loop through the batch
        for i in range(batch_size):

            index = start_index + i

            # prepare image (it needs to have 3 layers)
            image = Image.fromarray(Xtr_val[index]).convert('RGB')
            images.append(image)

        # encode the images through ResNet
        # note: there is an average pooling layer at the end
        output = encoder.forward(images)
        encoded_images_Xtr_val[start_index: start_index+100] = output

        end = timer()
        print(j, int(round(end - general_start)), int(round(end - start)))

    n_examples = Xts.shape[0]
    n_batches = n_examples // batch_size

    encoded_images_Xts = np.zeros((n_examples, 512))

    for j in range(n_batches):

        start = timer()

        start_index = batch_size * j
        images = []

        for i in range(batch_size):

            index = start_index + i

            image = Image.fromarray(Xts[index]).convert('RGB')
            images.append(image)

        output = encoder.forward(images)
        encoded_images_Xts[start_index: start_index+100] = output

        end = timer()
        print(j, int(round(end - general_start)), int(round(end - start)))

    encoded_dataset_path = ''

    if dataset_path == '../datasets/fashionMNIST0.5.npz':
        encoded_dataset_path = 'FashionMNIST0_ResNet18.5.npz'
    elif dataset_path == '../datasets/fashionMNIST0.6.npz':
        encoded_dataset_path = 'FashionMNIST0_ResNet18.6.npz'
    elif dataset_path == '../datasets/CIFAR.npz':
        encoded_dataset_path = 'CIFAR_ResNet18.npz'

    # store Numpy arrays as in the original dataset
    np.savez(encoded_dataset_path, Xtr=encoded_images_Xtr_val, Str=Str_val, Xts=encoded_images_Xts, Yts=Yts)
