import torch
import torchvision as tv


class Encoder(torch.nn.Module):
    """
    Pretrained CNN Encoder
    """

    # constructor
    def __init__(self, cnn_model_name, features_map_size):
        """
        Constructor

        :param cnn_model_name: the name of the ResNet model
        :param features_map_size: the size of the N square maps generated in the last but one layer of the CNN
        """

        super(Encoder, self).__init__()

        # set up the image transformations required to generate the input for the ResNet
        preprocess = tv.transforms.Compose([tv.transforms.Resize((32 * features_map_size,
                                                                  32 * features_map_size)),
                                            tv.transforms.ToTensor(),
                                            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

        # import the layers of the pretrained CNN model
        # and drop the final fully connected layer
        if cnn_model_name == 'ResNet18':
            cnn_model = tv.models.resnet18(pretrained=True)
            layers = list(cnn_model.children())[:-1]
        else:
            layers = None

        self.preprocess = preprocess
        self.encoder = torch.nn.Sequential(*layers)

    # generate the encoding
    def forward(self, images):
        """
        Forward Propagation

        :param images: a list of RGB image
        :return: a 1D tensor with N elements
        """

        # transform the image to fulfil the input requirement of the CNN
        preprocessed_images = []
        for image in images:
            preprocessed_image = self.preprocess(image)
            preprocessed_images.append(preprocessed_image)
        preprocessed_images = torch.stack(preprocessed_images)

        # forward propagate
        with torch.no_grad():
            output = self.encoder(preprocessed_images)

        # eliminate redundant dimensions
        # and typecast to Numpy array
        output = output.squeeze(3).squeeze(2).numpy()

        return output
