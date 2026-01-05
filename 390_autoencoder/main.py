import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

# To get the MNIST (digit images) dataset
from keras.datasets import mnist

torch.manual_seed(0)

# Download MNIST Data
(mnist_train, labels_train), (mnist_test, labels_test) = mnist.load_data()

# Load data as Numpy arrays of size (#datapoints, 28*28=784)
mnist_train = mnist_train.astype("float32") / 255.0
mnist_test = mnist_test.astype("float32") / 255.0
mnist_train = mnist_train.reshape((len(mnist_train), np.prod(mnist_train.shape[1:])))
mnist_test = mnist_test.reshape((len(mnist_test), np.prod(mnist_test.shape[1:])))

# Split test data into a test and validation set:
val_data = mnist_test[: (mnist_test.shape[0] // 2), :]
test_data = mnist_test[(mnist_test.shape[0] // 2) :, :]
train_data = mnist_train

val_labels = labels_test[: (mnist_test.shape[0] // 2)]
test_labels = labels_test[(mnist_test.shape[0] // 2) :]
train_labels = labels_train

# Display dataset information
print("Downloaded the following data:")
print(
    f"train_data has shape {train_data.shape}, containing {train_data.shape[0]} images represented as ({train_data.shape[1]}, 1) vectors"
)
print(
    f"val_data has shape {val_data.shape}, containing {val_data.shape[0]} images represented as ({val_data.shape[1]}, 1) vectors"
)
print(
    f"train_data has shape {test_data.shape}, containing {test_data.shape[0]} images represented as ({test_data.shape[1]}, 1) vectors"
)


# We'll create a Dataset class to use with PyTorch's Built-In Dataloaders
class MNISTDataset(Dataset):
    """
    A custom dataset class to use with PyTorch's built-in dataloaders.
    This will make feeding images to our models much easier downstream.

    data: np.arrays downloaded from Keras' databases
    vectorize: if True, outputed image data will be (784,)
                   if False, outputed image data will be (28,28)
    """

    def __init__(self, data, labels, vectorize=True):
        self.data = data
        self.labels = labels
        self.vectorize = vectorize

    def __getitem__(self, idx):
        image_data = self.data[idx, :]
        image_data = image_data.reshape((1, 28, 28))
        if self.vectorize:
            image_data = image_data.reshape((784,))
        image_label = self.labels[idx]
        return image_data, image_label

    def __len__(self):
        return self.data.shape[0]


# Create MNISTDataset objects for each of our train/val/test sets
train_dataset = MNISTDataset(train_data, train_labels)
val_dataset = MNISTDataset(val_data, val_labels)
test_dataset = MNISTDataset(test_data, test_labels)

# Create a PyTorch dataloader for each train/val/test set
# We'll use a batch size of 256 for the rest of this assignment.
train_loader = DataLoader(train_dataset, batch_size=256)
val_loader = DataLoader(val_dataset, batch_size=256)
test_loader = DataLoader(test_dataset, batch_size=256)

# Display dataloader info
print("Created the following Dataloaders:")
print(f"train_loader has {len(train_loader)} batches of training data")
print(f"val_loader has {len(val_loader)} batches of validation data")
print(f"test_loader has {len(test_loader)} batches of testing data")


def test(model, device, test_loader):
    """
    Function for testing our models. One call to test() runs through every
    datapoint in our dataset once.

    model: an instance of our model, in this assignment, this will be your autoencoder

    device: either "cpu" or "cuda:0", depending on if you're running with GPU support

    test_loader: the dataloader for the data to run the model on
    """
    # set model to evaluation mode
    model.eval()

    # we'll keep track of total loss to calculate the average later
    test_loss = 0

    # don’t track gradients in testing, since no backprop
    with torch.no_grad():
        # iterate through each test image
        for input, _ in test_loader:

            # send input image to GPU if using GPU
            input = input.to(device)

            # run input through our model
            output = model(input)

            loss_function = nn.MSELoss()
            test_loss += loss_function(output, input)
            ## END YOUR CODE

    # calculate average loss per batch
    test_loss /= len(test_loader)
    return test_loss.item()


################################## DEEP AUTOENCODER ##################################
class DeepAutoEncoder(nn.Module):
    """
    An autoencoder with two fully-connected layer for the encoder, and two
    fully-connected layer for the decoder.

    compress: number of units in hidden layer of encoder and decoder

    representation_size: integer reprsenting the size of the learned representation.
                         Also known as the encoding or embedding dimension.
                         This is also the size of the output of the encoder.

    input_size: integer representing the size of each input column vector.
                In this assignment, this will end up being 784.
    """

    def __init__(self, compress, representation_size, input_size=784):
        """
        This is where all the modules that make up our deep autoencoder get stored.

        self.encoder_preactivation_1 should implement the first fully-connected layer (without activation yet)
        self.encoder_activation_1 should implement the activation function of the encoder's first layer
        self.encoder_preactivation_2 and self.activation_2 mirror those of the first encoder layer

        self.decoder_preactivation_1 should implement the decoder's first fully-connected layer
        self.decoder_activation_1 should implement the activation function of the decoder's first layer
        self.decoder_preactivation_2 should implement the decoder's second fully-connected layer

        Ask yourself: Why don't we need a self.decoder_activation_2?
        """
        super(DeepAutoEncoder, self).__init__()
        self.representation_size = representation_size
        self.compress = compress
        self.input_size = input_size

        ## First layer of encoder
        self.encoder_preactivation_1 = nn.Linear(self.input_size, self.compress)
        self.encoder_activation_1 = nn.ReLU()

        ## Second layer of encoder
        self.encoder_preactivation_2 = nn.Linear(
            self.compress, self.representation_size
        )
        self.encoder_activation_2 = nn.ReLU()

        ## First layer of decoder
        self.decoder_preactivation_1 = nn.Linear(
            self.representation_size, self.compress
        )
        self.decoder_activation_1 = nn.ReLU()

        ## Second layer of decoder
        self.decoder_preactivation_2 = nn.Linear(self.compress, self.input_size)
        ## END YOUR CODE

    def encode(self, x):
        """
        Runs the encoder of the network. Implement this function to define how the
        encoder should process and input x, which implicitly sets how the encoder
        modules saved in __init__() should be chained together

        x: a single batch of input data
        """

        x = self.encoder_preactivation_1(x)
        x = self.encoder_activation_1(x)
        x = self.encoder_preactivation_2(x)
        x = self.encoder_activation_2(x)

        return x
        ## END YOUR CODE

    def decode(self, a):
        """
        Runs the encoder of the network. Implement this function to define how the
        decoder should process a latent representation (output of the encoder) a,
        which implicitly sets how the decoder modules saved in __init__() should be chained together

        a: the output of encode(x)
        """
        a = self.decoder_preactivation_1(a)
        a = self.decoder_activation_1(a)
        a = self.decoder_preactivation_2(a)

        return a
        ## END YOUR CODE

    def forward(self, x):
        """
        The forward pass defines how to process an input x. This implicitly sets how
        the modules saved in __init__() should be chained together.

        Every PyTorch module has a forward() function, and when defining our own
        modules like we're doing here, we're required to define its forward().
        Here, we'll use your encode() and decode() functions to define forward().

        x: a single batch of input data
        """
        z = self.encode(x)
        x_prime = self.decode(z)
        return x_prime


################################## TRAIN ##################################
# Set compress size
compress = 200

# Set representation size
representation_size = 32

# Train for 50 epochs.
epochs = 50

# check if running on CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize a simple autoencoder
deep_autoencoder = DeepAutoEncoder(compress, representation_size).to(device)

# We'll use the Adam optimization of gradient descent.
optimizer = torch.optim.Adam(deep_autoencoder.parameters())

# Train your autoencoder
for epoch in range(1, epochs + 1):
    train_loss = train(deep_autoencoder, device, train_loader, optimizer)
    val_loss = test(deep_autoencoder, device, val_loader)
    print(
        "Train Epoch: {:02d} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
            epoch, train_loss, val_loss
        )
    )

# Test your autoencoder
print("\nReport this number on the course website:")
print("Test Loss:", test(deep_autoencoder, device, test_loader))


################################## VISUALIZE ##################################
# Encode and decode some digits
# Note that we take them from the *test* set
x_test = torch.tensor(test_dataset.data).to(device)
encoded_imgs = deep_autoencoder.encode(x_test.float())
decoded_imgs = deep_autoencoder.decode(encoded_imgs).cpu().detach().numpy()

# Display the original and encoded images
n = 10  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    ax.set_title("Original")
    plt.imshow(x_test[i].reshape(28, 28).cpu())
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    ax.set_title("Reconstructed")
    plt.imshow(np.clip(decoded_imgs[i].reshape(28, 28), 0.0, 1.0))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
