import numpy as np
import torch
import torchvision
from torchvision import datasets
import matplotlib.pyplot as plt

############### load data ###############
# Load MNIST dataset
mnist_train = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
mnist_val = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

# Keep variables for compatibility - create indices for train/val splits
train_indices = torch.arange(len(mnist_train))
val_indices = torch.arange(len(mnist_val))

# Keep data variable for compatibility - store MNIST images
data = {'imgs': torch.stack([img for img, _ in mnist_train] + [img for img, _ in mnist_val])}


class Dataset:
    r'''
    Our dataset object for loading MNIST dataset.
    '''

    def __init__(self, split: str, transform=None, num_samples: int = 1):
        r'''
        split (str): Whether to load training of validation images. Should be 'train' or 'val'.
        transform: Transformations on raw data, e.g., augmentations and/or normalization.
                   `to_tensor` and normalization is called automatically.
                   No need to explicitly pass in `ToTensor()` or `Normalize()`.
        num_samples (int):Number of transformed versions to return for each sample.
                           For autoencoder, this is 1. For contrastive, this is 2.
        '''
        self.split = split
        if split == 'train':
            self.dataset = mnist_train
            self.indices = train_indices
        else:
            assert split == 'val'
            self.dataset = mnist_val
            self.indices = val_indices
        self.num_samples = num_samples
        if transform is None:
            transform = lambda x: x
        self.transform = transform

    def get_augs(self, idx, num_samples):
        img, _ = self.dataset[idx]  # MNIST returns (image, label), we only need image
        # img is already a tensor in [0, 1] range from ToTensor()
        return tuple(self.transform(img).clamp(0, 1) for _ in range(num_samples))

    def __getitem__(self, idx):
        r'''
        Fetech the data at index `idx`
        '''
        return tuple(tensor.sub(0.5).div(0.5) for tensor in self.get_augs(idx, num_samples=self.num_samples))

    def visualize(self, idx, num_samples=None):
        r'''
        Visualize the image at index `idx` for `num_samples` times (default to `self.num_samples`).

        These samples will be different if `self.transform` is random.
        '''
        if num_samples is None:
            num_samples = self.num_samples
        f, axs = plt.subplots(1, num_samples, figsize=(1.2 * num_samples, 1.4))
        if num_samples == 1:
            axs = [axs]
        else:
            axs = axs.reshape(-1)
        for ax, tensor in zip(axs, self.get_augs(idx, num_samples)):
            ax.axis('off')
            # MNIST is grayscale, so we need to handle single channel
            if tensor.shape[0] == 1:
                ax.imshow(tensor.squeeze(0), cmap="gray")
            else:
                ax.imshow(tensor.permute(1, 2, 0))
        title = f'{self.split} dataset[{idx}]'
        if num_samples > 1:
            title += f'  ({num_samples} samples)'
        f.suptitle(title, fontsize=17, y=0.98)
        f.tight_layout(rect=[0, 0.03, 1, 0.9])
        return f

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    dataset = Dataset('train')
    _ = dataset.visualize(0)
    plt.show()