import torch
import matplotlib.pyplot as plt
from typing import Optional
import torchvision

from data import Dataset
from autoencoder.autoencoder import Encoder, Decoder

device = torch.device("cpu")

@torch.no_grad()
def get_features(dataset: Dataset, encoder: Encoder, firstk: Optional[int] = None):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128)
    n = 0
    features = []
    for batches in dataloader:
        batch = batches[0]
        features.append(encoder(batch.to(device)))
        n += batch.shape[0]
        if firstk is not None and n >= firstk:
            break
    features = torch.cat(features)
    if firstk is not None:
        features = features[:firstk]
    return features

@torch.no_grad()
def nn_visualize(dataset: Dataset, encoder: Encoder, num_nn: int = 10, desc: str = ''):
    r'''
    For a given `dataset` object, visualize `encoder` by nearest neighbors.

    In particular, the function takes `8` samples from `dataset`, computes
    their `num_nn` nearest neighbor samples (w.r.t. l2 embedding distance induced by
    `encoder`), and produces a plot.

    `desc` is a string description that is used to set the title of the plot.
    '''
    all_features = get_features(dataset, encoder, firstk=24000)
    source_feat = all_features[:8]
    dists = (source_feat[:, None] - all_features).pow(2).sum(-1)
    knns = torch.topk(dists, num_nn, dim=-1, largest=False).indices.cpu()

    nrow = knns.shape[0]
    ncol = num_nn + 2
    f, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1, nrow * 1.1 + 1))
    for ax in axes.reshape(-1):
        ax.axis('off')

    axes[0, 0].set_title('Input Image', fontsize=15)

    for ax, imidx in zip(axes[:, 0], range(8)):
        img, _ = dataset.dataset[imidx]
        # MNIST is grayscale, handle single channel
        if img.shape[0] == 1:
            ax.imshow(img.squeeze(0), cmap="gray")
        else:
            ax.imshow(img.permute(1, 2, 0))

    for j in range(2, ncol):
        axes[0, j].set_title(f'NN {1 +j-2}', fontsize=15)

    for i in range(nrow):
        for j in range(2, ncol):
            img, _ = dataset.dataset[knns[i, j - 2]]
            # MNIST is grayscale, handle single channel
            if img.shape[0] == 1:
                axes[i, j].imshow(img.squeeze(0), cmap="gray")
            else:
                axes[i, j].imshow(img.permute(1, 2, 0))

    f.suptitle(f'{desc} Nesrest Neighbors (NNs) on {dataset.split.capitalize()} Set \n(near -> far, NN 1 is closest)', fontsize=20, y=0.98)
    f.tight_layout(rect=[0, 0.03, 1, 0.9])
    f.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.show()
    return f

def aug_b():
    interpolation = torchvision.transforms.InterpolationMode.NEAREST
    aug_B = [
        torchvision.transforms.Pad(24),
        torchvision.transforms.RandomRotation(degrees=(0, 360)),
        torchvision.transforms.RandomResizedCrop([28, 28], scale=(0.2, 0.6), ratio=(1, 1), interpolation=interpolation),
    ]

    dataset = Dataset('train', torchvision.transforms.Compose(aug_B), num_samples=2)
    print('Visualize augmentation B')
    for idx in [4, 5, 12]:
        _ = dataset.visualize(idx, 14)
    plt.show()

def aug_c():
    interpolation = torchvision.transforms.InterpolationMode.NEAREST
    aug_C = [
        torchvision.transforms.Pad(16),
        torchvision.transforms.RandomRotation(degrees=(0, 360)),
        torchvision.transforms.RandomApply(
            [
                torchvision.transforms.RandomAffine(degrees=0, translate=(0.3, 0.3), shear=(-80, 80, -80, 80), interpolation=interpolation),
                torchvision.transforms.RandomPerspective(distortion_scale=0.6,  p=1, interpolation=interpolation),
            ],
            p=0.7,
        ),
        torchvision.transforms.RandomResizedCrop([28, 28], scale=(0.4, 0.8), ratio=(0.2, 5), interpolation=interpolation),
        torchvision.transforms.RandomRotation(degrees=(0, 360)),
    ]


    dataset = Dataset('train', torchvision.transforms.Compose(aug_C), num_samples=2)
    print('Visualize augmentation C')
    for idx in [4, 5, 12]:
        _ = dataset.visualize(idx, 14)
    plt.show()

if __name__ == "__main__":
    # nn_visualize(
    #     dataset=Dataset('train'),
    #     encoder=lambda batch: torch.randn(batch.shape[0], 128, device=device),
    #     desc='Randomly Assigned Features',
    # )

    aug_c()