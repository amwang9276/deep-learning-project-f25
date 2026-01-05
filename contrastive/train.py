from typing import *
import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm.auto import tqdm
from IPython.display import display
import matplotlib.pyplot as plt
import torchvision

from data import Dataset
from autoencoder.autoencoder import Encoder, Decoder
from visualize import nn_visualize

device = torch.device("cpu")


def train_contrastive(transforms: List, latent_dim: int, *, tau: float = 0.07):
    r"""
    Train encoder with `latent_dim` latent dimensions according
    to the **contrastive** objective described above using temperature
    `tau`.

    Implementation should follow notes above (including negative sampling
    from batch).

    The postive pairs are generated using random augmentations
    specified in `transform`.

    Returns the trained encoder.
    """

    enc = Encoder(latent_dim, normalize=True).to(device)

    optim = torch.optim.Adam(enc.parameters(), lr=2e-4)

    dataset = Dataset(
        "train", torchvision.transforms.Compose(transforms), num_samples=2
    )
    print("Visualize dataset")
    display.display(dataset.visualize(1, 10))
    plt.close("all")
    display.display(dataset.visualize(2, 10))
    plt.close("all")
    display.display(dataset.visualize(4, 10))
    plt.close("all")
    display.display(dataset.visualize(5, 10))
    plt.close("all")
    display.display(dataset.visualize(12, 10))
    plt.close("all")
    display.display(dataset.visualize(15, 10))
    plt.close("all")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=True, pin_memory=True
    )
    num_epochs = 15

    for epoch in range(num_epochs):
        for batch1, batch2 in tqdm(dataloader, desc=f"Epoch {epoch} / {num_epochs}"):
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)
            # batch1: a batched image tensor of shape [B x 1 x 64 x 28]
            # batch2: a batched image tensor of shape [B x 1 x 28 x 28]

            # For each i, p
            #   Positive pairs are (batch1[i], batch2[i])
            #   Negative pairs are (batch1[i], batch2[j]), j != i.

            logits = (
                enc(batch1) @ enc(batch2).T / tau
            )  # [B x latent_dim] [latent_dim x B] => [B x B]
            loss = F.cross_entropy(logits, torch.arange(logits.shape[0], device=device))

            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f"[Contrastive] epoch {epoch: 4d}   loss = {loss.item():.4g}")

    return enc
