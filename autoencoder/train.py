import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm.auto import tqdm

from data import Dataset
from autoencoder.autoencoder import Encoder, Decoder
from visualize import nn_visualize

device = torch.device("cpu")


def train_autoencoder(latent_dim: int):
    r"""
    Train encoder and decoder networks with `latent_dim` latent dimensions according
    to the autoencoder objective (i.e., MSE reconstruction).

    Returns the trained encoder and decoder.
    """
    enc = Encoder(latent_dim).to(device)
    dec = Decoder(latent_dim).to(device)

    optim = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=2e-4)

    dataset = Dataset("train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    num_epochs = 30

    for epoch in tqdm(range(num_epochs), desc=f"{num_epochs} epochs total"):
        for (batch,) in dataloader:
            batch = batch.to(device)
            # batch: a batched image tensor of shape [B x 3 x 64 x 64]

            # FIXME
            # loss = '???'
            loss = F.mse_loss(dec(enc(batch)), batch)

            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f"[Autoencoder] epoch {epoch: 4d}   loss = {loss.item():.4g}")

    return enc, dec


if __name__ == "__main__":
    ae_enc, ae_dec = train_autoencoder(128)
    fig = nn_visualize(Dataset("train"), ae_enc, desc="Autoencoder Encoder")
