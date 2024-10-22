# src/autoencoder_model.py
import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim=128, latent_dim=512, l1_coeff=1e-3):
        super(AutoEncoder, self).__init__()
        self.l1_coeff = l1_coeff

        # Pre-encoder bias initialized to geometric median (approximated as zeros for simplicity)
        self.pre_encoder_bias = nn.Parameter(
            torch.zeros(latent_dim), requires_grad=True
        )

        # Encoder
        self.encoder = nn.Sequential(nn.Linear(input_dim, latent_dim), nn.ReLU())

        # Decoder
        self.decoder = nn.Sequential(nn.Linear(latent_dim, input_dim), nn.ReLU())

    def forward(self, x):
        # Subtract decoder bias (implemented as part of the decoder layer)
        # Add pre-encoder bias
        encoded = self.encoder(x) + self.pre_encoder_bias
        latent = torch.relu(encoded)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


if __name__ == "__main__":
    # Example usage
    autoencoder = AutoEncoder(input_dim=128, latent_dim=512, l1_coeff=1e-3)
    print(autoencoder)
