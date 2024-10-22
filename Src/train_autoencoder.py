# src/train_autoencoder.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from autoencoder_model import AutoEncoder


def train_autoencoder(
    autoencoder,
    dataloader,
    criterion,
    optimizer,
    device,
    total_steps,
    checkpoint_path,
    log_interval=1000,
):
    autoencoder.to(device)
    autoencoder.train()
    step = 0

    for epoch in range(1, (total_steps // len(dataloader)) + 2):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Autoencoder Epoch {epoch}"):
            batch_x = batch[0].to(device)  # Assuming dataloader returns a tuple
            optimizer.zero_grad()
            reconstructed, latent = autoencoder(batch_x)
            loss = criterion(reconstructed, batch_x)
            # Add L1 regularization
            l1_loss = autoencoder.l1_coeff * torch.mean(torch.abs(latent))
            total_loss = loss + l1_loss
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            step += 1

            if step % log_interval == 0:
                print(f"Step {step}, Loss: {total_loss.item():.6f}")
                # Optionally, save intermediate checkpoints
                torch.save(
                    autoencoder.state_dict(),
                    os.path.join(checkpoint_path, f"autoencoder_step_{step}.pt"),
                )

            if step >= total_steps:
                print("Reached total training steps.")
                return

        avg_loss = epoch_loss / len(dataloader)
        print(f"Autoencoder Epoch {epoch} Average Loss: {avg_loss:.6f}")
        # Save checkpoint at end of epoch
        torch.save(
            autoencoder.state_dict(),
            os.path.join(checkpoint_path, f"autoencoder_epoch_{epoch}.pt"),
        )

    print("Autoencoder Training completed.")


if __name__ == "__main__":
    # Hyperparameters
    input_dim = 128
    latent_dim = 64
    l1_coeff = 1e-3
    batch_size = 4096
    total_steps = 10000  # Adjust based on dataset size and GPU memory
    learning_rate = 3e-4
    checkpoint_path = "../checkpoints/autoencoder"
    os.makedirs(checkpoint_path, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Activations
    activation_path = "../activations/mlp_activations.pt"
    activations = torch.load(activation_path)
    print(f"Loaded activations shape: {activations.shape}")

    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(activations)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    print("DataLoader created.")

    # Autoencoder Model
    autoencoder = AutoEncoder(input_dim, latent_dim, l1_coeff)
    print(autoencoder)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    # Train Autoencoder
    train_autoencoder(
        autoencoder,
        dataloader,
        criterion,
        optimizer,
        device,
        total_steps,
        checkpoint_path,
    )
