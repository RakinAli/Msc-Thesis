# src/train_autoencoder.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from autoencoder_model import AutoEncoder


def neuron_resampling(
    autoencoder, dataloader, device, step, resample_steps, batch_size=8192
):
    """
    Perform neuron resampling at specified training steps.
    """
    if step in resample_steps:
        print(f"Resampling neurons at step {step}")
        # Identify dead neurons (zero activations)
        # Here, we approximate by checking the bias
        # More sophisticated methods can be implemented
        dead_neurons = (
            autoencoder.encoder[0].weight.data.abs().sum(dim=1) < 1e-6
        ).nonzero(as_tuple=True)[0]
        if len(dead_neurons) == 0:
            print("No dead neurons to resample.")
            return

        # Assign new weights to dead neurons
        for neuron in dead_neurons:
            # Initialize encoder weights for dead neuron
            nn.init.kaiming_uniform_(
                autoencoder.encoder[0].weight[neuron].unsqueeze(0), a=math.sqrt(5)
            )
            # Initialize pre-encoder bias
            autoencoder.pre_encoder_bias[neuron].data.zero_()
            # Initialize decoder weights for dead neuron
            nn.init.kaiming_uniform_(
                autoencoder.decoder[0].weight[:, neuron].unsqueeze(1), a=math.sqrt(5)
            )

        print(f"Resampled {len(dead_neurons)} neurons.")


def train_autoencoder(
    autoencoder,
    dataloader,
    criterion,
    optimizer,
    device,
    total_steps,
    checkpoint_path,
    resample_steps,
    log_interval=1000,
):
    autoencoder.to(device)
    autoencoder.train()
    step = 0

    for epoch in range(1, (total_steps // len(dataloader)) + 2):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Autoencoder Epoch {epoch}"):
            if step >= total_steps:
                print("Reached total training steps.")
                return

            batch_x = batch[0].to(device, non_blocking=True)
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

            # Neuron Resampling
            if step in resample_steps:
                neuron_resampling(autoencoder, dataloader, device, step, resample_steps)

            if step % log_interval == 0:
                print(f"Step {step}, Loss: {total_loss.item():.6f}")
                # Save intermediate checkpoints
                torch.save(
                    autoencoder.state_dict(),
                    os.path.join(checkpoint_path, f"autoencoder_step_{step}.pt"),
                )

        avg_loss = epoch_loss / len(dataloader)
        print(f"Autoencoder Epoch {epoch} Average Loss: {avg_loss:.6f}")
        # Save checkpoint at end of epoch
        torch.save(
            autoencoder.state_dict(),
            os.path.join(checkpoint_path, f"autoencoder_epoch_{epoch}.pt"),
        )

    print("Autoencoder Training completed.")


def resample_neurons(autoencoder, loss_weights, num_neurons, device):
    """
    Resample dead neurons based on loss weights.
    This is a simplified version and can be expanded based on specific needs.
    """
    # Placeholder for resampling logic
    pass


if __name__ == "__main__":
    import math

    # Hyperparameters
    input_dim = 128
    latent_dim = 512
    l1_coeff = 1e-3
    batch_size = 8192
    total_steps = 100000  # Adjust based on dataset size and GPU memory
    learning_rate = 3e-4
    checkpoint_path = "../checkpoints/autoencoder"
    os.makedirs(checkpoint_path, exist_ok=True)
    resample_steps = [25000, 50000, 75000, 100000]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Activations
    activation_path = "../activations/mlp_activations.pt"
    activations = torch.load(activation_path)
    print(f"Loaded activations shape: {activations.shape}")

    # Create DataLoader without replacement (shuffle=False)
    dataset = torch.utils.data.TensorDataset(activations)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
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
        resample_steps,
        log_interval=1000,
    )
