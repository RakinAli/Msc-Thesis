# src/train_transformer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformer_model import CharTransformer
from data_preprocess import load_data


def train(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    total_steps,
    checkpoint_path,
    log_interval=1000,
):
    model.to(device)
    model.train()
    step = 0
    optimizer.zero_grad()

    for epoch in range(1, (total_steps // len(dataloader)) + 2):
        epoch_loss = 0
        for batch_x, batch_y in tqdm(dataloader, desc=f"Epoch {epoch}"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs, _ = model(batch_x)
            loss = criterion(outputs.view(-1, outputs.size(-1)), batch_y.view(-1))
            loss.backward()

            epoch_loss += loss.item()

            if (step + 1) % log_interval == 0:
                print(f"Step {step+1}, Loss: {loss.item():.4f}")
                # Optionally, save intermediate checkpoints
                torch.save(
                    model.state_dict(),
                    os.path.join(checkpoint_path, f"transformer_step_{step+1}.pt"),
                )

            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step >= total_steps:
                print("Reached total training steps.")
                return

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        # Save checkpoint at end of epoch
        torch.save(
            model.state_dict(),
            os.path.join(checkpoint_path, f"transformer_epoch_{epoch}.pt"),
        )

    print("Training completed.")


if __name__ == "__main__":
    # Hyperparameters
    seq_length = 1024
    batch_size = 64
    d_model = 128
    n_head = 4
    d_mlp = 512
    num_layers = 1
    dropout = 0.1
    total_steps = 100000  # Adjust to fit GPU memory and training time
    learning_rate = 3e-4
    checkpoint_path = "../checkpoints/transformer"
    os.makedirs(checkpoint_path, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    dataloader, dataset = load_data("../data/harry_potter.txt", seq_length, batch_size)
    vocab_size = dataset.vocab_size
    print(f"Vocab Size: {vocab_size}")

    # Model
    model = CharTransformer(vocab_size, d_model, n_head, d_mlp, num_layers, dropout)
    print(model)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    train(model, dataloader, criterion, optimizer, device, total_steps, checkpoint_path)
