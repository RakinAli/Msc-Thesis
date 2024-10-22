# src/collect_activations.py
import os
import torch
from tqdm import tqdm
from transformer_model import CharTransformer
from data_preprocess import load_data


def collect_mlp_activations(
    model,
    dataloader,
    device,
    activation_save_path,
    max_prompts=200000,
    activations_per_prompt=200,
):
    model.to(device)
    model.eval()
    activations = []

    with torch.no_grad():
        prompt_count = 0
        for batch_x, _ in tqdm(dataloader, desc="Collecting Activations"):
            batch_x = batch_x.to(device)
            _, transformer_outputs = model(batch_x)  # (batch_size, seq_length, d_model)

            # Extract MLP activations
            # Assuming MLP is a single layer after self-attention
            # Modify if multiple MLP layers exist
            # For simplicity, using transformer_outputs as MLP activations
            # If you have access to MLP layer outputs, hook them here

            # Example: Suppose MLP is a separate layer, use hooks to capture activations
            # Here, we are simplifying by using transformer_outputs
            activations.append(transformer_outputs.cpu())

            prompt_count += batch_x.size(0)
            if prompt_count * activations_per_prompt >= max_prompts:
                break

    activations = torch.cat(activations, dim=0)  # (num_samples, d_model)
    torch.save(activations, activation_save_path)
    print(f"Saved MLP activations to {activation_save_path}")


if __name__ == "__main__":
    # Hyperparameters
    seq_length = 1024
    batch_size = 64
    activation_save_path = "../activations/mlp_activations.pt"
    max_prompts = 200000
    activations_per_prompt = 200  # Adjust as needed

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    dataloader, dataset = load_data("../data/harry_potter.txt", seq_length, batch_size)
    print("Data loaded.")

    # Load Transformer Model
    d_model = 128
    n_head = 4
    d_mlp = 512
    num_layers = 1
    dropout = 0.1
    vocab_size = dataset.vocab_size

    model = CharTransformer(vocab_size, d_model, n_head, d_mlp, num_layers, dropout)
    transformer_ckpt = (
        "../checkpoints/transformer/transformer_epoch_100000.pt"  # Adjust path
    )
    model.load_state_dict(torch.load(transformer_ckpt, map_location=device))
    print("Transformer model loaded.")

    # Collect Activations
    collect_mlp_activations(
        model,
        dataloader,
        device,
        activation_save_path,
        max_prompts,
        activations_per_prompt,
    )
