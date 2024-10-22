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
    max_contexts=1000000,
    activations_per_context=200,
):
    model.to(device)
    model.eval()
    activations = []
    contexts_collected = 0

    with torch.no_grad():
        for batch_x, _ in tqdm(dataloader, desc="Collecting Activations"):
            batch_size = batch_x.size(0)
            for i in range(batch_size):
                if contexts_collected >= max_contexts:
                    break
                context = batch_x[i].unsqueeze(0)  # (1, seq_length)
                _, transformer_outputs = model(context)  # (1, seq_length, d_model)
                # Sample 200 tokens from the context
                seq_length = transformer_outputs.size(1)
                if seq_length < activations_per_context:
                    sample_indices = torch.arange(seq_length)
                else:
                    sample_indices = torch.randperm(seq_length)[
                        :activations_per_context
                    ]
                sampled_activations = transformer_outputs[
                    0, sample_indices, :
                ]  # (activations_per_context, d_model)
                activations.append(sampled_activations.cpu())
                contexts_collected += 1

                if contexts_collected % 10000 == 0:
                    print(f"Collected {contexts_collected} contexts")

            if contexts_collected >= max_contexts:
                break

    activations = torch.cat(
        activations, dim=0
    )  # (max_contexts * activations_per_context, d_model)
    torch.save(activations, activation_save_path)
    print(f"Saved MLP activations to {activation_save_path}")


if __name__ == "__main__":
    # Hyperparameters
    seq_length = 1024
    batch_size = 64
    activation_save_path = "../activations/mlp_activations.pt"
    max_contexts = 1000000  # 1 million contexts
    activations_per_context = 200

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    dataloader, dataset = load_data(
        "../data/harry_potter.txt", seq_length, batch_size, shuffle=False
    )
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
        max_contexts,
        activations_per_context,
    )
