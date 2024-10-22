# src/probe_visualize.py
import torch
from transformer_model import CharTransformer
from autoencoder_model import AutoEncoder
from data_preprocess import load_data
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import umap
import hdbscan


def load_models(transformer_ckpt, autoencoder_ckpt, vocab_size, device):
    # Load Transformer
    transformer = CharTransformer(vocab_size)
    transformer.load_state_dict(torch.load(transformer_ckpt, map_location=device))
    transformer.to(device)
    transformer.eval()

    # Load Autoencoder
    autoencoder = AutoEncoder(input_dim=128, latent_dim=512, l1_coeff=1e-3)
    autoencoder.load_state_dict(torch.load(autoencoder_ckpt, map_location=device))
    autoencoder.to(device)
    autoencoder.eval()

    return transformer, autoencoder


def probe_transformer(transformer, autoencoder, input_text, char2idx, idx2char, device):
    transformer.eval()
    autoencoder.eval()
    input_ids = [char2idx.get(ch, char2idx[" "]) for ch in input_text]
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(
        0
    )  # (1, seq_length)

    with torch.no_grad():
        output, transformer_outputs = transformer(input_tensor)
        last_output = transformer_outputs[:, -1, :]  # (1, d_model)
        reconstructed, latent = autoencoder(last_output)

    latent_np = latent.cpu().numpy().flatten()
    reconstructed_np = reconstructed.cpu().numpy().flatten()

    return latent_np, reconstructed_np


def visualize_latent_space(latent_vectors, labels=None):
    # UMAP Embedding
    umap_embedding = umap.UMAP(
        n_neighbors=15, metric="cosine", min_dist=0.05
    ).fit_transform(latent_vectors)

    # HDBSCAN Clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric="euclidean")
    clusters = clusterer.fit_predict(umap_embedding)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        umap_embedding[:, 0], umap_embedding[:, 1], c=clusters, cmap="tab20", alpha=0.6
    )
    plt.colorbar(scatter, label="Cluster ID")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.title("UMAP Projection of Autoencoder Latent Space with HDBSCAN Clusters")
    plt.show()


if __name__ == "__main__":
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data to get vocab
    dataloader, dataset = load_data(
        "../data/harry_potter.txt", seq_length=1024, batch_size=64, shuffle=False
    )
    vocab_size = dataset.vocab_size
    char2idx = dataset.char2idx
    idx2char = dataset.idx2char

    # Load Models
    transformer_ckpt = (
        "../checkpoints/transformer/transformer_epoch_100000.pt"  # Adjust path
    )
    autoencoder_ckpt = (
        "../checkpoints/autoencoder/autoencoder_epoch_100000.pt"  # Adjust path
    )
    transformer, autoencoder = load_models(
        transformer_ckpt, autoencoder_ckpt, vocab_size, device
    )
    print("Models loaded.")

    # Example Probing Inputs
    test_inputs = [
        "Harry looked at the map and thought about",
        "She asked, 'What is the meaning of life",
        "The castle was dark and quiet, except for the sound of",
        "Ron exclaimed, 'This is incredible!",
        "Hermione whispered, 'We need to find the secret passage'",
    ]

    latent_vectors = []
    for text in test_inputs:
        latent, reconstructed = probe_transformer(
            transformer, autoencoder, text, char2idx, idx2char, device
        )
        latent_vectors.append(latent)
        print(f"Input Text: {text}")
        print(f"Latent Vector: {latent}")
        print("-" * 50)

    # Visualize Latent Vectors
    latent_np = np.array(latent_vectors)
    visualize_latent_space(latent_np)
