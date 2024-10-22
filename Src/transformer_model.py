# src/transformer_model.py
import torch
import torch.nn as nn


class CharTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_head=4,
        d_mlp=512,
        num_layers=1,
        dropout=0.1,
        max_seq_length=1024,
    ):
        super(CharTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_mlp,
            dropout=dropout,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        batch_size, seq_length = x.size()
        positions = (
            torch.arange(0, seq_length, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, seq_length)
        )
        x = self.embed(x) + self.pos_embed(positions)
        x = self.transformer(x)  # (batch_size, seq_length, d_model)
        out = self.fc_out(x)  # (batch_size, seq_length, vocab_size)
        return out, x  # Returning transformer output for activation collection

    def generate(
        self, start_text, char2idx, idx2char, generation_length=100, device="cuda"
    ):
        self.eval()
        input_ids = [char2idx.get(ch, char2idx[" "]) for ch in start_text]
        input_tensor = torch.tensor(
            input_ids, dtype=torch.long, device=device
        ).unsqueeze(
            0
        )  # (1, seq_length)
        generated = start_text

        for _ in range(generation_length):
            with torch.no_grad():
                output, _ = self.forward(input_tensor)
                next_token_logits = output[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).item()
                generated += idx2char[next_token]
                input_tensor = torch.cat(
                    [input_tensor, torch.tensor([[next_token]], device=device)], dim=1
                )
        return generated


if __name__ == "__main__":
    # Example usage
    from data_preprocess import load_data

    dataloader, dataset = load_data(
        "../data/harry_potter.txt", seq_length=1024, batch_size=64
    )
    model = CharTransformer(vocab_size=dataset.vocab_size)
    print(model)
