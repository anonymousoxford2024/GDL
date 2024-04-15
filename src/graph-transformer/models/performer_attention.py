import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LayerNorm


class PerformerAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, features_dim):
        super(PerformerAttention, self).__init__()

        self.query_layer = nn.Linear(input_dim, hidden_dim, bias=False)
        self.key_layer = nn.Linear(input_dim, hidden_dim, bias=False)
        self.value_layer = nn.Linear(input_dim, hidden_dim, bias=False)
        self.output_layer = nn.Linear(hidden_dim, input_dim, bias=False)

        self.features_dim = features_dim  # Dimensionality of the random features
        self.create_random_projection_matrices(hidden_dim, features_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.norm1 = LayerNorm(input_dim)
        self.norm2 = LayerNorm(input_dim)

    def create_random_projection_matrices(self, hidden_dim, features_dim):
        # Create orthogonal random projection matrices for the keys and queries
        self.R = torch.randn(hidden_dim, features_dim) / (features_dim ** 0.5)

    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x

    def attention(self, x):
        queries = self.query_layer(x)
        keys = self.key_layer(x)
        values = self.value_layer(x)

        # Project queries and keys using random features
        Q_prime = torch.matmul(queries, self.R)
        K_prime = torch.matmul(keys, self.R)

        # Compute the dot products and softmax for attention scores
        attention_scores = torch.matmul(Q_prime, K_prime.transpose(-2, -1))
        attention_scores = F.softmax(attention_scores, dim=-1)

        # Apply attention scores to values
        out = torch.matmul(attention_scores, values)
        out = self.output_layer(out)
        return out


class PerformerGraphTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, features_dim=64):
        super(PerformerGraphTransformer, self).__init__()
        self.layers = nn.ModuleList(
            [PerformerAttention(input_dim, hidden_dim, features_dim) for _ in range(num_layers)]
        )
        self.final_layer = nn.Linear(input_dim, output_dim)

    def forward(self, data):
        x = data.x
        for layer in self.layers:
            x = layer(x)
        out = self.final_layer(x)
        return F.log_softmax(out, dim=-1)
