import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import LayerNorm


class PerformerAttentionLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_features: int = None,
    ) -> None:
        super(PerformerAttentionLayer, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_features = num_features or hidden_dim

        self.key_layer = nn.Linear(input_dim, hidden_dim, bias=False)
        self.query_layer = nn.Linear(input_dim, hidden_dim, bias=False)
        self.value_layer = nn.Linear(input_dim, hidden_dim, bias=False)
        self.output_layer = nn.Linear(hidden_dim, input_dim, bias=False)

        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.norm1 = LayerNorm(input_dim)
        self.norm2 = LayerNorm(input_dim)

        # Random feature generation for FAVOR+
        self.register_buffer("random_matrix", torch.randn(hidden_dim, num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)

        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        queries = self.query_layer(x)
        keys = self.key_layer(x)
        values = self.value_layer(x)

        q_prime = torch.matmul(queries, self.random_matrix)
        k_prime = torch.matmul(keys, self.random_matrix)
        # Check for infinities and replace them with a large but finite number
        q_prime[torch.isinf(q_prime)] = 1e10  # or another very high but finite value
        k_prime[torch.isinf(k_prime)] = 1e10

        # Then perform the max subtraction
        max_q = torch.max(q_prime, dim=-1, keepdim=True)[0]
        max_k = torch.max(k_prime, dim=-1, keepdim=True)[0]
        q_prime -= max_q
        k_prime -= max_k

        QK = torch.matmul(torch.exp(q_prime), torch.exp(k_prime).t())
        attention_scores = QK / (QK.sum(dim=-1, keepdim=True) + 1e-8)

        out = torch.matmul(attention_scores, values)
        out = self.output_layer(out)
        return out


class PerformerGraphTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        num_features: int,
    ):
        super(PerformerGraphTransformer, self).__init__()
        self.layers = nn.ModuleList(
            [
                PerformerAttentionLayer(input_dim, hidden_dim, num_features)
                for _ in range(num_layers)
            ]
        )
        self.final_layer = nn.Linear(input_dim, output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        for layer in self.layers:
            x = layer(x)

        out = self.final_layer(x)
        return F.log_softmax(out, dim=-1)
