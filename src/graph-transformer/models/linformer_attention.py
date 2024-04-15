import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import LayerNorm


def _project(x, projector):
    return torch.transpose(projector(torch.transpose(x, 1, 0)), 1, 0)


class LinformerAttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, projection_dim, n_nodes):
        super(LinformerAttentionLayer, self).__init__()

        self.key_layer = nn.Linear(input_dim, hidden_dim, bias=False)
        self.query_layer = nn.Linear(input_dim, hidden_dim, bias=False)
        self.value_layer = nn.Linear(input_dim, hidden_dim, bias=False)

        # Projection matrices for keys and values
        self.key_projector = nn.Linear(n_nodes, projection_dim, bias=False)
        self.value_projector = nn.Linear(n_nodes, projection_dim, bias=False)

        self.output_layer = nn.Linear(hidden_dim, input_dim, bias=False)

        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.norm1 = LayerNorm(input_dim)
        self.norm2 = LayerNorm(input_dim)

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

        # Project keys and values to lower dimensions
        keys = _project(keys, self.key_projector)
        values = _project(values, self.value_projector)

        attention_scores = torch.matmul(queries, keys.transpose(0, 1))
        attention_scores = torch.softmax(attention_scores, dim=-1)

        out = torch.matmul(attention_scores, values)
        out = self.output_layer(out)
        return out


class LinformerGraphTransformer(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_layers, projection_dim, n_nodes
    ):
        super(LinformerGraphTransformer, self).__init__()
        self.layers = nn.ModuleList(
            [
                LinformerAttentionLayer(input_dim, hidden_dim, projection_dim, n_nodes)
                for _ in range(num_layers)
            ]
        )
        self.final_layer = nn.Linear(input_dim, output_dim)

    def forward(self, data):
        x = data.x
        for layer in self.layers:
            x = layer(x)

        out = self.final_layer(x)
        return F.log_softmax(out, dim=-1)
