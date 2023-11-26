import torch
from torch import nn


class MLP(nn.Module):

    def __init__(self, in_dim, hidden_dims=[512], out_dim=1):
        super(MLP, self).__init__()

        assert len(hidden_dims) >= 1

        self.lin_in = nn.Linear(in_dim, hidden_dims[0])
        self.lin_hid = nn.ModuleList([nn.Linear(in_d, out_d) for in_d, out_d in zip(hidden_dims[:-1], hidden_dims[1:])])
        self.lin_out = nn.Linear(hidden_dims[-1], out_dim)


    def forward(self, X):

        X = self.lin_in(X)
        X = torch.relu(X)

        for hidden_layer in self.lin_hid:
            X = hidden_layer(X)
            X = torch.relu(X)

        X = self.lin_out(X)

        return X
