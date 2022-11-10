"""
Stuff that is used by multiple files.
"""

import torch

class MLP(torch.nn.Module): # only 1 layer
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.dropout = torch.nn.Dropout(dropout) if dropout else None
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
        self.act = torch.nn.GELU()

    def forward(self, x):
        x = self.act(self.linear1(x))
        if self.dropout:
            x = self.dropout(x)
        x = self.linear2(x)
        return x
