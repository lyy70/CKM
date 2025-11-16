import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, bn, decoder):
        super(CNN, self).__init__()
        self.bn = bn
        self.decoder = decoder

    def forward(self, x):
        b = self.bn(x)
        outputs = self.decoder(b)

        return outputs
