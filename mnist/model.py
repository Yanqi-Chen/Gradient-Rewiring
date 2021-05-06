import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import functional, layer, surrogate, neuron
from torchvision import transforms

class MnistNet(nn.Module):
    def __init__(self, T=8, v_threshold=1.0, v_reset=0.0, tau=2.0, surrogate_function=surrogate.ATan()):
        super().__init__()

        self.train_times = 0
        self.epochs = 0
        self.max_test_acccuracy = 0
        self.T = T

        self.static_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 800, bias=False),
        )
        self.fc = nn.Sequential(
            neuron.LIFNode(v_threshold=v_threshold, v_reset=v_reset, tau=tau, surrogate_function=surrogate_function, detach_reset=True),

            nn.Linear(800, 10, bias=False),
            neuron.LIFNode(v_threshold=v_threshold, v_reset=v_reset, tau=tau, surrogate_function=surrogate_function, detach_reset=True)
        )

    def forward(self, x):
        x = self.static_fc(x)
        out_spikes_counter = self.fc(x)
        for _ in range(1, self.T):
            out_spikes_counter += self.fc(x)

        return out_spikes_counter

