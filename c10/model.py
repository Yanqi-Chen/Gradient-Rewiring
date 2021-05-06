import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import functional, layer, surrogate, neuron
from torchvision import transforms

class Cifar10Net(nn.Module):
    def __init__(self, T=8, v_threshold=1.0, v_reset=0.0, tau=2.0, surrogate_function=surrogate.ATan()):
        super().__init__()

        self.train_times = 0
        self.epochs = 0
        self.max_test_acccuracy = 0
        self.T = T

        self.static_conv = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )

        self.conv = nn.Sequential(
            neuron.LIFNode(v_threshold=v_threshold, v_reset=v_reset, tau=tau, surrogate_function=surrogate_function, detach_reset=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            neuron.LIFNode(v_threshold=v_threshold, v_reset=v_reset, tau=tau, surrogate_function=surrogate_function, detach_reset=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            neuron.LIFNode(v_threshold=v_threshold, v_reset=v_reset, tau=tau, surrogate_function=surrogate_function, detach_reset=True),

            nn.MaxPool2d(2, 2),  # 16 * 16

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            neuron.LIFNode(v_threshold=v_threshold, v_reset=v_reset, tau=tau, surrogate_function=surrogate_function, detach_reset=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            neuron.LIFNode(v_threshold=v_threshold, v_reset=v_reset, tau=tau, surrogate_function=surrogate_function, detach_reset=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            neuron.LIFNode(v_threshold=v_threshold, v_reset=v_reset, tau=tau, surrogate_function=surrogate_function, detach_reset=True),

            nn.MaxPool2d(2, 2)  # 8 * 8
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.5),
            
            nn.Linear(256 * 8 * 8, 128 * 4 * 4, bias=False),
            neuron.LIFNode(v_threshold=v_threshold, v_reset=v_reset, tau=tau, surrogate_function=surrogate_function, detach_reset=True),

            nn.Linear(128 * 4 * 4, 100, bias=False),
            neuron.LIFNode(v_threshold=v_threshold, v_reset=v_reset, tau=tau, surrogate_function=surrogate_function, detach_reset=True)
        )
        self.boost = nn.AvgPool1d(10, 10)

    def forward(self, x):
        x = self.static_conv(x)
        out_spikes_counter = self.boost(self.fc(self.conv(x)).unsqueeze(1)).squeeze(1)
        for _ in range(1, self.T):
            out_spikes_counter += self.boost(self.fc(self.conv(x)).unsqueeze(1)).squeeze(1)

        return out_spikes_counter

