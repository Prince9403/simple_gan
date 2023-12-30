import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision


initial_transform = torchvision.transforms.ToTensor()

train_loader = torch.utils.data.DataLoader(
    dataset=torchvision.datasets.MNIST(root="mnist_data", train=True, download=True, transform=initial_transform),
    batch_size=16,
    shuffle=True
)
