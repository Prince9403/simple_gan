import datetime
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets

from torch import nn


class Generator(torch.nn.Module):
    def __init__(self, input_dim, inner_dim_0, inner_dim_1, output_size):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=input_dim, out_features=inner_dim_0)
        self.act_1 = nn.ReLU()
        self.layer_2 = nn.Linear(in_features=inner_dim_0, out_features=inner_dim_1)
        self.act_2 = nn.ReLU()
        self.layer_3 = nn.Linear(in_features=inner_dim_1, out_features=output_size[0] * output_size[1])
        self.act_3 = nn.Sigmoid()
        self.output_size = output_size


    def forward(self, x):
        x = self.act_1(self.layer_1(x))
        x = self.act_2(self.layer_2(x))
        x = self.act_3(self.layer_3(x))
        return torch.reshape(x, (x.shape[0], 1, self.output_size[0], self.output_size[1]))

class Discriminator(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3))
        self.act_1 = nn.ReLU()
        self.layer_2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(4, 4))
        self.act_2 = nn.ReLU()
        self.layer_3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(10, 10))
        self.act_3 = nn.ReLU()

        self.act_35 = torch.nn.Flatten()

        x = self.layer_1(torch.rand(size=[5, 1, input_size[0], input_size[1]]))
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.act_35(x)

        self.layer_4 = torch.nn.Linear(in_features=x.shape[1], out_features=hidden_size)
        self.act_4 = torch.nn.ReLU()

        self.layer_5 = torch.nn.Linear(in_features=hidden_size, out_features=1)
        self.act_5 = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.act_1(self.layer_1(x))
        x = self.act_2(self.layer_2(x))
        x = self.act_3(self.layer_3(x))
        x = self.act_35(x)
        x = self.act_4(self.layer_4(x))
        x = self.act_5(self.layer_5(x))
        return x


if __name__ == "__main__":
    input_noise_shape = 30

    generator = Generator(input_dim=input_noise_shape, inner_dim_0=15, inner_dim_1=15, output_size=(28, 28))
    discriminator = Discriminator(input_size=(28, 28), hidden_size=10)

    batch_size = 8
    num_epochs = 30

    random.seed(42)
    np.random.seed(19)
    torch.manual_seed(19)
    torch.cuda.manual_seed(0)

    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    generator.to(device)
    discriminator.to(device)

    initial_transform = torchvision.transforms.ToTensor()

    train_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.MNIST(root="mnist_data", train=True, download=True, transform=initial_transform),
        batch_size=16,
        shuffle=True
    )

    print(f"{datetime.datetime.now()} Started training")

    loss_fn = torch.nn.BCELoss(reduction='mean')

    optimizer_gen = torch.optim.Adam(params=generator.parameters(), lr=0.01)
    optimizer_disc = torch.optim.Adam(params=discriminator.parameters(), lr=0.01)

    for i in range(num_epochs):
        for batch, y in train_loader:
            batch = batch.to(device)
            batch = batch / 255.0

            x = torch.rand(size=[batch_size, input_noise_shape])
            x = x.to(device)
            img_gen = generator(x)

            fake_input = discriminator(img_gen)
            target_fake = torch.zeros((len(fake_input),1))
            loss = loss_fn(fake_input, target_fake)

            true_input = discriminator(batch)
            target_true = torch.ones((len(true_input), 1))
            loss = loss + loss_fn(true_input, target_true)

            optimizer_gen.zero_grad()
            optimizer_disc.zero_grad()

            loss.backward()

            for param in generator.parameters():
                param.grad = -param.grad

            optimizer_gen.step()
            optimizer_disc.step()

        print(f"{datetime.datetime.now()} Epoch {i + 1} out of {num_epochs} passed")


    x = torch.rand(size=[1, input_noise_shape])
    plt.imshow(int(255.0 * generator(x)), cmap='autumn')
    plt.show()
