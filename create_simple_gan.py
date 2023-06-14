import datetime
import random

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
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

class DeeperGenerator(torch.nn.Module):
    def __init__(self, input_dim, inner_dim_0, inner_dim_1, first_conv_size, out_channels_0, kernel_size_0,
                 out_channels_1, kernel_size_1, output_size):
        super().__init__()
        self.output_size = output_size

        self.first_conv_size = first_conv_size
        self.layer_1 = nn.Linear(in_features=input_dim, out_features=inner_dim_0)
        self.act_1 = nn.ReLU()
        self.layer_2 = nn.Linear(in_features=inner_dim_0, out_features=inner_dim_1)
        self.act_2 = nn.ReLU()
        self.layer_3 = nn.Linear(in_features=inner_dim_1, out_features=first_conv_size[0] * first_conv_size[1])
        self.act_3 = nn.ReLU()
        self.layer_4 = torch.nn.Conv2d(in_channels=1, out_channels=out_channels_0, kernel_size=kernel_size_0)
        self.act_4 = nn.ReLU()
        self.layer_5 = torch.nn.Conv2d(in_channels=out_channels_0, out_channels=out_channels_1,
                                       kernel_size=kernel_size_1)
        self.act_5 = nn.ReLU()

        z = torch.rand(size=(5, input_dim,))

        z = self.act_1(self.layer_1(z))
        z = self.act_2(self.layer_2(z))
        z = self.act_3(self.layer_3(z))
        z = torch.reshape(z, (z.shape[0], 1, self.first_conv_size[0], self.first_conv_size[1]))
        z = self.act_4(self.layer_4(z))
        z = self.act_5(self.layer_5(z))

        m = z.shape[2] + 1 - output_size[0]
        n = z.shape[3] + 1 - output_size[1]
        if (m <= 0) or (n <= 0):
            raise ValueError("Incorrect dimensions of the generator network")
        print(f"Using ({m}, {n}) size for convolution of  the 6th generator layer")
        self.layer_6 = torch.nn.Conv2d(in_channels=out_channels_1, out_channels=1,
                                       kernel_size=(m, n))
        self.act_6 = torch.nn.Sigmoid()

        z = self.act_6(self.layer_6(z))

        if (z.shape[2],z.shape[3]) != output_size:
            raise RuntimeError("SOmething bad with generator network dimensions")

    def forward(self, x):
        x = self.act_1(self.layer_1(x))
        x = self.act_2(self.layer_2(x))
        x = self.act_3(self.layer_3(x))
        x = torch.reshape(x, (x.shape[0], 1, self.first_conv_size[0], self.first_conv_size[1]))
        x = self.act_4(self.layer_4(x))
        x = self.act_5(self.layer_5(x))
        x = self.act_6(self.layer_6(x))
        return x


class Discriminator(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layer_1 = nn.Conv2d(in_channels=1, out_channels=7, kernel_size=(3, 3))
        self.act_1 = nn.ReLU()
        self.layer_2 = nn.Conv2d(in_channels=7, out_channels=4, kernel_size=(4, 4))
        self.act_2 = nn.ReLU()
        self.layer_3 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(10, 10))
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

    image_size = (28, 28)

    # generator = Generator(input_dim=input_noise_shape, inner_dim_0=15, inner_dim_1=15, output_size=(28, 28))
    generator = DeeperGenerator(input_dim=input_noise_shape, inner_dim_0=40, inner_dim_1=60,
                                first_conv_size=(60, 60), out_channels_0=5, kernel_size_0=(10, 10),
                                out_channels_1=3, kernel_size_1=(7, 7), output_size=image_size)
    discriminator = Discriminator(input_size=image_size, hidden_size=10)

    batch_size = 8
    num_epochs = 500
    num_plots = 3

    random.seed(42)
    np.random.seed(19)
    torch.manual_seed(19)
    torch.cuda.manual_seed(0)

    font = {'weight': 'bold', 'size': 20}
    matplotlib.rc('font', **font)

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

    loss_fn = torch.nn.BCELoss(reduction='sum')

    optimizer_gen = torch.optim.Adam(params=generator.parameters(), lr=0.01)
    optimizer_disc = torch.optim.Adam(params=discriminator.parameters(), lr=0.05)

    losses = []

    for i in range(num_epochs):
        curr_epoch_loss = 0.0
        for batch, y in train_loader:
            batch = batch.to(device)
            batch = batch / 255.0

            x = torch.rand(size=[batch_size, input_noise_shape], device=device)
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

            curr_epoch_loss += loss.item()

            for param in generator.parameters():
                param.grad = -param.grad

            optimizer_gen.step()
            optimizer_disc.step()

        losses.append(curr_epoch_loss)

        print(f"{datetime.datetime.now()} Epoch {i + 1} out of {num_epochs} passed")

    generator.to(device=torch.device('cpu'))
    generator.eval()

    for i in range(num_plots):
        plt.subplot(1, num_plots, i + 1)
        x = torch.rand(size=[1, input_noise_shape])
        img_to_plot = generator(x).detach().numpy()
        # img_to_plot = 255.0 * img_to_plot
        img_to_plot = img_to_plot.reshape((28, 28))
        plt.imshow(img_to_plot, cmap='autumn')
    plt.show()

    plt.plot(losses)
    plt.grid()
    plt.title(f"Losses")
    plt.show()
    plt.savefig(f"losses_epochs_{num_epochs}.png")

    generator_scripted = torch.jit.script(generator)
    generator_scripted.save(f"generator_mnist_epochs_{num_epochs}.pt")
