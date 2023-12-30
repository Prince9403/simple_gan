import torch
from torch import nn


class SimpleGenerator(torch.nn.Module):
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

class Generator_6Layers(torch.nn.Module):
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
        print(f"Using ({m}, {n}) size for convolution of the 6th generator layer")
        self.layer_6 = torch.nn.Conv2d(in_channels=out_channels_1, out_channels=1,
                                       kernel_size=(m, n))
        self.act_6 = torch.nn.Sigmoid()

        z = self.act_6(self.layer_6(z))

        if (z.shape[2],z.shape[3]) != output_size:
            raise RuntimeError("Something bad with generator network dimensions")

    def forward(self, x):
        x = self.act_1(self.layer_1(x))
        x = self.act_2(self.layer_2(x))
        x = self.act_3(self.layer_3(x))
        x = torch.reshape(x, (x.shape[0], 1, self.first_conv_size[0], self.first_conv_size[1]))
        x = self.act_4(self.layer_4(x))
        x = self.act_5(self.layer_5(x))
        x = self.act_6(self.layer_6(x))
        return x


class SimpleDiscriminator(torch.nn.Module):
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
