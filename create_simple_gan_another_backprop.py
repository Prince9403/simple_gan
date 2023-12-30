import datetime
import random

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision.datasets


import gan_base_classes as gbc


if __name__ == "__main__":

    image_size = (28, 28)
    input_noise_shape = 64
    batch_size = 8
    num_epochs = 60
    num_plots = 4

    random.seed(42)
    np.random.seed(19)
    # torch.manual_seed(19)
    # torch.cuda.manual_seed(0)

    # torch.backends.cudnn.deterministic = True

    font = {'weight': 'bold', 'size': 20}
    matplotlib.rc('font', **font)

    # generator = gbc.SimpleGenerator(input_dim=input_noise_shape, inner_dim_0=15, inner_dim_1=15, output_size=(28, 28))
    generator = gbc.Generator_6Layers(input_dim=input_noise_shape, inner_dim_0=40, inner_dim_1=60,
                                first_conv_size=(60, 60), out_channels_0=5, kernel_size_0=(10, 10),
                                out_channels_1=3, kernel_size_1=(7, 7), output_size=image_size)
    discriminator = gbc.SimpleDiscriminator(input_size=image_size, hidden_size=10)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    generator.to(device)
    discriminator.to(device)

    initial_transform = torchvision.transforms.ToTensor()

    train_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.MNIST(root="mnist_data", train=True, download=True, transform=initial_transform),
        batch_size=batch_size,
        shuffle=True
    )

    print(f"{datetime.datetime.now()} Started training")

    loss_fn = torch.nn.BCELoss(reduction='mean')

    optimizer_gen = torch.optim.Adam(params=generator.parameters(), lr=0.0002)
    optimizer_disc = torch.optim.Adam(params=discriminator.parameters(), lr=0.0002)

    generator_losses = []
    discriminator_losses = []

    for i in range(num_epochs):
        curr_generator_loss = 0.0
        curr_discriminator_loss = 0.0

        num_correct_true_for_epoch = 0
        num_correct_fake_for_epoch = 0
        num_items_epoch = 0
        num_batches = 0

        for j, (batch, y) in enumerate(train_loader):
            batch = batch.to(device)

            x = torch.rand(size=[batch_size, input_noise_shape], device=device)
            img_gen = generator(x)

            fake_input = discriminator(img_gen.data)
            target_discriminator_fake = torch.zeros((len(fake_input), 1))
            loss_discriminator_fake = loss_fn(fake_input, target_discriminator_fake)

            true_input = discriminator(batch)
            target_discriminator_true = torch.ones((len(true_input), 1))
            loss_discriminator_true = loss_fn(true_input, target_discriminator_true)

            loss_discriminator = loss_discriminator_fake + loss_discriminator_true
            curr_discriminator_loss += loss_discriminator.item()

            # train discriminator
            optimizer_disc.zero_grad()
            loss_discriminator.backward()

            optimizer_disc.step()

            x = torch.rand(size=[batch_size, input_noise_shape], device=device)
            img_gen = generator(x)

            fake_input = discriminator(img_gen)
            target_generator_fake = torch.ones((len(fake_input), 1))
            loss_generator = loss_fn(fake_input, target_generator_fake)
            curr_generator_loss += loss_generator.item()

            # train generator.
            optimizer_gen.zero_grad()
            loss_generator.backward()
            optimizer_gen.step()

            num_correct_true = (true_input > 0.5).float().sum().item()
            num_correct_fake = (fake_input < 0.5).float().sum().item()

            num_correct_true_for_epoch += num_correct_true
            num_correct_fake_for_epoch += num_correct_fake
            num_items_epoch += len(batch)

            num_batches += 1

        acc_true = num_correct_true_for_epoch / num_items_epoch
        acc_fake = num_correct_fake_for_epoch / num_items_epoch

        generator_losses.append(curr_generator_loss / num_batches)
        discriminator_losses.append(curr_discriminator_loss / num_batches)

        print(f"{datetime.datetime.now()} Epoch {i + 1} out of {num_epochs} passed, "
              f"generator loss: {curr_generator_loss:.4f}, "
              f"discriminator loss: {curr_discriminator_loss:.4f}, "
              f"accuracy on true items: {acc_true:.4f}, "
              f"accuracy on fake items: {acc_fake:.4f}, "
              f"num images per epoch: {num_items_epoch}"
              )

    generator.to(device=torch.device('cpu'))
    generator.eval()

    for i in range(num_plots):
        plt.subplot(1, num_plots, i + 1)
        x = torch.rand(size=[1, input_noise_shape])
        img_to_plot = generator(x).detach().numpy()
        # img_to_plot = 255.0 * img_to_plot
        img_to_plot = img_to_plot.reshape(image_size)
        plt.imshow(img_to_plot, cmap='autumn')
    plt.show()

    plt.plot(generator_losses, label="Generator losses")
    plt.plot(discriminator_losses, label="Discriminator losses")
    plt.legend()
    plt.grid()
    plt.title(f"Losses")
    plt.savefig(f"losses_epochs_{num_epochs}.png")
    plt.close()

    generator_scripted = torch.jit.script(generator)
    generator_scripted.save(f"generator_mnist_epochs_{num_epochs}.pt")
