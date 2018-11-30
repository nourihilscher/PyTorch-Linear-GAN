import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import Discriminator, Generator

# Make directories for images and mnist dataset
os.makedirs('mnist_data', exist_ok=True)
os.makedirs('generated_images', exist_ok=True)

##### Terminal arguments #####
parser = argparse.ArgumentParser()
# Data parameters
parser.add_argument("--custom_image_path", type=str, default=None, help="path to own dataset of black and white images | if not specified MNIST will be downloaded and used.")
parser.add_argument("--image_size", type=int, default=28, help="height and width of image | image will be rescaled to that size")
parser.add_argument("--batch_size", type=int, default=128, help="number of images in one batch")
# Model parameters
parser.add_argument("--z_size", type=int, default=100, help="dimensionality of the latent vector z")
# Training parameters
parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--lr", type=float, default=0.0002, help="learning rate | adam optimizer")
parser.add_argument("--beta1", type=float, default=0.5, help="beta 1 value | adam optimizer")
parser.add_argument("--beta2", type=float, default=0.999, help="beta 2 value | adam optimizer")
parser.add_argument("--print_every", type=int, default=200, help="number of BATCHES after loss is printed and sample images saved")
parser.add_argument("--sample_size", type=int, default=8, help="number of samples")
parser.add_argument("--plot_every", type=int, default=5, help="number of EPOCHS after plot of loss will be saved")
parser.add_argument("--model_save_path", type=str, default="trained_gan", help="path where the model should be saved including the filename")

opt = parser.parse_args()
print(opt)

##### Import and transform data #####
# Resize data
transform = transforms.Compose([
    transforms.Resize(opt.image_size),
    transforms.Grayscale(),
    transforms.ToTensor()
])

if opt.custom_image_path != None:
    train_dataset = datasets.ImageFolder(opt.custom_image_path)
else:
    train_dataset = datasets.MNIST(root="mnist_data", train=True, download=True, transform=transform)

train_loader = DataLoader(train_dataset, opt.batch_size, shuffle=True)

def im_convert(tensor, rescale=False):
    tensor = tensor.view(1, opt.image_size, opt.image_size)
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    if rescale:
        # Rescale image from tanh output (1, -1) to rgb (0, 255)
        image = ((image + 1) * 255 / (2)).astype(np.uint8)
    return image

# Scale image values from -1 to 1 to be close to the output of the tanh function
def scale(x, feature_range=(-1, 1)):
    min, max = feature_range
    x = x * (max-min) + min
    return x

##### Models #####
D = Discriminator(image_size=opt.image_size)
G = Generator(input_features=opt.z_size, image_size=opt.image_size)

# If a gpu is available move all models to gpu
if torch.cuda.is_available():
    G = G.cuda()
    D = D.cuda()
    print("GPU available. Moved models to GPU.")
else:
    print("Training on CPU.")

# Loss functions
def real_loss(predictions, smooth=False):
    batch_size = predictions.shape[0]
    labels = torch.ones(batch_size)
    # Smooth labels for discriminator to weaken learning
    if smooth:
        labels = labels * 0.9
    # We use the binary cross entropy loss | Model has a sigmoid function
    criterion = nn.BCELoss()
    # Move models to GPU if available
    if torch.cuda.is_available():
        labels = labels.cuda()
        criterion = criterion.cuda()
    loss = criterion(predictions.squeeze(), labels)
    return loss

def fake_loss(predictions):
    batch_size = predictions.shape[0]
    labels = torch.zeros(batch_size)
    criterion = nn.BCELoss()
    # Move models to GPU if available
    if torch.cuda.is_available():
        labels = labels.cuda()
        criterion = criterion.cuda()
    loss = criterion(predictions.squeeze(), labels)
    return loss

def random_vector(batch_size, length):
    # Sample from a Gaussian distribution
    z_vec = torch.randn(batch_size, length).float()
    if torch.cuda.is_available():
        z_vec = z_vec.cuda()
    return z_vec

##### Training #####
# Optimizers
d_optimizer = optim.Adam(D.parameters(), lr=opt.lr, betas=[opt.beta1, opt.beta2])
g_optimizer = optim.Adam(G.parameters(), lr=opt.lr, betas=[opt.beta1, opt.beta2])

# Function to train a discriminator
def train_discriminator(generator, discriminator, optimizer, real_data, batch_size, z_size):
    # Reshape real_data to vector
    real_data = real_data.view(batch_size, -1)
    # Rescale real_data to range -1 - 1
    real_data = scale(real_data)
    
    # Reset gradients and set model to training mode
    optimizer.zero_grad()
    discriminator.train()
    
    # Train on real data
    real_data_logits = discriminator.forward(real_data)
    loss_real = real_loss(real_data_logits, smooth=True)
    # Generate fake data
    z_vec = random_vector(batch_size, z_size)
    fake_data = generator.forward(z_vec)
    # Train on fake data
    fake_data_logits = discriminator.forward(fake_data)
    loss_fake = fake_loss(fake_data_logits)
    # Calculate total loss
    total_loss = loss_real + loss_fake
    total_loss.backward()
    optimizer.step()
    
    return total_loss

# Function to train a generator
def train_generator(generator, discriminator, optimizer, batch_size, z_size):
    # Reset gradients and set model to training mode
    optimizer.zero_grad()
    generator.train()
    # Generate fake data
    z_vec = random_vector(batch_size, z_size)
    fake_data = generator.forward(z_vec)
    # Train generator with output of discriminator
    discriminator_logits = discriminator.forward(fake_data)
    # Reverse labels
    loss = real_loss(discriminator_logits)
    loss.backward()
    optimizer.step()
    return loss

##### Train the models #####
# Create some sample noise
sample_noise = random_vector(opt.sample_size, opt.z_size)

# Keep track of losses
d_losses = []
g_losses = []

for e in range(opt.epochs):
    for batch_i, (images, _) in enumerate(train_loader):
        batch_size = images.shape[0]
        # Move images to GPU if available
        if torch.cuda.is_available():
            images = images.cuda()
        # Train discriminator
        d_loss = train_discriminator(G, D, d_optimizer, images, batch_size, opt.z_size)
        # Train generator
        g_loss = train_generator(G, D, g_optimizer, batch_size, opt.z_size)
        
        # Keep track of losses
        d_losses.append(d_loss)
        g_losses.append(g_loss)
        
        # Save some sample pictures
        if (batch_i % opt.print_every == 0):
            print("Epoch: {}, Batch: {}, D-Loss: {}, G-Loss: {}".format(e, batch_i, d_loss, g_loss))
            # Make sample generation
            G.eval()
            fig, axes = plt.subplots(1, opt.sample_size, figsize=(20, 10))
            # Generate predictions
            predictions = G.forward(sample_noise)
            for i in range(opt.sample_size):
                axes[i].imshow(im_convert(predictions[i], rescale=True), cmap="gray")
            plt.savefig("generated_images/sample_E{}_B{}.png".format(e, batch_i))
            plt.close("all")
    if (e % opt.plot_every == 0):
        # Print losses
        plt.plot(d_losses, label="Discriminator", alpha=0.5)
        plt.plot(g_losses, label="Generator", alpha=0.5)
        plt.title("Trainings loss")
        plt.legend()
        plt.savefig("generated_images/loss_E{}.png".format(e))
        plt.close("all")

torch.save({
    "epochs": opt.epochs,
    "g_model_dict": G.state_dict(),
    "d_model_dict": D.state_dict(),
    "g_optim": g_optimizer.state_dict(),
    "d_optim": d_optimizer.state_dict()
}, opt.model_save_path + ".tar")

print("Saved models into directory")