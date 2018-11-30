# PyTorch-Linear-GAN
A PyTorch notebook and implementation of a normal linear GAN

This repository contains a ready to use vanilla GAN accessible through the terminal as well as a jupyter notebook with the exact same network structure for easier overview. The notebook is also optimized for training in Google Collaboratory.
GANs were [first introduced](https://arxiv.org/abs/1406.2661) in 2014 from Ian Goodfellow and others. 
GANs consists of two networks: a generator ***G*** and a discriminator ***D***. The generator is designed to generate realistic looking data "fake data" and the discriminator is designed to distinguish between generated "fake data" and real data from a dataset. Both networks play a game against each other where the generator tries to fool the discriminator by generating data as real as possible and the discriminator tries to classify fake and real data correctly.

* The generator is a linear feedforward network which takes a random vector as input and outputs another vector which can be reshaped to an image
* The discriminator is a linear classifier that takes an image vector as input and outputs a probability of that image being real.

This game leads to a generator producing data that is indistinguishable from real data to the discriminator

```
usage: main.py [-h] [--epochs EPOCHS] [--lr LR] [--batch_size BATCH] [--beta1 BETA1] [--beta2 BETA2] [--print_every EVERY] 
[--sample_size SIZE] [--plot_every EVERY] [--model_save_path PATH] [--custom_image_path PATH] [--image_size SIZE] [--z_size Z_SIZE] 

optional arguments:
  -h, --help                 show this help message and exit
  --epochs EPOCHS            number of epochs
  --lr LR                    learning rate for both optimizers
  --batch_size BATCH         number of images in one batch
  --beta1 BETA1              beta 1 value | adam optimizer
  --beta2 BETA2              beta 2 value | adam optimizer
  --print_every EVERY        number of BATCHES after loss is printed and sample images are saved
  --sample_size SIZE         number of samples
  --plot_every EVERY         number of EPOCHS after plot of loss will be saved
  --model_save_path PATH     path where the model should be saved including the filename
  --custom_image_path PATH   path to own dataset of black and white images | if not specified MNIST will be downloaded and used.
  --image_size SIZE          height and width of image | image will be rescaled to that size
  --z_size Z_SIZE            dimensionality of the latent vector z
  ```

