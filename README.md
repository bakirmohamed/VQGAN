# VQGAN

**Vector Quantized Generative Adversarial Networks (VQGAN)** is a generative model for image synthesis, introduced in the paper [*Taming Transformers for High-Resolution Image Synthesis*](https://arxiv.org/abs/2012.09841). It combines convolutional neural networks and transformers to learn discrete latent representations of images.

The training process is divided into two distinct stages:

1. A **vector-quantized autoencoder** that learns to compress and reconstruct images using a learned codebook.
2. A **transformer model** that learns the structure of the latent space to enable autoregressive generation of novel image content.

## Project Setup and Execution

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Use 'venv\Scripts\activate' on Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your dataset path

In the `training_vqgan.py` file, locate the line:

```python
args.dataset_path = r"/home/mbakir/datasets/flowers"
```

Replace the path with the location of your dataset on your system.

### 4. Start training

```bash
python training_vqgan.py
```

## First Training Part: Vector-Quantized Autoencoder

In this stage, the model learns to represent images in a compressed latent space using a convolutional autoencoder architecture:

* The **encoder** transforms input images into low-dimensional latent representations.
* These representations are then **quantized** using a learned **codebook**: each latent vector is replaced with its closest matching vector from the codebook.
* The **decoder** reconstructs the original image from the quantized latent representation.
* Training minimizes a combination of reconstruction loss and a commitment loss to encourage efficient codebook usage.

This phase enables the model to represent images with discrete, learnable latent codes.



## Second Training Part: Transformer on Latent Codes

Once the autoencoder and codebook are trained, the second stage trains a transformer to model the distribution of the quantized latent codes:

* The transformer is trained **autoregressively** on sequences of latent codebook indices.
* It learns patterns and dependencies between codes, effectively modeling the structure of the image space.
* At inference time, the transformer can generate new sequences of latent codes, which are then decoded by the pre-trained decoder to produce novel images.

This stage enables the model to generate coherent and realistic images by sampling directly in the learned discrete latent space.

