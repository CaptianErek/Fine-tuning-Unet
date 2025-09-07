# Fine-Tuned UNet with DDPM Scheduler on FashionMNIST

## Overview  

This repository contains a **Diffusion Model implementation** using a **UNet architecture fine-tuned with the DDPM (Denoising Diffusion Probabilistic Model) scheduler** on the **FashionMNIST dataset**. Leveraging the Hugging Face **`diffusers`** library along with PyTorch, the model learns to generate high-quality synthetic fashion images, demonstrating the power of diffusion-based generative modeling.  

The fine-tuned model achieved **high visual fidelity**, with generated samples closely resembling real FashionMNIST data.  

## Features  

- 🌀 **Diffusion Model**: Implemented with **DDPM Scheduler** for iterative denoising  
- 🏗️ **UNet Backbone**: Optimized for image generation tasks  
- 👗 **Dataset**: FashionMNIST, showcasing generative capabilities on grayscale fashion items  
- 📈 **Accurate Results**: Produces realistic and diverse outputs after training  
- 🧩 **Modular Design**: Easily extendable to other datasets or schedulers  
- 🔧 **Built on Diffusers**: Clean integration with Hugging Face `diffusers` library  

## Requirements  

Install the dependencies using:  

```bash
pip install torch torchvision diffusers matplotlib numpy
```

## Libraries Used
torch, torchvision – Core deep learning framework and dataset utilities

diffusers – Hugging Face library for diffusion models and schedulers

numpy, matplotlib – Data handling and visualization

## Repository Structure
### 1. Data Loading
Download and prepare FashionMNIST dataset

Normalize and preprocess images for diffusion training
### 2. Model Architecture
UNet backbone as the core denoiser

Configurable channels, blocks, and attention layers
### 3. Training with DDPM Scheduler
Progressive noise addition and removal

Trained with Hugging Face DDPMScheduler

Loss optimization for stable generation
### 4. Evaluation & Sampling
Generate samples from random noise

Visualize generated images alongside FashionMNIST ground truth
