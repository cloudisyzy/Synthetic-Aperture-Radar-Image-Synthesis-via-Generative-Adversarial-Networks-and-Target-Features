
# SAR Image Generation with GANs

This repository hosts the code and resources for the "SAR Image Generation based on Generative Adversarial Networks and Target Characteristics" project. This project explores the application of various GAN architectures to generate Synthetic Aperture Radar (SAR) images to enhance target recognition algorithms under limited data conditions. This is my bachelor degree project.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [GAN Architectures](#gan-architectures)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [Data and Model Availability](#data-and-model-availability)
- [Acknowledgments](#acknowledgments)

## Introduction

This project evaluates the effectiveness of several GAN architectures for the synthesis of SAR images, particularly under conditions of data scarcity. This research is crucial for applications where SAR data is limited due to the high costs associated with its acquisition. The GANs tested include DCGAN, WGAN-GP, WGAN-DIV, SNGAN, Enhanced SNGAN, and VAE-GAN, each offering unique advantages in terms of image quality and training stability.

## Installation

To set up the project environment, clone this repository and install the required dependencies:

```bash
git clone https://github.com/cloudisyzy/SAR-Image-Generation-based-on-Generative-Adversarial-Networks-and-Target-Characteristics
cd SAR-Image-Generation-based-on-Generative-Adversarial-Networks-and-Target-Characteristics
pip install -r requirements.txt
```

## Usage

To generate SAR images using a pre-trained model, run:

```python
python SAR_Fast_Generation.py
```

For training new models, navigate to the specific GAN directory and run the training script, e.g., DCGAN:

```bash
cd DCGAN
python DCGAN_Train.py
```

## GAN Architectures

- **DCGAN**: Basic GAN architecture with convolutional layers.
- **WGAN-GP and WGAN-DIV**: Introduce Wasserstein loss and gradient penalties for more stable training.
- **SNGAN and Enhanced SNGAN**: Utilize spectral normalization to stabilize GAN training.
- **VAE-GAN**: Combines VAE and GAN for generating high-quality images.

## Repository Structure

- `DCGAN`, `SNGAN`, etc.: Directories containing the specific GAN training code.
- `__models`: Contains PyTorch definitions of the GAN architectures.
- `__utils`: Helper functions used across the project.
- `_Angle_Extraction(beta)`: Estimates the angle of the object inside a SAR image.
- `_Histogram_Gray_Level`: Compares pictures via grey scale distribution
- `_Classification_SVM`: Uses SVM to testify whether the generated images are validate.
- `Slides&Reports`: Contains project presentations and the final report.
- `SAR_Fast_Generation.ipynb`: Jupyter notebook for quick generation using trained models.
- `requirements.txt`: Project dependencies.

## Dependencies

Ensure you have Python 3.6+ installed, and install all required libraries with:

```bash
pip install -r requirements.txt
```

## Data and Model Availability

The SAR data used in this project is based on the MSTAR dataset, which is not included in the repository. You can download it from [MSTAR Dataset](https://www.sdms.afrl.af.mil/index.php?collection=mstar).

Note: This repository does not include the pre-trained model parameters or the MSTAR dataset used for training due to licensing and data size constraints. As a result, the code cannot be executed directly to reproduce the results. To use the models, you will need to first train them using the GAN-specific training scripts provided in each GAN's directory.

Note: There is also a DDIM (Diffusion Implicit Model) implementation with much better results, which I will upload in the future.

## Acknowledgments

- University of Glasgow and University of Electronic Science and Technology of China for providing resources and support.
- My supervisor for his guidance and feedback on this research project.

Thank you for your interest in our SAR image generation research!

