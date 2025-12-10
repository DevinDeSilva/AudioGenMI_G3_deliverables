# Generative Audio Inversion (GAI)

**Official PyTorch Implementation of "Generative Audio Inversion for Speaker Recognition Systems"**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

## 📝 Abstract

This repository hosts the code for the paper **Generative Audio Inversion (GAI)**. We introduce a novel framework that adapts Generative Model Inversion (GMI) to the acoustic domain to reconstruct high-fidelity speaker representations from a target Speaker Recognition System (specifically SincNet).

Unlike existing audio MI techniques that optimize raw waveforms and result in noisy, intelligible static, our approach leverages a **Generative Adversarial Network (GAN)** trained on public auxiliary data (LibriSpeech) to serve as a distributional prior. By optimizing in the latent space of the GAN and utilizing a neural vocoder (HiFi-GAN), we recover mel-spectrograms that preserve harmonic structures and formant patterns characteristic of natural human speech.

## 🖼️ Methodology

Our framework consists of two main stages:
1.  **Prior Learning:** Training a GAN ($G_{mel}$, $D_{mel}$, $D_{aud}$) on public data to model realistic mel-spectrogram distributions.
2.  **Inversion Attack:** Freezing the generator and optimizing the latent vector $z$ to minimize the identity loss of the target speaker in the victim model (SincNet).

### Attack Pipeline
![System Diagram](images/audioMIV2.png)
*Figure 1: The proposed Generative Audio Inversion attack pipeline.*

### GAN Training
![Training Diagram](images/AudioMI-ModelTrainingV2.png)
*Figure 2: The GAN training procedure using auxiliary public data.*

## 🛠️ Installation

### Prerequisites
* Linux or macOS
* Python 3.8+
* NVIDIA GPU + CUDA

### Setup
```bash
# Clone the repository
git clone [https://github.com/yourusername/generative-audio-inversion.git](https://github.com/yourusername/generative-audio-inversion.git)
cd generative-audio-inversion

# Install dependencies
pip install -r requirements.txt