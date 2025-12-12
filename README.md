# Generative Audio Inversion (GAI)

**Official PyTorch Implementation of "Generative Audio Inversion for Speaker Recognition Systems"**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

## Resources
1. [Report](Report/AudioMI_G3_Report.pdf)
2. [Inverted Samples and Model Saves](https://drive.google.com/drive/folders/1uxwEYjZH9Es582MMqcqQ0vGjquZzJXBX?usp=drive_link)

## Setup

### 1. Setup Enviroment and add [neptune api-key](https://docs.neptune.ai/api_token/) to a .env file (example is given in [.env.example](.env.example))
```bash
# Clone the repository
https://github.com/DevinDeSilva/AudioGenMI_G3_deliverables.git
cd AudioGenMI_G3_deliverables

# Install dependencies
pip install -r requirements.txt
```
#### Setup .env
```bash
NEPTUNE_API_TOKEN="example-api-key"
NEPTUNE_PROJECT="example-project-name"
```

### 2. Download Data to `data` folder (Please note this is close to 50 GB)
```bash
# Move to the data folder
cd data

# run download_data.sh
source download_data.sh
```



### 3. Run Data-preprocessing. This contains the part where the datasets are 

### 4. Download and extract the pretrained models available in the gdrive link available in Resources section to `models/` folder (Contains all the models I trained + pretrainining weights of hifigan originally available in [here](https://github.com/jik876/hifi-gan))

### 5. train main speaker recognition model and evaluation models (optional since step 3 downlods my pretrains)

### 6. Train GAI model (optional since step 3 downlods my pretrains)

### 7. Run Model Inversion baseline methods (I have added the inverted samples I generated in the gdrive link available in Resources section).

### 8. Run Model Inversion our method (I have added the inverted samples I generated in the gdrive link available in Resources section).

### 9. Run evaluation.


## 📝 Abstract

This repository hosts the code for the paper **Generative Audio Inversion (GAI)**. We introduce a novel framework that adapts Generative Model Inversion (GMI) to the acoustic domain to reconstruct high-fidelity speaker representations from a target Speaker Recognition System (specifically SincNet).

Unlike existing audio MI techniques that optimize raw waveforms and result in noisy, intelligible static, our approach leverages a **Generative Adversarial Network (GAN)** trained on public auxiliary data (LibriSpeech) to serve as a distributional prior. By optimizing in the latent space of the GAN and utilizing a neural vocoder (HiFi-GAN), we recover mel-spectrograms that preserve harmonic structures and formant patterns characteristic of natural human speech.

## 🖼️ Methodology

Our framework consists of two main stages:
1.  **Prior Learning:** Training a GAN ($G_{mel}$, $D_{mel}$, $D_{aud}$) on public data to model realistic mel-spectrogram distributions.
2.  **Inversion Attack:** Freezing the generator and optimizing the latent vector $z$ to minimize the identity loss of the target speaker in the victim model (SincNet).

### Attack Pipeline
![System Diagram](Docs/img/audioMIV2.png)
*Figure 1: The proposed Generative Audio Inversion attack pipeline.*

### GAN Training
![Training Diagram](Docs/img/AudioMI-ModelTrainingV2.png)
*Figure 2: The GAN training procedure using auxiliary public data.*

## 🛠️ Installation

### Prerequisites
* Linux or macOS
* Python 3.8+
* NVIDIA GPU + CUDA
