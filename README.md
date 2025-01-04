# AI for RF Signal Modulation Classification
## 2024 AI Challenge by Ball Aerospace

This repository is a submission for the AI for RF Signal Modulation Classification Challenge. The challenge aims to develop an automated signal classifier for use as a payload skill on a next-generation satellite mission. The primary goal is to autonomously classify radio frequency (RF) signals into predefined classes based on sampled waveforms.

RF signal classification is critical for various applications, including:
	•	RF interference detection
	•	RF intelligence gathering
	•	Spectrum management
	•	Jammer detection
	•	Spectral compliance verification

## Introduction
Radio frequency signal classification is a critical task in wireless communication systems, allowing for efficient and accurate interpretation of signals. This project aims to classify RF signals into various modulation types using a neural network architecture optimized for accuracy.

---

## Features
Input Format: Time-series data of complex-valued IQ samples at a sampling rate of 100 MHz.
Supported Modulation Types:
	1.	BPSK
	2.	QPSK
	3.	8PSK
	4.	MSK
	5.	FSK
	6.	PAM4
	7.	GMSK
	8.	GFSK
	9.	16QAM
	10.	64QAM
	11.	128QAM
Training Dataset:
    •	297,000 examples across 11 modulation types
    •	Labels provided as one-hot encoded vectors

---

## Installation

Grab the training data: https://www.icloud.com/iclouddrive/04dGShJz9KTKeWditkLtJyeDw#AI_Challenge_Training_Data

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/RF-Signal-Classifier.git
    cd RF-Signal-Classifier
    ```

2. Set up a Python virtual environment:
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### Training the Model
Run the following command to start training the model:
```bash
python rf_signal_classifier/train.py
```

### Inference
Run the model on the test dataset
python run_inference.py --test_data_path <path-to-test-data>

### Evaluation
Score = (accuracy / 100) + (10 / log10(Nparams)) + β - ε