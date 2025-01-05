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

## Results

### Model Performance

THe model achieved a validation accuracy of 78.27% with a drastically reduced parameter count of 68,923 parameters, making the model lightweight and efficeint while maintaining strong performance across most modulation types. 

### Confusion Matrix
[[2702   10    3    1    1    0   48    0    0    0    0]
 [  41 2424  243    8    1    2   39    0    2    5    0]
 [  33  337 2310   24    2    5   29    0    0    3    0]
 [  25   63  150 1932  140  417   30    0    0    0    1]
 [  16   57   84 1042  341 1148   27    0    0    0    0]
 [  15   46   81  927  227 1328   32    0    0    1    2]
 [  56    7    0    1    3    2 2593    0    0    0    0]
 [   0    2    7    0    0    0    2 1850  607   13  167]
 [   0    4    3    0    0    0    0   39 2527   84    4]
 [   1    6    1    2    1    0    0    0  113 2532    7]
 [   0    2    0    0    0    0    2   81    0    0 2576]]

 ### Classification Report

              precision    recall  f1-score   support

        BPSK       0.97      0.96      0.97      2765
        QPSK       0.83      0.87      0.85      2765
        8PSK       0.82      0.85      0.84      2743
       16QAM       0.49      0.72      0.58      2758
       64QAM       0.44      0.18      0.26      2715
      128QAM       0.46      0.46      0.46      2659
        PAM4       0.93      0.98      0.95      2662
         FSK       0.93      0.70      0.80      2648
         MSK       0.79      0.94      0.86      2661
        GMSK       0.95      0.97      0.96      2663
        GFSK       0.94      0.96      0.95      2661

    accuracy                           0.78     29700
   macro avg       0.78      0.78      0.77     29700
weighted avg       0.78      0.78      0.77     29700

### Future Improvements
1. Enhance data augmentation: Apply more targeted automation for underperforming classes likes 64QAM ans 128QAM
