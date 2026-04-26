# CIFAR-10 Image Classification – CAP 4630 Project 4

**Team Members:** Michael, Zharah

## Overview

This project implements and compares two neural network architectures for image classification on the CIFAR-10 dataset:

1. **Baseline Model** – A fully connected (dense) feedforward network
2. **CNN Model** – A Convolutional Neural Network with Batch Normalization and Dropout

## Requirements

Install dependencies using pip:

```bash
pip install tensorflow numpy matplotlib
```

Python 3.8 or higher is recommended. The project was developed and tested with TensorFlow 2.x.

## How to Run

1. Clone or extract the project directory.
2. Navigate to the project folder:

```bash
cd ai-project-4-main
```

3. Run the main script:

```bash
python cifar10_project.py
```

The script will automatically download the CIFAR-10 dataset via Keras on first run (requires internet connection).

## Output

Running the script produces the following output files:

**`figures/`**
- `sample_images.png` – A grid of 10 sample training images with class labels
- `baseline_accuracy.png` – Training and validation accuracy curves for the baseline model
- `baseline_loss.png` – Training and validation loss curves for the baseline model
- `cnn_accuracy.png` – Training and validation accuracy curves for the CNN model
- `cnn_loss.png` – Training and validation loss curves for the CNN model
- `baseline_predictions.png` – Example predictions from the baseline model on 10 test images
- `cnn_predictions.png` – Example predictions from the CNN model on 10 test images

**`results/`**
- `model_results.txt` – Final test loss and accuracy for both models

## Project Structure

```
ai-project-4-main/
├── cifar10_project.py   # Main script (data loading, model definitions, training, evaluation)
├── README.md            # This file
├── figures/             # Generated plots (created on run)
└── results/             # Generated results file (created on run)
```

## Notes

- A fixed random seed (42) is used for reproducibility.
- The CIFAR-10 dataset is split into 45,000 training, 5,000 validation, and 10,000 test images.
- Training runs for 10 epochs with a batch size of 32.
- A `ReduceLROnPlateau` callback reduces the learning rate automatically if validation loss stalls.
