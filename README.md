# DNN Shrinking and Quantization

A hands-on deep learning project exploring the complete workflow from **CNN definition and training** to **model shrinking, quantization-aware training, integer inference, and ONNX export**.

The experiments use **Fashion-MNIST** and PyTorch, with **Brevitas** for quantization and **DeepQuant** for fake-to-true quantization.

## Overview

This repository contains two main notebook-based labs:

1. **DNN Definition and Training**
   - Define a convolutional neural network from scratch
   - Train and validate it on Fashion-MNIST
   - Measure parameters, MAC operations, and model size
   - Save and reload trained PyTorch weights

2. **DNN Shrinking and Quantization**
   - Reduce the computational cost of the CNN
   - Retrain the smaller architecture
   - Convert a floating-point model to a fake-quantized model
   - Calibrate activation ranges
   - Perform Quantization-Aware Training (QAT)
   - Convert the network to true integer inference using DeepQuant
   - Export float and quantized models to ONNX

## Key Results

| Model / Stage | Parameters | MACs | Validation Accuracy |
|---|---:|---:|---:|
| Original CNN | ~155.6K | ~18.35M | ~90.48% |
| Reduced CNN | ~54.8K | ~4.66M | ~89.10% |
| Initial 8-bit fake-quantized model | — | — | ~75.35% |
| After calibration | — | — | ~90.46% |
| After 1 epoch of QAT | — | — | ~91.27% |
| True-quantized integer model | — | — | ~91.30% |

The reduced CNN satisfies the lab constraint of **fewer than 5 million MAC operations**, reducing compute by roughly **75%** compared with the original architecture.

> Results are taken from the executed notebook outputs and can vary slightly between runs.

## Model Architecture

The original CNN processes `28 x 28` grayscale Fashion-MNIST images using three convolutional blocks:

```text
Input: 1 x 28 x 28
        |
Conv 3x3, 32 channels
BatchNorm
ReLU
        |
Conv 3x3, 64 channels
BatchNorm
ReLU
        |
MaxPool 2x2
        |
Conv 3x3, stride 2, 128 channels
BatchNorm
ReLU
        |
Dropout (0.5)
        |
Flatten
        |
Fully Connected
        |
10 output classes
```

For the shrinking experiment, the channel widths are halved:

```text
1 -> 16 -> 32 -> 64
```

This reduces the network from approximately **18.35M MACs to 4.66M MACs** and from approximately **155.6K to 54.8K parameters**.

## Quantization Pipeline

The second notebook demonstrates a practical post-training and quantization-aware workflow:

```text
Float32 PyTorch Model
        |
Activation Equalization
        |
Brevitas Graph Preprocessing
        |
Float-to-Fake Quantization
        |
8-bit Weights + Activations
        |
Calibration
        |
Quantization-Aware Training
        |
Fake-to-True Quantization
        |
Integer Model with DeepQuant
        |
ONNX Export
```

### 1. Float-to-Fake Quantization

Standard PyTorch layers are replaced with Brevitas quantized equivalents such as:

- `QuantConv2d`
- `QuantLinear`
- `QuantReLU`
- Quantized BatchNorm scale/bias operations

The notebook uses **8-bit weights and activations** as the main configuration.

### 2. Calibration

Calibration estimates appropriate activation ranges and quantization parameters using the validation data.

In the recorded run, calibration recovered most of the accuracy lost immediately after fake quantization.

### 3. Quantization-Aware Training

The fake-quantized model is fine-tuned while simulating quantization effects during training.

The notebook performs QAT with a small learning rate:

```python
lr = 1e-5
```

### 4. Fake-to-True Quantization

DeepQuant converts the Brevitas fake-quantized network into a model that performs integer-oriented inference.

A sanity check compares predictions from the floating-point and true-quantized networks.

### 5. ONNX Export

The project exports both floating-point and quantized network representations to ONNX for deployment and visualization.

The resulting models can be inspected with [Netron](https://netron.app/).

## Dataset

The experiments use **Fashion-MNIST**, containing 28x28 grayscale images across 10 clothing categories.

The notebooks automatically download the dataset using `torchvision.datasets.FashionMNIST`.

Images are converted to tensors and normalized using statistics computed from the training data.

## Repository Structure

```text
DNN-Shrinking-Quantization/
|
├── DNN Definition and Training.ipynb
│   └── CNN definition, training, evaluation, complexity analysis,
│       and model save/load workflow
|
├── DNN_shrinking_and_quantization.ipynb
│   └── Model shrinking, Brevitas quantization, calibration,
│       QAT, DeepQuant conversion, and ONNX export
|
└── LAB01/
    ├── matrix_vector.c
    └── vectorsum.c
```

## Requirements

The first notebook uses standard Python deep-learning packages including:

```text
numpy
tqdm
pillow
torch
torchvision
matplotlib
scikit-learn
torchinfo
ptflops
thop
```

The shrinking and quantization notebook additionally uses:

```text
brevitas
onnxscript
sympy==1.11
DeepQuant
```

For compatibility with the DeepQuant workflow, the notebook explicitly installs:

```text
torch==2.4.0
torchvision==0.19.0
torchaudio==2.4.0
sympy==1.11
```

## Running the Project

### Option 1 — Google Colab

The notebooks are designed to work well in Google Colab.

Open either `.ipynb` file in Colab and execute the cells sequentially.

For the quantization notebook, the setup section changes the PyTorch environment and installs DeepQuant. After installation, **restart the Colab runtime when instructed**, then continue from the main lab section rather than rerunning the setup cells.

### Option 2 — Local Jupyter Environment

Clone the repository:

```bash
git clone <your-repository-url>
cd DNN-Shrinking-Quantization
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install Jupyter and the dependencies needed for the notebook you want to run.

For the basic DNN notebook:

```bash
pip install jupyter numpy tqdm pillow torch torchvision matplotlib \
    scikit-learn torchinfo ptflops thop
```

Start Jupyter:

```bash
jupyter notebook
```

For the quantization workflow, it is recommended to follow the **version-pinned setup cells inside the notebook**, because DeepQuant requires a compatible PyTorch/SymPy environment.

## DeepQuant Setup

The quantization notebook uses the `unitn25-exercise` branch of DeepQuant:

```bash
git clone --branch unitn25-exercise --single-branch \
    https://github.com/FrancescoConti/DeepQuant.git

cd DeepQuant
git submodule update --init --recursive
pip install -e .
```

## Training Configuration

The notebooks use configurations such as:

### Original CNN

```text
Epochs:      5
Batch size:  128
Loss:        CrossEntropyLoss
Dataset:     Fashion-MNIST
```

The recorded training run reached approximately **91.3% validation accuracy** after five epochs.

### Reduced CNN

```text
Epochs:      2
Batch size:  128
```

The recorded run reached approximately **89.1% validation accuracy** while keeping computation below 5M MACs.

### Quantization-Aware Training

```text
QAT epochs:  1
Learning rate: 1e-5
```

## Tools and Technologies

- **Python**
- **PyTorch**
- **Torchvision**
- **Fashion-MNIST**
- **Brevitas**
- **DeepQuant**
- **ONNX**
- **Netron**
- **TorchInfo**
- **ptflops**
- **Scikit-learn**
- **Jupyter / Google Colab**

## What This Project Demonstrates

This project focuses on practical techniques used when moving neural networks from experimentation toward resource-constrained deployment:

- CNN architecture design
- Training and validation loops in PyTorch
- Parameter and MAC analysis
- Model complexity reduction
- Accuracy/efficiency trade-offs
- Post-training quantization
- Activation calibration
- Quantization-Aware Training
- Integer inference
- ONNX model export
- Deployment-oriented neural network optimization

## Notes

- Fashion-MNIST is downloaded automatically when the notebooks are executed.
- Generated weight files and ONNX models may not be present until the corresponding notebook cells are run.
- DeepQuant compatibility depends on the package versions specified by the notebook.
- CPU execution is sufficient for the lab-sized models, although training may be faster with compatible acceleration.
- Reported metrics correspond to the outputs currently stored in the notebooks and are not guaranteed to be identical on every run.

## Acknowledgements

This repository uses open-source tools and libraries including:

- [PyTorch](https://pytorch.org/)
- [Brevitas](https://github.com/Xilinx/brevitas)
- [DeepQuant](https://github.com/FrancescoConti/DeepQuant)
- [ONNX](https://onnx.ai/)
- [Netron](https://netron.app/)
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

## License

No license file is currently included in this repository. Add a license before distributing or reusing the project beyond its intended academic context.
