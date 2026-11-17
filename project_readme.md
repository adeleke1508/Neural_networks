# PyTorch MNIST Classification Project

This repository contains a PyTorch implementation of a Multilayer Perceptron (MLP) trained to classify handwritten digits from the MNIST dataset.

## 📌 Project Overview
- **Dataset:** MNIST ($28 \times 28$ grayscale images of handwritten digits $0-9$)
- **Architecture:** 3-layer Fully Connected Neural Network (MLP)
- **Input Dimension:** $784$ ($28 \times 28$ pixels flattened)
- **Hidden Layers:** Layer 1 ($128$ units, ReLU), Layer 2 ($64$ units, ReLU)
- **Output Dimension:** $10$ classes
- **Optimization:** Adam Optimizer ($\text{lr} = 0.001$) with Cross-Entropy Loss

---

## 📊 Dataset Split & Preprocessing
- **Preprocessing:** Standardized pixel intensity normalization ($\mu = 0.1307$, $\sigma = 0.3081$).
- **Data Splits:**
  - Training Set: $55,000$ samples
  - Validation Set: $5,000$ samples
  - Test Set: $10,000$ samples
- **Batch Size:** $128$

---

## 📈 Results
After training for 15 epochs, the model achieved the following performance metrics:
- **Final Validation Accuracy:** $97.58\%$
- **Final Test Accuracy:** **$97.71\%$**

---

## 🚀 How to Run

### Prerequisites
Install PyTorch and torchvision:
```bash
pip install torch torchvision
```

### Execution
Open and execute the notebook in Google Colab or Jupyter Notebook:
```bash
jupyter notebook mnist_training.ipynb
```