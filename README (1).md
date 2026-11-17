# Neural Network from Scratch: MNIST Digit Classifier

A fully-connected neural network (MLP) built **from scratch with NumPy only**: no autograd, no `torch.nn`. Initialization, forward pass, loss, backpropagation, optimizers (SGD and Adam), the training loop, evaluation, and model saving are all hand-written and verified. `torchvision` is used only to download the MNIST dataset.

## Highlights

- **97.65% test accuracy** on MNIST with a 109,386-parameter network (784 → 128 → 64 → 10).
- **Backward pass verified by gradient checking**: analytic gradients match finite differences with relative error below 4e-9.
- Passes the classic **"overfit a tiny subset"** sanity test (100 images memorized to 100% accuracy).
- Two optimizers implemented from scratch (**SGD** and **Adam**) and compared under identical conditions.
- Full evaluation: confusion matrix, per-class precision/recall, and an analysis of the most confident mistakes.
- Model saved together with its preprocessing statistics; reloading reproduces the test score exactly.

## Results

| | SGD (lr 0.1) | Adam (lr 1e-3) |
|---|---|---|
| Best validation accuracy | 97.70% | 97.72% |
| **Test accuracy** | **97.65%** | 97.54% |
| Test loss | 0.0850 | 0.1005 |

Setup for both: batch size 128, 15 epochs, same seed and initial weights, best-validation-accuracy checkpoint.

**Training curves (SGD)**

![Training curves](assets/training_curves_sgd.png)

**Confusion matrix on the test set**

![Confusion matrix](assets/confusion_matrix.png)

**The 12 most confident mistakes** (many are genuinely ambiguous or badly written digits)

![Misclassified examples](assets/misclassified_examples.png)

### Key observations

- **Adam learns faster early on** (95.1% vs 94.0% validation accuracy after epoch 1) but finishes in the same place as SGD. The 11-image test difference (97.65% vs 97.54%) is within noise: the standard error at this accuracy on 10,000 images is about 0.15 points.
- **Both runs overfit mildly.** Training accuracy reaches about 99.98% while validation stalls near 97.7%, and validation loss stops improving after roughly epoch 6 to 7. Adam's validation loss rises more (0.093 → 0.127), so it overfits sooner.
- **The model is overconfident.** Its most confident mistakes all sit at probability 1.00, a typical effect of driving the training loss near zero without regularization.
- **Weakest classes:** digit 5 has the lowest recall (0.961) and digit 8 the lowest precision (0.958). The most common confusions are 4 → 9, 7 → 2 and 5 → 3.

## Architecture and design choices

| Component | Choice | Why |
|---|---|---|
| Input | 784 pixels, standardized with train-set mean/std | zero-centered inputs train faster with ReLU; train-only statistics avoid data leakage |
| Hidden layers | 128 and 64 units, **ReLU** | cheap, and its gradient does not saturate for positive inputs |
| Output | 10 units, **softmax** (computed with the row-max trick) | class probabilities; subtracting the max prevents `exp` overflow |
| Initialization | **He**: `W ~ N(0, 2 / fan_in)`, zero biases | keeps activation variance stable through ReLU layers |
| Loss | **Cross-entropy**, averaged over the batch | pairs with softmax to give the simple gradient `(probs - Y) / N` |
| Optimizers | SGD, Adam (in-place parameter updates) | compare a simple baseline against an adaptive method |
| Model selection | deep copy of the weights with best validation accuracy | the test set is touched exactly once, at the end |

### How the backward pass works

For each layer, from the output back to the input:

```
dz (output)  = (probs - Y) / batch_size          # softmax + cross-entropy
dW           = a_prev.T @ dz
db           = dz.sum(axis=0)
da_prev      = dz @ W.T
dz (previous) = da_prev * (z_prev > 0)            # ReLU derivative
```

`grad_check` validates this against centered finite differences, `(L(w + ε) - L(w - ε)) / 2ε`, in float64.

## Project structure

```
Neural_networks/
├── neural_networks_from_scratch.ipynb   # driver notebook: runs and explains every step
├── data_preprocessing.py                # load MNIST, flatten/scale, split, standardize, one-hot, batch_generator
├── model_architecture.py                # layers_size, init_params (He init), relu, softmax, forward
├── loss_and_accuracy.py                 # cross_entropy, accuracy
├── backpropagation.py                   # backward, grad_check
├── optimizers.py                        # SGD, Adam
├── training_loop.py                     # evaluate, train
├── load_model.py                        # load_model (weights + mean/std)
└── assets/                              # figures used in this README
```

The notebook loads the modules with `%run`, so keep them in the same folder.

> **Label convention:** lowercase `y_*` are integer labels (accuracy, confusion matrix); uppercase `Y_*` are one-hot labels (loss, gradient).

## Getting started

### Google Colab (as developed)

1. Put the `.py` files and the notebook in `MyDrive/Neural_networks/` on Google Drive.
2. Open the notebook in Colab and run all cells. The first cell mounts Drive and the next one adds the folder to `sys.path`.

### Locally

```bash
pip install numpy matplotlib torch torchvision jupyter
```

In the notebook, remove the `google.colab` mount cell and change the folder path in the `sys.path.append(...)` cell and in each `%run ...` line to your local project folder. The first run downloads MNIST into a `data/` folder.

## Reproducibility

The train/validation split, weight initialization and batch shuffling all use fixed seeds, so rerunning the notebook reproduces the numbers above (up to library-version differences).

## Limitations and next steps

- [ ] **Regularization:** L2 weight decay and dropout, to close the train/validation gap and reduce overconfidence.
- [ ] **Learning-rate schedule** to smooth the noisy late epochs.
- [ ] **Inference on real handwriting:** crop, center and resize a photo into MNIST format with the saved `mean`/`std`.
- [ ] **PyTorch cross-check:** same architecture and seed, to validate the from-scratch implementation.
- [ ] **Small CNN from scratch** (convolution and pooling) for 99%+ accuracy.

## Author

Khaleed ([github.com/adeleke1508](https://github.com/adeleke1508))
