from model_architecture import forward
from loss_and_accuracy import cross_entropy, accuracy 
from data_preprocessing import batch_generator
from backpropagation import backward
import copy
import numpy as np

def evaluate(X,Y, y, params):
  probs, _ = forward(X, params)
  return cross_entropy(probs, Y), accuracy(probs, y)

def train(params, optimizer, X_train, Y_train, y_train, X_valid, Y_valid, y_valid,
  epochs=15, bs=128, seed=0):
  rng = np.random.default_rng(seed)
  history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
  best_val_acc, best_params = 0.0, None

  for epoch in range(1, epochs + 1):
    for xb, Yb in batch_generator(X_train, Y_train, bs, rng=rng):
      probs, cache = forward(xb, params)
      grads = backward(Yb, params, cache)
      optimizer.step(params, grads)

    tr_loss , tr_acc = evaluate(X_train, Y_train, y_train, params)
    val_loss , val_acc = evaluate(X_valid, Y_valid, y_valid, params)
    for k, v in zip(history, (tr_loss, tr_acc, val_loss, val_acc)):
      history[k].append(float(v))

    if val_acc > best_val_acc:
      best_val_acc = val_acc
      best_params = copy.deepcopy(params)

    print(f'epoch {epoch:2d} | train_loss {tr_loss:.4f} acc {tr_acc:.4f}) | '
          f'val loss {val_loss:.4f} acc {val_acc:.4f}')
  return best_params, history