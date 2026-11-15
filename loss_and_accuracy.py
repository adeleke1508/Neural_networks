import numpy as np

def cross_entropy(probs, Y, eps=1e-12):
  return -np.sum(Y * np.log(probs + eps), axis=1).mean()

def accuracy(probs, y):
  return (probs.argmax(axis=1) == y).mean()