import numpy as np 
from torchvision import datasets

train_ds = datasets.MNIST('data', train=True, download=True)
test_ds = datasets.MNIST('data', train=False, download=True)

X_full = train_ds.data.numpy().astype(np.float32)
y_full = train_ds.targets.numpy().astype(np.int64)
X_test = test_ds.data.numpy().astype(np.float32)
y_test = test_ds.targets.numpy().astype(np.int64)

X_full = X_full.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

rng = np.random.default_rng(42)
perm = rng.permutation(len(X_full))
X_full, y_full = X_full[perm], y_full[perm]

X_train, y_train = X_full[:55000], y_full[:55000]
X_valid, y_valid = X_full[55000:], y_full[55000:]

mean, std = X_train.mean(), X_train.std()
X_train = (X_train - mean) / std
X_valid = (X_valid - mean) / std
X_test = (X_test - mean) / std

def one_hot(y, num_classes=10):
  return np.eye(num_classes, dtype=np.float32)[y]

Y_train, Y_valid, Y_test = one_hot(y_train), one_hot(y_valid), one_hot(y_test)

def batch_generator(X, Y, bs=128, shuffle=True, rng=rng):
  idx = rng.permutation(len(X)) if shuffle else np.arange(len(X))
  for start in range(0, len(X), bs):
    b = idx[start:start + bs]
    yield X[b], Y[b]