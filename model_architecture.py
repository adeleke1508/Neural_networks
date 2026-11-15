import numpy as np

layers_size = [784, 128, 64, 10]
def init_params(layers_size, seed=0):
  rng = np.random.default_rng(seed)
  params = {}
  for i in range(1, len(layers_size)):
    fan_in, fan_out = layers_size[i - 1], layers_size[i]
    params[f'W{i}'] = (rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)).astype(np.float32)
    params[f'b{i}'] = np.zeros((1, fan_out), dtype=np.float32)
  return params

def relu(z):
  return np.maximum(0, z)

def softmax(z):
  z = z - z.max(axis=1, keepdims=True)
  e = np.exp(z)
  return e /e.sum(axis=1, keepdims=True)

def forward(X, params):
  L = len(params) // 2
  cache = {'a0': X}
  a = X
  for i in range(1, L + 1):
    z = a @ params[f'W{i}'] + params[f'b{i}']
    cache[f'z{i}'] = z
    a = relu(z) if i < L else softmax(z)
    cache[f'a{i}'] = a
  return a, cache 