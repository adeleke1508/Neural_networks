import numpy as np

class SGD:
  def __init__(self, params, lr=0.1):
    self.lr = lr
  
  def step(self, params, grads):
    for k in params:
      params[k] -= self.lr * grads[k]

class Momentum:
  def __init__(self, params, lr=0.01, beta=0.9):
    self.lr , self.beta = lr, beta
    self.v = {k: np.zeros_loke(p) for k, p in params.items()}

  def step(self, params, grads):
    for k in params:
      self.v[k] = self.beta * self.v[k] + grads[k]
      params[k] -= self.lr * self.v[k] 

class Adam:
  def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    self.lr, self.b1, self.b2 , self.eps = lr, beta1, beta2, eps
    self.m = {k:np.zeros_like(p) for k, p in params.items()}
    self.v = {k: np.zeros_like(p) for k, p in params.items()}
    self.t = 0

  def step(self, params, grads):
    self.t += 1
    for k in params:
      self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
      self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k]**2
      m_hat = self.m[k] / (1 - self.b1 ** self.t)
      v_hat = self.v[k] / (1 - self.b2 ** self.t)
      params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)