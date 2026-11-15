def sgd_step(params, grads, lr=0.1):
  for k in params:
    params[k] -= lr * grads[k] 