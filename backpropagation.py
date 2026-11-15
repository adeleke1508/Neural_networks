import numpy as np 
from model_architecture import forward 
from loss_and_accuracy import cross_entropy

def backward(Y, params, cache):
  L = len(params) // 2
  bs = Y.shape[0]
  grads = {}
  dz = (cache[f'a{L}'] - Y) / bs

  for i in range(L, 0, -1):
    a_prev = cache[f'a{i-1}']
    grads[f'W{i}'] = a_prev.T @ dz
    grads[f'b{i}'] = dz.sum(axis = 0, keepdims=True)

    if i > 1:
      da_prev = dz @ params[f'W{i}'].T
      dz = da_prev * (cache[f'z{i-1}'] > 0 )
  return grads

def grad_check(params, X, Y, eps=1e-5, n_checks=5):
    p64 = {k: v.astype(np.float64) for k, v in params.items()}
    X = X.astype(np.float64)

    _, cache = forward(X, p64)
    grads = backward(Y, p64, cache)

    rng_check = np.random.default_rng(0)
    for k in p64:
        assert grads[k].shape == p64[k].shape, f"shape mismatch for {k}"
        for _ in range(n_checks):
            idx = tuple(rng_check.integers(0, s) for s in p64[k].shape)
            orig = p64[k][idx]

            p64[k][idx] = orig + eps
            loss_plus = cross_entropy(forward(X, p64)[0], Y)
            p64[k][idx] = orig - eps
            loss_minus = cross_entropy(forward(X, p64)[0], Y)
            p64[k][idx] = orig

            num = (loss_plus - loss_minus) / (2 * eps)
            ana = grads[k][idx]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            print(f"{k}{idx}  numeric={num:.3e}  analytic={ana:.3e}  rel_err={rel:.1e}")
