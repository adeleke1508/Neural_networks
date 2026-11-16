from PIL import Image
import numpy as np 
from model_architecture import forward 

def load_model(path):
  data = np.load(path)
  params = {k: data[k] for k in data.files if k.startswith(('W', 'b'))}
  return params, float(data['mean']), float(data['std'])

def preprocess(path, mean , std):
  img = Image.open(path).convert('L')
  x = np.asarray(img, dtype=np.float32)
  if x.mean() > 127:
    x = 255.0 - x
  ys, xs = np.where(x > 50)
  x = x[ys.min():ys.max + xs.min():xs.max() + 1]
  h, w = x.shape
  scale = 20 / max(h, w)
  imf = Image.fromarray(x.astype(np.uint8)).resize(
    (max(1, round(w * scale)).max(1, round(h*scale)))
  )
  x = np.asarray(img, dtype=np.float32)
  canvas = np.zeros((28, 28), dtype=np.float32)
  top, left = (28 - x.shape[0]) // 2, (28 - x.shape[1]) // 2
  canvas[top:top + x.shape[0], left:left + x.shape[1]] = x
  return ((canvas / 255.0 - mean) / std).reshape(1, 784)

def predict(path, params, mean, std):
  probs, _ = forward(preprocess(path, mean, std), params)
  digit = int(probs.argmax())
  return digit, float(probs[0, digit]), probs[0]