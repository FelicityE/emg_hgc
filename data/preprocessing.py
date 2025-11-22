import numpy as np
from sklearn.preprocessing import normalize

def rectify(emg: np.ndarray):
  """
  Rectifies data using the absolute value.
  """
  return abs(emg)


def apply_normalize(emg, mode='all', axis=0, **kwargs):
  """
  Dispatch to the appropriate normalization method.
  """
  if mode == 'all':
    return normalize(emg, axis=axis, **kwargs)
  elif mode == 'by_window':
    normalize_by_window(emg, axis=axis, **kwargs)
  else:
    raise ValueError(f"Unknown normalization mode '{mode}'.")

def normalize_by_window(data:list, axis=0, **kwargs):
  """
  Normalize by window from a list of lists 
  """
  stm = []
  for i in range(len(data)):
    rep = []
    for j in range(len(data[i])):
      rep.append(normalize(data[i][j], axis=axis, **kwargs))
  stm.append(rep)  

  return stm
