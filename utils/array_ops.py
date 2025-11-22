# import numpy
# def resize(attributes:list):
#   """
#   Ensures all attributes ndarrays are the same size and adjusts as needed.
#   Parameters: attributes [list]: list of ndarrays
#   Returns: attributes 
#   """
#   sizes = [];
#   for i in range(len(attributes)):
#     sizes.append(len(attributes[i]));
#   resz = min(sizes)
#   for i in range(len(attributes)):
#     attributes[i] = attributes[i][:resz]

#   return attributes

import numpy as np

def resize_arrays(arrays):
  """
  Truncate all arrays to the minimum common length along the first dimension.
  Parameters:
    arrays : list of np.ndarray
      Arrays to resize.
  Returns
    list of np.ndarray
  """
  min_len = min(len(arr) for arr in arrays)
  return [arr[:min_len] for arr in arrays]



def pad(ndArray, N, padMode = 'linear_ramp'):
  rtn = np.zeros(np.shape(ndArray))
  rtn[:,:] = ndArray;
  
  nAdd = N-len(rtn);
  if N < len(rtn):
    raise ValueError("Input signal should be less than padded signal.")
  addTop = int(np.floor(nAdd/2))
  addBot = int(np.ceil(nAdd/2))

  return np.pad(rtn,((addTop, addBot),(0,0)),mode=padMode)