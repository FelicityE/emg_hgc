# import numpy as np
# def relabel(label:np.ndarray, strt:int=0):
#   """
#   Ensures that Classes labels start at `strt`.
#   Parameters:
#     label (ndarray): an array of class labels
#     strt (int): starting integer for all classes.
#   Returns: label
#   """
#   lablst = np.unique(label)
#   for i in range(strt,len(lablst)):
#     label[label == lablst[i]] = i;
  
#   return label

import numpy as np

def relabel_labels(labels, start=0):
  """
  Reindex class labels so they start from `start`.
  Parameters:
    labels : np.ndarray
        Input labels.
    start : int
        Starting label index.
  Returns
    np.ndarray
  """
  unique_labels = np.unique(labels)
  for i, lbl in enumerate(unique_labels):
    labels[labels == lbl] = i + start
  return labels
