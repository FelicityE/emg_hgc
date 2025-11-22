import numpy as np

###############################################################################
# Winding functions
############################################################################### 

def apply_windowing(
  emg: np.ndarray, stim: np.ndarray, rep: np.ndarray,
  mode: str = "by_label", **kwargs
):
  """
  Dispatch to the appropriate windowing method.

  Returns:
  - dict with keys ['emg', 'stimulus', 'repetition', 'metadata']
  """
  if mode == "by_label":
    return window_by_label(emg, stim, rep, **kwargs)
  elif mode == "by_label_fixed":
    return window_by_label_fixed(emg, stim, rep, **kwargs)
  elif mode == "changepoint":
    return window_changepoint(emg, stim, rep, **kwargs)
  elif mode == "changepoint_fixed":
    return window_changepoint_fixed(emg, stim, rep, **kwargs)
  elif mode == "rolling":
    return rolling_window(emg, stim, rep, **kwargs)
  else:
    raise ValueError(f"Unknown windowing mode '{mode}'.")

#######################################
# A) Window by label transitions
#######################################
def window_by_label(emg, stm, rep, replace=True, repStart=1, report=False):
  """
  Segments continuous EMG signal by contiguous regions of identical stimulus labels.
  Each window corresponds to one gesture occurrence (start/end of same label).

  Returns:
    Data List (list of lists): data[stimulus][repetition][voltage sample][channel]
  """
  dataList = [];
  rmList = []
  stmCnt = 0;

  for s in np.unique(stm):
    stmList = [];
    repCnt = 0;
    for r in np.unique(rep)[repStart:]:
      segment = emg[(stm.flatten() == s) & (rep.flatten() == r)]
      if np.shape(segment)[0] == 0:
        rmList.append([stmCnt, repCnt]);
      
      
      stmList.append(segment)
      repCnt+=1
    
    dataList.append(stmList);
    stmCnt+=1

  if rmList and replace:
    if report:
      print("WARNING: The following [Stimulus, Repetitions] are empty: ")
      print(rmList)
      print("Replacing with similar repetition...")

    for seg in sorted(rmList, reverse=[True]):
      try:
        original_matrix = dataList[seg[0]][seg[1]-1]
      except:
        original_matrix = dataList[seg[0]][seg[1]+1]

      # 1. Define noise parameters
      mu = 0  # Mean of the Gaussian distribution (center of the noise)
      std_dev = 0.5  # Standard deviation (intensity or spread of the noise)

      # 2. Generate Gaussian noise with the same shape as the original matrix
      noise = np.random.normal(mu, std_dev, size=original_matrix.shape)

      # 3. Add the noise to the original matrix
      dataList[seg[0]][seg[1]] = original_matrix + noise
        

  return dataList

#######################################
# Placeholders for B, C, D, and E (can be expanded later)
#######################################
def window_by_label_fixed(emg, stim, rep, window_size=8000):
  raise NotImplementedError("Fixed windowing not yet implemented.")

def window_changepoint(emg, stim, rep, **kwargs):
  raise NotImplementedError("Change-point detection windowing not yet implemented.")

def window_changepoint_fixed(emg, stim, rep, **kwargs):
  raise NotImplementedError("Change-point fixed windowing not yet implemented.")

def rolling_window(emg, stim, rep, window_size=8000, stride=50):
  raise NotImplementedError("Rolling windows not yet implemented.")
