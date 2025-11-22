import os
import pickle
import numpy as np
import scipy.io as sio
from emg_hgc.utils.labeling import relabel_labels
from emg_hgc.utils.array_ops import resize_arrays
from emg_hgc.data import preprocessing
from emg_hgc.data import windowing
from emg_hgc.features import aggregation


class EMG:
  """
  Container class for EMG data, including voltage signals, class labels,
  and repetitions. Supports loading, saving, and basic manipulation.

  Variables:
    - emg (np.ndarray): EMG input signal 
    - stimulus (np.ndarray): Gesture class
    - repetition (np.ndarray): Gesture class repetition
    - windowedData (list): List of lists indexed [Stimulus][Repetition][Volatage Sample][Channel]
    - afm (np.ndarray): Uniform `Aggregated Feature Matrices` indexed [Stimulus][Repetition][Feature][Aggregated Volatage Sample][Channel]
    - x 
  """

  def __init__(self, emg=None, stimulus=None, repetition=None, metadata=None):
    """
    Initialize an EMG dataset object.
    """
    self.emg = emg
    self.stimulus = stimulus
    self.repetition = repetition
    self.metadata = metadata or {}

  # ----------------------------
  # Constructors / Loaders
  # ----------------------------
  @classmethod
  def from_mat(self, filepath, attributes=('emg', 'stimulus', 'repetition')):
    """
    Load EMG data from a MATLAB .mat file.

    Parameters:
      filepath : str
        Path to the .mat file.
      attributes : tuple of str
        Keys to extract from the file (default: ('emg', 'restimulus', 'rerepetition')).
    
    Returns:
      EMG: An EMG object instance.
    """
    if not os.path.isfile(filepath):
      raise FileNotFoundError(f"File not found: {filepath}")

    data = sio.loadmat(filepath)
    try:
      emg = np.array(data[attributes[0]])
      stim = np.array(data[attributes[1]]).flatten()
      rep = np.array(data[attributes[2]]).flatten()
    except KeyError as k:
      raise KeyError(f"Missing expected key in MATLAB file: {k}")

    stim = relabel_labels(stim)
    emg, stim, rep = resize_arrays([emg, stim, rep])

    metadata = {"source": filepath, "attributes": attributes}
    return self(emg, stim, rep, metadata)

  @classmethod
  def from_pickle(self, filepath):
    """
    Load a previously pickled EMG dataset.
    """
    with open(filepath, "rb") as f:
      data = pickle.load(f)
    return self(**data)

  @classmethod 
  def from_XY_npy(self, xpath, ypath, segmentation='NA', features = 'NA'):
    self.x = np.load(xpath)
    self.y = np.load(ypath)
    meta = {
      'sources':[xpath, ypath],
      'aggregation':{'segmentation': segmentation, 'features':features}
    }
    emg = self.x
    stm = self.y
    rep = np.zeros_like(stm)

    seen = {}
    for i, s in enumerate(stm):
      rep[i] = seen.get(s, 0)
      seen[s] = seen.get(s, 0) + 1

    return self(emg, stm, rep, meta)


  # ----------------------------
  # Instance Methods
  # ----------------------------
  def __len__(self):
    """Return number of samples."""
    return len(self.emg) if self.emg is not None else 0

  def __repr__(self): 
    return f"EMG dataset: {len(self)} samples, {self.emg.shape[-1] if self.emg is not None else '?'} channels"
  
  def remove_channels(self, channels):
    self.emg = np.delete(self.emg, channels, axis=1)

  def to_dict(self):
    """Return the dataset as a dictionary."""
    return {
      "emg": self.emg,
      "stimulus": self.stimulus,
      "repetition": self.repetition,
      "metadata": self.metadata,
    }

  def to_xy(self):
    """Returns [x, y] np.ndarrays for use in models."""
    if not hasattr(self, 'afm'):
      raise ValueError(f"AFM not available. See {self.get_afm}.")
    
    a = self.afm.copy()
    nStm = a.shape[0] # Number to stimuli 
    nRep = a.shape[1] # Number of repetitions
    y = np.ones(nStm*nRep) 
    for i in range(nStm):
      for j in range(nRep):
        y[i*nRep+j] = i
  
    x = self.afm.reshape(a.shape[0]*a.shape[1], *a.shape[2:])
    self.x = x
    self.y = y
    return [x, y]

  def save_xy(self, savepath):
    if (not hasattr(self, 'x')) or (not hasattr(self, 'y')):
      raise ValueError(f"x or y not available. See {self.to_xy}.")
    
    print(f'Saving x to: {savepath}x.npy')
    np.save(f'{savepath}x.npy', self.x)
    print(f'Saving y to: {savepath}y.npy')
    np.save(f'{savepath}y.npy', self.y)


  def save_pickle(self, savepath):
    """Save dataset as a pickle file."""
    try:
      with open(savepath, "wb") as f:
        pickle.dump(self.to_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)
      print(f"Saved EMG dataset to {savepath}")
    except Exception as x:
      print(f"Error saving dataset: {x}")

  def set_data(self, emg, stimulus, repetition, metadata=None):
    """Manually assign data arrays."""
    self.emg, self.stimulus, self.repetition = resize_arrays([emg, stimulus, repetition])
    if metadata:
      self.metadata.update(metadata)

  def summary(self):
    """Print a brief summary of the dataset."""
    print("EMG Dataset Summary")
    print("--------------------")
    print(f"Samples: {len(self.emg)}")
    print(f"Channels/Features: {self.emg.shape[-1] if self.emg is not None else 'N/A'}")
    print(f"Unique stimuli: {len(np.unique(self.stimulus)) if self.stimulus is not None else 'N/A'}")
    print(f"Unique repetitions: {len(np.unique(self.repetition)) if self.repetition is not None else 'N/A'}")
    if hasattr(self, 'windowedData'): print(f"Windowed Data Available")
    if hasattr(self, 'afm'): 
      print(f"Aggregated Data Shape: {self.afm.shape}")
    if hasattr(self, 'x'):
      print(f"X Available: {self.x.shape}")
    if hasattr(self, 'y'):
      print(f"Y Available: {self.y.shape}")

    if hasattr(self, 'metadata'):
      print("Metadata:")
      for key, value in self.metadata.items():
        print(f"  {key.capitalize()}: {value}")

  # ----------------------------
  # Utility Methods
  # ----------------------------
  def apply_process(self, rectify=False, normalize=False, **kwargs):
    self.metadata = {
      **getattr(self, "metadata", {}),
      "Preprocess": {
        "rectified": rectify,
        "normalized": normalize,
      }
    }

    if rectify:
      self.emg = preprocessing.rectify(self.emg)
    if normalize:
      self.emg = preprocessing.apply_normalize(self.emg, **kwargs)

  def remove_stimuli(self, remove=None, balance=False, random_state=None):
    """
    Remove or balance specific stimulus classes in the EMG data.

    Parameters:
      remove (list of int) optional:
        List of stimulus labels to remove entirely (e.g., [0] to remove 'rest').
      balance (bool) default=False:
        If True, balances the number of samples across remaining classes 
        by downsampling to the smallest class size.
      random_state (int) optional:
        Random seed for reproducibility during balancing.

    Returns:
      EMG: A new EMG object with filtered (and optionally balanced) data.
    """
    if self.emg is None or self.stimulus is None:
      raise ValueError("EMG data not loaded.")

    emg = self.emg.copy()
    stm = self.stimulus.copy()
    rep = self.repetition.copy() 

    # Remove specified stimuli
    if remove is not None and len(remove) > 0:
      mask = ~np.isin(stm, remove)
      emg, stm = emg[mask], stm[mask]
      if rep is not None:
        rep = rep[mask]

    # Balance remaining classes
    if balance:
      rng = np.random.default_rng(random_state)
      unique_classes, counts = np.unique(stm, return_counts=True)
      min_count = counts.min()

      indices = np.hstack([
        rng.choice(np.where(stm == cls)[0], min_count, replace=False)
        for cls in unique_classes
      ])
      indices = np.sort(indices)

      emg, stm = emg[indices], stm[indices]
      if rep is not None:
        rep = rep[indices]

    stm = relabel_labels(stm)

    # Return a new EMG instance
    filtered = {
      "emg": emg,
      "stimulus": stm,
      "repetition": rep,
      "metadata": {
        **getattr(self, "metadata", {}),
        "filter": {
          "removed": remove,
          "balanced": balance,
          "random_state": random_state
        }
      }
    }
    return EMG(**filtered)

  def get_windowedData(self, mode:str="by_label", **kwargs):
    """
    Window the EMG signal according to the specified mode.

    Parameters:
      mode (str): Windowing mode, one of:
        'by_label'     – split by stimulus transitions (A)
        'by_label_fixed'   – fixed window length per stimulus (B) (Not implemented)
        'changepoint'  – variable windowing via change-point detection (C) (Not implemented)
        'changepoint_fixed' – fixed size from changepoint starts (D) (Not implemented)
        'rolling'      – sliding/rolling window (E) (Not implemented)
      kwargs (dict): Parameters passed to the selected windowing function.

    Returns:
      Windowed Data (list of lists): data[stimulus][repetition][voltage sample][channel]
    """
    if self.emg is None or self.stimulus is None:
      raise ValueError("EMG data or stimulus labels not loaded.")

    # Dispatch to appropriate windowing function
    self.windowedData = windowing.apply_windowing(self.emg, self.stimulus, self.repetition, mode=mode, **kwargs)
    self.metadata.update({'windowed':mode})

  def get_afm(self, mode:str='dynamic', fems:list=[], **kwargs):
    """
    Get the aggregated feature matrices (afm) by segmenting the windowed EMG signal according to the specified mode and feature extraction methods.

    Parameters:
      mode (str): Segmentation mode, one of:
        'Dynamic'     – Using N = x(n+1)+y(n), evenly distribute voltage samples between segments (A)
        'Static'   – Given a fixed window size, evenly distribute voltage samples between segments (B)
      kwargs (dict): Parameters passed to the selected windowing function.

    Returns:
      Aggregated feature matrices (np.ndarray): data[stimulus][repetition][feature method][aggregated voltage sample][channel]
    """
    if not hasattr(self, 'windowedData'):
      raise ValueError("EMG data has not been windowed. See aggregation.py.")
  
    self.afm = aggregation.apply_aggregation(self.windowedData, fems, mode, **kwargs)
    fem_names = []
    for f in fems:
      fem_names.append(f.__name__)
    self.metadata.update({'aggregation':{'segmentation': mode, 'features':fem_names}})




    
