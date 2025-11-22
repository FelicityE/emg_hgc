import numpy as np

#######################################
# Mean Based Methods
#######################################
def mean(signal, axis=0, **kwargs):
  if signal.ndim < 2:
    return signal
  elif signal.shape[axis] < 2:
    return signal
  else:
    return np.mean(signal, axis=axis)
  # mu = np.mean(signal, axis=axis)
  # if len(mu) != signal.shape[axis]:
  #   return signal
  # else:
  #   return mu

#######################################
# Frequency based methods
#######################################
# Aggregated Fast Fourier Transform
# Numpy.fft.fft https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html
def fft(signal, nAVS=200, **kwargs):
  if nAVS:
    rtn = np.fft.fft(signal, n=nAVS, axis=0);
  else:
    rtn = np.fft.fft(signal, axis=0);
  return np.abs(rtn)

def mnf(signal, sampling_rate=2000, **kwargs):
  """
  Calculate the Mean Frequency (MNF) for each channel in an EMG signal using the Fourier Transform.

  Parameters:
      signal (ndarray): An n x m ndarray where n is the signal length in time, and m is the number of channels.
      sampling_rate (float): The sampling rate of the EMG signal in Hz.

  Returns:
      ndarray: A 1D array of length m containing the MNF for each channel.
  """
  # Ensure the input is a numpy array
  signal = np.asarray(signal)

  # Check that the input is 2D
  if signal.ndim != 2:
      raise ValueError("Input signal must be a 2D array with dimensions n x m (time x channels).")

  # Number of channels
  num_channels = signal.shape[1]
  n_samples = signal.shape[0]

  # Frequency axis
  freqs = np.fft.rfftfreq(n_samples, d=1/sampling_rate)

  # Initialize array to store MNF for each channel
  mnf = np.zeros(num_channels)

  # Compute MNF for each channel
  for channel in range(num_channels):
    # Compute the Fourier Transform and corresponding power spectrum
    fft_vals = np.fft.rfft(signal[:, channel])
    psd = np.abs(fft_vals) ** 2  # Power spectral density

    # Compute the mean frequency
    mnf[channel] = np.sum(freqs * psd) / np.sum(psd)

  return mnf


#######################################
# Ratio / Abstracted Methods
#######################################
def zcr(signal, **kwargs):
  """
  Calculate the Zero Crossing Rate (ZCR) for each channel in an EMG signal.

  Parameters:
  - signal (ndarray): An n x m ndarray where n is the signal length in time, and m is the number of channels.

  Returns:
  - ndarray: A 1D array of length m containing the ZCR for each channel.
  """
  # Ensure the input is a numpy array
  signal = np.asarray(signal)

  # Check that the input is 2D
  if signal.ndim != 2:
      raise ValueError("Input signal must be a 2D array with dimensions n x m (time x channels).")

  # Compute the sign of the signal (1 for positive, -1 for negative, 0 for zero)
  sign_changes = np.diff(np.sign(signal), axis=0)

  # Identify zero crossings (nonzero entries in sign_changes indicate a crossing)
  zero_crossings = np.count_nonzero(sign_changes, axis=0)

  # Compute the Zero Crossing Rate for each channel
  zcr = zero_crossings / signal.shape[0]

  return zcr

def tap(signal, channel_mod = 1, weights_mod = 1, **kwargs):
  """
  Target Activation Projection (TAP)

  Parameters:
  - signal (ndarray): Input data array [nSamples, nChannels].
  - channel_mod (optional): modify number of channels set to 1 (default 1).
  - weights_mod (optional): modify channel scaling (default 1).

  Returns:
  - ndarray: TAP values for the data.
  """
  if signal.ndim > 1:
    colSum = np.sum(abs(signal), axis=0);
  else:
    colSum = signal

  idx = np.argsort(colSum)
  tap = np.zeros(np.shape(colSum));

  nCol = len(idx)
  for i in range(nCol):
    if i <= channel_mod:
     tap[idx[i]] = 1;
    else:
      tap[idx[i]] = (nCol-i)/(nCol*weights_mod);
    
  return tap